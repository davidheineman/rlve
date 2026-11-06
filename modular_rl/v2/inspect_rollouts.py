"""Snapshot scored training batches and expose their reward/gradient evidence."""

import argparse
import ast
import concurrent.futures
import hashlib
import json
import random
import re
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from modular_rl.graders import extract_code, grade_code
from modular_rl.prepare import write_json


def read_events(path):
    if not path.exists():
        return []
    # Literal Unicode paragraph separators are valid inside JSON strings. Only
    # physical newline boundaries delimit records; str.splitlines() also splits
    # those Unicode characters and corrupts otherwise valid Tulu examples.
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mean(values):
    return statistics.mean(values) if values else None


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


def annotate(row, source, name):
    metadata, reward = decoded(row["metadata_json"]), decoded(row["reward_json"])
    forcing = metadata.get("_modular_v2_forcing")
    response = row["response"]
    graded = response[forcing["answer_char_start"] :] if forcing else response
    final = graded.rsplit("</think>", 1)[-1].strip()
    task = metadata["task"]
    required = re.findall(
        r"^def (\w+)\(", metadata["messages"][-1]["content"], re.MULTILINE
    )
    functions, code, fingerprint = [], None, None
    if task == "code":
        code = extract_code(graded)
        try:
            tree = ast.parse(code)
            functions = [
                n.name
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            fingerprint = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
        except SyntaxError:
            pass
    tags = []
    if row["correct"]:
        tags.append("passes" if task != "chat" else "RM win")
    elif reward.get("error"):
        tags.append(reward["error"])
    elif task == "code":
        if fingerprint and set(required) - set(functions):
            tags.append("missing required function")
        elif any(reward.get("passes", [])):
            tags.append("some tests pass")
        else:
            tags.append("all tests fail")
    else:
        tags.append("fails" if task != "chat" else "RM loss")
    if row["status"] == "truncated":
        tags.append("truncated")
    return {
        "run": name,
        "task": task,
        "rollout_id": row["rollout_id"],
        "sample_index": row["row_index"],
        "prompt": metadata["messages"],
        "prompt_id": row["prompt_id"],
        "dataset_id": metadata["id"],
        "dataset_key": metadata["key"],
        "target": metadata["target"],
        "response": response,
        "final_answer": final,
        "graded_response": graded,
        "reward": reward["reward"],
        "accuracy": reward["accuracy"],
        "reward_detail": reward,
        "status": row["status"],
        "forced": bool(forcing),
        "sampled_tokens": row["response_tokens"]
        - (forcing["prefix_tokens"] if forcing else 0),
        "required_functions": required,
        "defined_functions": functions,
        "code": code,
        "code_fingerprint": fingerprint,
        "tags": tags,
        "source": str(source),
    }


def inspect(name, directory):
    progress = read_events(directory / "progress.jsonl")
    updates = max(
        (e["completed_updates"] for e in progress if e["phase"] == "train"), default=0
    )
    log = (directory / "train.log").read_text(errors="replace")
    metrics = {}
    for match in re.finditer(r"step (\d+): (\{[^\n]*\})", log):
        try:
            metrics[int(match[1])] = ast.literal_eval(match[2])
        except (ValueError, SyntaxError):
            continue
    groups, samples, problems, disagreements = [], [], [], []
    experiment = directory.parents[1]
    task = directory.name.rsplit("-", 1)[-1]
    selected = {
        row["metadata"]["key"]: row["metadata"]
        for row in read_events(experiment / "selection" / task / "train.jsonl")
    }
    val_keys = {
        row["metadata"]["key"]
        for row in read_events(experiment / "selection" / task / "val.jsonl")
    }
    std_normalization = bool(re.search(r"grpo_std_normalization \.{2,} True", log))
    for path in sorted((directory / "rollouts/train").glob("*.jsonl")):
        rollout_id = int(path.stem)
        if rollout_id >= updates:
            continue
        batch = [annotate(row, path, name) for row in read_events(path)]
        if len(batch) != 8 or len({s["dataset_key"] for s in batch}) != 1:
            problems.append(
                {"file": str(path), "error": "wrong batch size or mixed prompts"}
            )
        for sample in batch:
            key = sample["dataset_key"]
            if (
                key not in selected
                or sample["target"] != selected[key]["target"]
                or sample["prompt"] != selected[key]["messages"]
            ):
                problems.append({"file": str(path), "error": "prompt/target mismatch"})
            if key in val_keys:
                problems.append(
                    {"file": str(path), "error": "validation prompt in training"}
                )
            if task == "chat" and sample["accuracy"] != float(
                sample["reward"] > sample["reward_detail"]["rejected_score"]
            ):
                problems.append({"file": str(path), "error": "incorrect RM win flag"})
        rewards = [s["reward"] for s in batch]
        mixed = len(set(rewards)) > 1
        deviation = statistics.stdev(rewards) if len(rewards) > 1 else 0
        advantages = [
            (x - mean(rewards)) / (deviation + 1e-6)
            if std_normalization
            else x - mean(rewards)
            for x in rewards
        ]
        training = metrics.get(rollout_id, {})
        gradient = training.get("train/grad_norm")
        if gradient is not None and (gradient > 1e-10) != mixed:
            disagreements.append(
                {
                    "rollout_id": rollout_id,
                    "mixed_rewards": mixed,
                    "gradient_norm": gradient,
                }
            )
        for sample, advantage in zip(batch, advantages):
            sample.update(
                reconstructed_group_advantage=advantage,
                group_rewards=rewards,
                gradient_norm=gradient,
                entropy=training.get("train/entropy_loss"),
            )
        group = {
            "run": name,
            "task": task,
            "rollout_id": rollout_id,
            "samples": batch,
            "mixed_rewards": mixed,
            "reward_std": deviation,
            "train_accuracy": mean([s["accuracy"] for s in batch]),
            "gradient_norm": gradient,
            "entropy": training.get("train/entropy_loss"),
            "unique_final_code": len(
                {s["code_fingerprint"] for s in batch if s["code_fingerprint"]}
            )
            if task == "code"
            else None,
        }
        groups.append(group)
        samples.extend(batch)
    generation = {
        (e["phase"], e["rollout_id"]): e
        for e in read_events(directory / "generation.jsonl")
    }
    evaluations = {
        e["completed_updates"]: {
            "updates": e["completed_updates"],
            "pass@1": e["accuracy"],
        }
        for e in progress
        if e["phase"] == "eval"
    }

    def window(rows):
        records = [s for g in rows for s in g["samples"]]
        return {
            "groups": len(rows),
            "mixed_reward_fraction": mean([g["mixed_rewards"] for g in rows]),
            "train_accuracy": mean([s["accuracy"] for s in records]),
            "mean_sampled_tokens": mean([s["sampled_tokens"] for s in records]),
            "entropy": mean([g["entropy"] for g in rows if g["entropy"] is not None]),
            "unique_final_code_per_8": mean(
                [
                    g["unique_final_code"]
                    for g in rows
                    if g["unique_final_code"] is not None
                ]
            ),
        }

    run_id = (directory / "wandb_run_id.txt").read_text().strip()
    summary = {
        "run": name,
        "task": task,
        "directory": str(directory),
        "wandb_id": run_id,
        "wandb_url": "https://wandb.ai/david-heineman/modular-mopd-rl-initial/runs/"
        + run_id,
        "completed_updates": updates,
        "saved_groups": len(groups),
        "saved_responses": len(samples),
        "unique_training_prompts": len({s["dataset_key"] for s in samples}),
        "mixed_reward_groups": sum(g["mixed_rewards"] for g in groups),
        "all_incorrect_groups": sum(g["train_accuracy"] == 0 for g in groups),
        "all_correct_groups": sum(g["train_accuracy"] == 1 for g in groups),
        "groups_with_gradient_log": sum(g["gradient_norm"] is not None for g in groups),
        "nonzero_gradient_groups": sum((g["gradient_norm"] or 0) > 0 for g in groups),
        "reward_gradient_disagreements": disagreements,
        "integrity_errors": problems,
        "validation": list(evaluations.values()),
        "response_tags": dict(Counter(tag for s in samples for tag in s["tags"])),
        "truncated_winners": sum(
            s["status"] == "truncated" and s["accuracy"] == 1 for s in samples
        ),
        "train_truncated_fraction": mean([s["status"] == "truncated" for s in samples]),
        "eval_truncated_fraction": sum(
            e["truncated_responses"]
            for (phase, _), e in generation.items()
            if phase == "eval"
        )
        / sum(
            e["responses"] for (phase, _), e in generation.items() if phase == "eval"
        ),
        "first_100": window(groups[:100]),
        "last_100": window(groups[-100:]),
    }
    return summary, groups


def replay(sample):
    if sample["task"] == "code":
        grade = grade_code(
            sample["graded_response"], sample["target"], diagnostics=True
        )
    else:
        from .math_reward import grade as grade_math

        grade = grade_math(sample["response"], sample["target"]["answer"])
    return {
        "run": sample["run"],
        "rollout_id": sample["rollout_id"],
        "sample_index": sample["sample_index"],
        "saved_accuracy": sample["accuracy"],
        "replayed": grade,
        "matches": grade["accuracy"] == sample["accuracy"],
    }


def select_groups(groups):
    selected = {}

    def pick(label, predicate, reverse=False):
        candidates = reversed(groups) if reverse else groups
        group = next((g for g in candidates if predicate(g)), None)
        if group is not None:
            selected.setdefault(
                group["rollout_id"], {**group, "selection_reasons": []}
            )["selection_reasons"].append(label)

    pick("first training batch", lambda g: True)
    pick("latest completed batch", lambda g: True, True)
    pick(
        "mixed rewards and gradient",
        lambda g: g["mixed_rewards"] and (g["gradient_norm"] or 0) > 0,
        True,
    )
    pick("all eight incorrect", lambda g: g["train_accuracy"] == 0, True)
    pick("all eight correct", lambda g: g["train_accuracy"] == 1, True)
    for tag in (
        "missing required function",
        "some tests pass",
        "SyntaxError",
        "truncated",
    ):
        pick(tag, lambda g, tag=tag: any(tag in s["tags"] for s in g["samples"]))
    pick(
        "truncated answer scored successful",
        lambda g: any(
            s["status"] == "truncated" and s["accuracy"] == 1 for s in g["samples"]
        ),
        True,
    )
    return list(selected.values())


def render(output, summary, groups):
    payload = json.dumps(
        {"summary": summary, "groups": groups}, ensure_ascii=False
    ).replace("<", "\\u003c")
    template = """<!doctype html><html lang="en"><meta charset="utf-8"><title>RL rollout review</title>
<style>body{font:15px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;background:#f8fafc;color:#182638}h1{margin-bottom:5px}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}td,th{text-align:left;padding:9px;border-bottom:1px solid #ddd}select,input{padding:8px;margin:8px 12px 8px 0}details{background:white;border:1px solid #d5dde7;border-radius:7px;margin:10px 0;padding:12px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px ui-monospace,monospace;max-height:600px;overflow:auto;background:#f1f5f9;padding:12px}.ok{color:#047857}.bad{color:#b91c1c}.muted{color:#53657b}.sample{margin-left:18px}label{font-weight:600}</style>
<h1>RL rollout review</h1><p>Full prompts, generated responses, grading targets and optimizer evidence from completed training batches.</p>
<p class="muted">Examples are selected to expose successes and failures; use summary counts for rates. Advantages are reconstructed from saved rewards using the logged GRPO normalization settings. A nonzero gradient establishes an update signal, not improved validation performance.</p>
<div id="stats"></div><label>Run </label><select id="run"><option value="">All runs</option></select><label>Show </label><select id="kind"><option value="">All selected batches</option><option value="zero">No reward variation</option><option value="mixed">Mixed rewards</option><option value="truncated">Contains truncation</option></select><input id="search" placeholder="Search prompt or response"><span id="count"></span><main id="batches"></main>
<script id="data" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),byId=id=>document.getElementById(id);
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const pct=x=>(100*x).toFixed(1)+'%';const table=el('table');table.append(el('thead'));const h=el('tr');['Run','Updates / saved responses','Val start / best / latest','Groups with reward variation','Train / eval truncation'].forEach(x=>h.append(el('th',x)));table.firstChild.append(h);
for(const r of data.summary.runs){const ev=r.validation;const tr=el('tr');const label=el('td');const a=el('a',r.run);a.href=r.wandb_url;label.append(a);tr.append(label,el('td',`${r.completed_updates} / ${r.saved_responses}`),el('td',[ev[0]['pass@1'],Math.max(...ev.map(e=>e['pass@1'])),ev.at(-1)['pass@1']].map(pct).join(' / ')),el('td',`${r.mixed_reward_groups}/${r.saved_groups} (${pct(r.mixed_reward_groups/r.saved_groups)})`),el('td',`${pct(r.train_truncated_fraction)} / ${pct(r.eval_truncated_fraction)}`));table.append(tr);const opt=el('option',r.run);opt.value=r.run;byId('run').append(opt);}byId('stats').append(table);
function detail(parent,title,text){const d=el('details');d.append(el('summary',title),el('pre',text));parent.append(d);}
function render(){const root=byId('batches');root.replaceChildren();let count=0;const q=byId('search').value.toLowerCase();for(const g of data.groups){if(byId('run').value&&g.run!==byId('run').value)continue;const k=byId('kind').value;if(k==='zero'&&g.mixed_rewards||k==='mixed'&&!g.mixed_rewards||k==='truncated'&&!g.samples.some(s=>s.status==='truncated'))continue;if(q&&!JSON.stringify(g).toLowerCase().includes(q))continue;count++;const d=el('details');d.append(el('summary',`${g.run} · update ${g.rollout_id+1} · rewards [${g.samples.map(s=>Number(s.reward.toFixed(3))).join(', ')}] · grad ${g.gradient_norm===null?'unavailable':g.gradient_norm.toFixed(3)} · ${g.selection_reasons.join('; ')}`));detail(d,'Prompt and grading target',JSON.stringify({messages:g.samples[0].prompt,target:g.samples[0].target},null,2));for(const s of g.samples){const b=el('details',undefined,'sample');b.append(el('summary',`Sample ${s.sample_index+1} · ${s.tags.join(', ')} · reward ${s.reward.toFixed(3)} · advantage ${s.reconstructed_group_advantage.toFixed(3)} · ${s.sampled_tokens} tokens`,s.accuracy===1?'ok':'bad'));b.append(el('pre',s.final_answer));detail(b,'Full response including reasoning',s.response);detail(b,'Reward and replayed test diagnostics',JSON.stringify({saved:s.reward_detail,replay:s.replay||null},null,2));const a=el('a','Raw JSONL batch');a.href=s.source;b.append(a);d.append(b);}root.append(d);}byId('count').textContent=count+' batches';}for(const id of ['run','kind','search'])byId(id).addEventListener('input',render);render();</script></html>"""
    (output / "rollouts.html").write_text(template.replace("__PAYLOAD__", payload))
    rows = [
        "# Saved rollout review",
        "",
        f"Snapshot: {summary['created_at']}",
        "",
        "Open [the filterable rollout viewer](rollouts.html) for complete batches, targets, rewards and test diagnostics.",
        "",
        "| Run | Completed updates | Saved responses | Initial / best / latest val pass@1 | Mixed-reward groups |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for run in summary["runs"]:
        ev = run["validation"]
        values = [ev[0]["pass@1"], max(e["pass@1"] for e in ev), ev[-1]["pass@1"]]
        rows.append(
            f"| [{run['run']}]({run['wandb_url']}) | {run['completed_updates']} | {run['saved_responses']} | {' / '.join(f'{100 * x:.1f}%' for x in values)} | {run['mixed_reward_groups']} / {run['saved_groups']} |"
        )
    rows += [
        "",
        "The batch excerpts below are diagnostic selections, not a random sample. Full counts and checks are in `summary.json`.",
        "",
    ]
    for group in groups:
        sample = next(
            (s for s in group["samples"] if s["accuracy"] == 0), group["samples"][0]
        )
        if "truncated answer scored successful" in group["selection_reasons"]:
            sample = next(
                s
                for s in group["samples"]
                if s["status"] == "truncated" and s["accuracy"] == 1
            )
        rows += [
            f"## {group['run']}, update {group['rollout_id'] + 1}",
            "",
            "; ".join(group["selection_reasons"]),
            "",
            f"Group rewards: `{group['samples'][0]['group_rewards']}`. Gradient norm: `{group['gradient_norm']}`.",
            "",
            "**Prompt**",
            "",
            sample["prompt"][-1]["content"],
            "",
            "**Final response**",
            "",
            "```text",
            sample["final_answer"],
            "```",
            "",
            "**Grading target**",
            "",
            "```json",
            json.dumps(sample["target"], ensure_ascii=False, indent=2),
            "```",
            "",
            "**Saved reward / replay**",
            "",
            "```json",
            json.dumps(
                {"saved": sample["reward_detail"], "replay": sample.get("replay")},
                indent=2,
            ),
            "```",
            "",
            f"[Raw eight-response batch]({sample['source']})",
            "",
        ]
    (output / "rollout-review.md").write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="LABEL=/path/to/model-task"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-per-run", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, selected, candidates = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [
            pool.submit(inspect, name, Path(directory).resolve())
            for name, directory in (spec.split("=", 1) for spec in args.run)
        ]
        inspected = [job.result() for job in jobs]
    for summary, groups in inspected:
        summaries.append(summary)
        chosen = select_groups(groups)
        selected.extend(chosen)
        if summary["task"] in ("code", "math"):
            items = [s for g in groups for s in g["samples"]]
            sample = random.Random(42).sample(
                items, min(args.replay_per_run, len(items))
            )
            sample.extend(s for g in chosen for s in g["samples"])
            candidates.extend(
                {(s["rollout_id"], s["sample_index"]): s for s in sample}.values()
            )
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "run",
                        "completed_updates",
                        "saved_responses",
                        "mixed_reward_groups",
                        "reward_gradient_disagreements",
                        "integrity_errors",
                    )
                }
            ),
            flush=True,
        )
    results = []
    code = [s for s in candidates if s["task"] == "code"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results.extend(pool.map(replay, code))
    math = [s for s in candidates if s["task"] == "math"]
    if math:
        import multiprocessing

        from .math_reward import initialize_worker

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=4,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
        ) as pool:
            results.extend(pool.map(replay, math))
    indexed = {(r["run"], r["rollout_id"], r["sample_index"]): r for r in results}
    for group in selected:
        for sample in group["samples"]:
            sample["replay"] = indexed.get(
                (sample["run"], sample["rollout_id"], sample["sample_index"])
            )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "runs": summaries,
        "replay": {
            "responses": len(results),
            "code_responses": len(code),
            "math_responses": len(math),
            "mismatches": [r for r in results if not r["matches"]],
        },
        "sampling": "64 random responses per verifiable run plus all responses from the explicitly selected diagnostic batches",
    }
    write_json(args.output / "summary.json", report)
    write_json(args.output / "replay.json", results)
    write_json(args.output / "selected-batches.json", selected)
    render(args.output, report, selected)
    print(
        json.dumps({"output": str(args.output), "replay": report["replay"]}), flush=True
    )


if __name__ == "__main__":
    main()
