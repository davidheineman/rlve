"""Summarize the full-catalog Qwen3 difficulty screen and shortlist late games."""

import argparse
import ast
import csv
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path


def literal_solution(path: Path) -> bool:
    """Flag a static reference answer for manual review, without judging its validity."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "solution":
            returns = [child for child in ast.walk(node) if isinstance(child, ast.Return)]
            return len(returns) == 1 and isinstance(returns[0].value, ast.Constant) \
                and isinstance(returns[0].value.value, str)
    return False


def summarize(pool: Path, repeats: int) -> dict:
    selection = json.loads((pool / "selection.json").read_text())
    probes = sorted(pool.glob("probe-shard-*.jsonl"))
    by_game = defaultdict(dict)
    for path in probes:
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            by_game[row["environment"]][(row["seed"], row["repeat"])] = row
    records = []
    for game in selection["accepted_games"]:
        name = Path(game["path_in_dataset"]).stem
        rows = list(by_game.get(f"SPADE/{name}", {}).values())
        valid = [row for row in rows if not row.get("error")]
        rewards = [row["reward"] for row in valid]
        full = sum(reward >= 1.0 for reward in rewards)
        partial = sum(0 < reward < 1.0 for reward in rewards)
        errors = len(rows) - len(valid)
        if len(rows) < repeats:
            status = "incomplete"
        elif errors:
            status = "error"
        elif full == repeats:
            status = "full on every play"
        elif len(set(rewards)) > 1:
            status = "mixed"
        elif partial:
            status = "partial, flat"
        elif rewards and rewards[0] < 0:
            status = "negative, flat"
        else:
            status = "zero on every play"
        records.append({
            "game": name, "step": game["step"], "skill": game.get("skill"),
            "path_in_dataset": game["path_in_dataset"], "status": status,
            "episodes": len(rows), "errors": errors, "full": full, "partial": partial,
            "error_messages": sorted({row["error"] for row in rows if row.get("error")}),
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "reward_values": sorted(set(rewards)),
            "first_turn_caps": sum(row.get("stop_reasons", [])[:1] == ["length"] for row in rows),
            "positive_truncated": sum(row["reward"] > 0 and row["truncated"] for row in rows),
            "mean_turns": sum(row["turns"] for row in rows) / len(rows) if rows else None,
            "literal_solution": literal_solution(pool / game["path_in_dataset"]),
        })
    bands = ((0, 99), (100, 199), (200, 299), (300, 396))
    band_counts = {}
    rejected_band_counts = {}
    for low, high in bands:
        band = f"{low}-{high}"
        band_counts[band] = dict(Counter(
            record["status"] for record in records if low <= record["step"] <= high
        ))
        rejected_band_counts[band] = sum(
            low <= game["step"] <= high for game in selection["rejected_games"]
        )
    complete = [r for r in records if r["status"] != "incomplete"]
    shortlist = sorted(
        [r for r in complete if r["step"] >= 300 and r["status"] == "mixed" and r["errors"] == 0
         and r["first_turn_caps"] == 0 and 0 < r["full"] < repeats],
        key=lambda r: (-r["step"], abs(r["mean_reward"] - 0.5), r["game"]),
    )
    summary = {
        "dataset": selection["dataset"], "revision": selection["revision"],
        "manifest_games": selection["manifest_games"],
        "validated_games": len(records), "rejected_games": len(selection["rejected_games"]),
        "probe_shards": len(probes), "expected_shards": selection["shards"],
        "expected_repeats": repeats, "completed_games": len(complete),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "step_band_counts": band_counts, "rejected_band_counts": rejected_band_counts,
        "records": records,
        "shortlist": [r["game"] for r in shortlist],
    }
    (pool / "difficulty-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    columns = ("game", "step", "skill", "status", "mean_reward", "full", "partial",
               "errors", "episodes", "first_turn_caps", "positive_truncated",
               "mean_turns", "literal_solution", "path_in_dataset")
    with (pool / "difficulty-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: record[column] for column in columns} for record in records)
    lines = [
        "# Full SPADE Qwen3-1.7B difficulty screen", "",
        (f"Pinned revision `{selection['revision']}`. Validated {len(records)}/"
         f"{selection['manifest_games']} released games; {len(selection['rejected_games'])} "
         "failed reset validation. Each completed game has four stochastic plays of "
         "seed 0 at temperature 0.7. This is a first screen, not a stable difficulty "
         "estimate; shortlisted games need multi-seed probes and code review."), "",
        f"Probe shards: {len(probes)}/{selection['shards']}; completed games: {len(complete)}/{len(records)}.",
        "", "| Step | Released | Valid | Rejected | Mixed | Full on all | Partial flat | Negative flat | Zero on all | Error | Incomplete |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for band, counts in band_counts.items():
        total = sum(counts.values())
        lines.append(
            f"| {band} | {total + rejected_band_counts[band]} | {total} | "
            f"{rejected_band_counts[band]} | {counts.get('mixed', 0)} | "
            f"{counts.get('full on every play', 0)} | {counts.get('partial, flat', 0)} | "
            f"{counts.get('negative, flat', 0)} | "
            f"{counts.get('zero on every play', 0)} | {counts.get('error', 0)} | "
            f"{counts.get('incomplete', 0)} |"
        )
    lines += ["", "## Late-step mixed full-win candidates", "",
              ("These have both full wins and lower rewards across four plays of one seed. "
               "A fixed scripted solution or reward bug can still make them unsuitable. "
               "A literal reference answer flags code for review; it is not proof of triviality."), "",
              "| Game | Step | Skill | Mean reward | Full wins | Positive truncations | Literal reference answer | Reward values |",
              "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |"]
    for r in shortlist[:80]:
        lines.append(
            f"| {r['game']} | {r['step']} | {r['skill'] or 'unknown'} | "
            f"{r['mean_reward']:.3f} | {r['full']}/{repeats} | "
            f"{r['positive_truncated']}/{repeats} | "
            f"{'yes' if r['literal_solution'] else 'no'} | "
            f"{', '.join(f'{x:g}' for x in r['reward_values'])} |"
        )
    lines += ["", (f"Full machine-readable results: `difficulty-summary.json` and `difficulty-summary.csv`. "
                   f"{len(shortlist)} games meet the mixed-full-win screen."), ""]
    (pool / "difficulty-summary.md").write_text("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("data/SPADE-all-qwen3"))
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    result = summarize(args.pool, args.repeats)
    print(f"{result['completed_games']}/{result['validated_games']} games complete; "
          f"{len(result['shortlist'])} mixed full-win candidates")


if __name__ == "__main__":
    main()
