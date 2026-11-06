"""Summarize a no-training SPADE probe by game and within-seed variation."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(probe_paths: list[Path], output_dir: Path, expected_repeats: int = 8,
              expected_seeds: int = 2,
              wrapper_errors_path: Path | None = None,
              steps_label: str = "396") -> list[dict]:
    unique = {}
    for probe_path in probe_paths:
        for line in probe_path.open():
            if line.strip():
                row = json.loads(line)
                unique.setdefault((row["environment"], row["seed"], row["repeat"]), row)
    rows = list(unique.values())
    wrapper_errors = []
    if wrapper_errors_path is not None and wrapper_errors_path.exists():
        wrapper_errors = json.loads(wrapper_errors_path.read_text())
        errors_by_key = {(row["environment"], row["seed"], row["repeat"]): row["error"]
                         for row in wrapper_errors}
        for row in rows:
            row["error"] = row.get("error") or errors_by_key.get(
                (row["environment"], row["seed"], row["repeat"])
            )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["environment"]].append(row)
    summaries = []
    for environment, episodes in sorted(grouped.items()):
        valid = [episode for episode in episodes if not episode.get("error")]
        by_seed = defaultdict(list)
        for episode in valid:
            by_seed[episode["seed"]].append(episode["reward"])
        mixed_seeds = sum(max(rewards) > min(rewards) for rewards in by_seed.values())
        full = sum(episode["reward"] >= 1 for episode in valid)
        positive = sum(episode["reward"] > 0 for episode in valid)
        errors = sum(bool(episode.get("error")) for episode in episodes)
        first_turn_caps = sum(episode["stop_reasons"][:1] == ["length"] for episode in episodes)
        n = len(valid)
        expected = expected_seeds * expected_repeats
        if errors:
            status = "error"
        elif len(episodes) < expected:
            status = "incomplete"
        elif full == n:
            status = "saturated"
        elif mixed_seeds:
            status = "mixed"
        elif positive:
            status = "positive, no variance"
        elif any(episode["reward"] < 0 for episode in valid):
            status = "negative reward"
        else:
            status = "zero reward"
        summaries.append({
            "environment": environment,
            "status": status,
            "episodes": len(episodes),
            "valid_episodes": n,
            "errors": errors,
            "mean_reward": sum(episode["reward"] for episode in valid) / n if n else None,
            "positive": positive,
            "full": full,
            "mixed_seeds": mixed_seeds,
            "seeds": len(by_seed),
            "first_turn_caps": first_turn_caps,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    markdown = [
        "# Qwen3-1.7B no-training SPADE scan",
        "",
        (
            f"Saved step(s) {steps_label}; concise boxed-action prompt with `/no_think`; "
            f"{expected_seeds} seeds × {expected_repeats} stochastic plays per game. "
            "The probe uses temperature 0.7, top-p 0.8, a 256-token turn cap, "
            "a 2,048-token episode cap, and a 4,096-token context cap. "
            "A mixed seed has both higher and lower terminal rewards across plays. "
            "The probe uses the same terminal reward rule as training."
        ),
        "",
        "Statuses: " + "; ".join(
            f"{count} {status}" for status, count in sorted(Counter(
                item["status"] for item in summaries
            ).items())
        ) + ".",
        "",
        "| Game | Status | Mean reward | Positive | Full | Mixed seeds | Errors | First-turn caps |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        n = item["valid_episodes"]
        mean = item["mean_reward"]
        markdown.append(
            f"| {item['environment'].removeprefix('SPADE/')} | {item['status']} | "
            f"{mean:.3f} | {item['positive']}/{n} | {item['full']}/{n} | "
            f"{item['mixed_seeds']}/{item['seeds']} | {item['errors']} | "
            f"{item['first_turn_caps']}/{item['episodes']} |"
            if mean is not None else
            f"| {item['environment'].removeprefix('SPADE/')} | {item['status']} | — | "
            f"0/0 | 0/0 | 0/0 | {item['errors']} | "
            f"{item['first_turn_caps']}/{item['episodes']} |"
        )
    markdown += [
        "",
        (
            f"Raw trajectories: {', '.join(f'`{path.name}`' for path in probe_paths)}. "
            "This scan is exploratory; zero observed variance does not prove a game is saturated."
        ),
        "",
    ]
    if wrapper_errors:
        markdown.extend(["Wrapper-caught game errors (recorded in `wrapper-errors.json`):", ""])
        for environment, error in sorted({(row["environment"], row["error"])
                                          for row in wrapper_errors}):
            markdown.append(f"- `{environment.removeprefix('SPADE/')}`: {error}")
        markdown.append("")
    (output_dir / "summary.md").write_text("\n".join(markdown))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, nargs="+",
                        default=[Path("data/SPADE-qwen3-scan/probe-seeds01-r8.jsonl")])
    parser.add_argument("--output-dir", type=Path, default=Path("data/SPADE-qwen3-scan"))
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--wrapper-errors", type=Path)
    parser.add_argument("--steps-label", default="396")
    args = parser.parse_args()
    wrapper_errors_path = args.wrapper_errors or args.output_dir / "wrapper-errors.json"
    summaries = summarize(args.probe, args.output_dir, args.repeats, args.seeds,
                          wrapper_errors_path, args.steps_label)
    for item in summaries:
        print(f"{item['environment']}: {item['status']}, mean={item['mean_reward']}, "
              f"full={item['full']}/{item['valid_episodes']}, "
              f"mixed={item['mixed_seeds']}/{item['seeds']}")


if __name__ == "__main__":
    main()
