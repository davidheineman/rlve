"""Summarize two fixed-seed Qwen3 screens of the latest SPADE games."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("data/SPADE-all-qwen3"))
    args = parser.parse_args()
    pool = args.pool
    selection = json.loads((pool / "late-seed1-selection.json").read_text())
    catalog = json.loads((pool / "selection.json").read_text())
    games = {
        f"SPADE/{Path(game['path_in_dataset']).stem}": game
        for game in catalog["accepted_games"]
    }
    by_game = defaultdict(lambda: defaultdict(dict))
    files = [*sorted(pool.glob("probe-shard-*.jsonl")),
             *sorted(pool.glob("late-seed1-probe-shard-*.jsonl"))]
    for path in files:
        for line in path.open():
            row = json.loads(line)
            name = row["environment"]
            if name in selection["games"] and row["seed"] in (0, 1):
                by_game[name][row["seed"]][row["repeat"]] = row
    records = []
    for name in selection["games"]:
        groups = by_game[name]
        seed0 = list(groups[0].values())
        seed1 = list(groups[1].values())
        rows = seed0 + seed1
        rewards = [row["reward"] for row in rows if not row.get("error")]
        full = sum(reward >= 1.0 for reward in rewards)
        mixed_groups = sum(
            len(groups[seed]) == 4
            and not any(row.get("error") for row in groups[seed].values())
            and len({row["reward"] for row in groups[seed].values()}) > 1
            for seed in (0, 1)
        )
        records.append({
            "game": name.removeprefix("SPADE/"), "step": games[name]["step"],
            "skill": games[name].get("skill"), "seed0_plays": len(seed0),
            "seed1_plays": len(seed1), "seed0_full": sum(r["reward"] >= 1 for r in seed0),
            "seed1_full": sum(r["reward"] >= 1 for r in seed1),
            "mixed_groups": mixed_groups, "full": full,
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "errors": sum(bool(row.get("error")) for row in rows),
            "positive_truncated": sum(row["reward"] > 0 and row["truncated"] for row in rows),
            "first_turn_caps": sum(row.get("stop_reasons", [])[:1] == ["length"] for row in rows),
        })
    complete = [r for r in records if r["seed0_plays"] == r["seed1_plays"] == 4]
    candidates = sorted(
        [r for r in complete if r["errors"] == r["first_turn_caps"] == 0
         and r["mixed_groups"] > 0 and 0 < r["full"] < 8],
        key=lambda r: (-r["step"], abs(r["mean_reward"] - 0.5), r["game"]),
    )
    summary = {
        "revision": selection["revision"], "min_step": selection["min_step"],
        "games": len(records), "complete": len(complete), "probe_files": len(files),
        "complete_counts": {
            "mixed_seed_groups": sum(r["mixed_groups"] > 0 for r in complete),
            "full_on_all": sum(r["full"] == 8 and r["errors"] == 0 for r in complete),
            "zero_on_all": sum(r["mean_reward"] == 0 and r["errors"] == 0 for r in complete),
            "any_error": sum(r["errors"] > 0 for r in complete),
        },
        "records": records, "candidates": [r["game"] for r in candidates],
    }
    (pool / "late-seed1-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    columns = ("game", "step", "skill", "seed0_full", "seed1_full", "mixed_groups",
               "mean_reward", "errors", "positive_truncated", "first_turn_caps",
               "seed0_plays", "seed1_plays")
    with (pool / "late-seed1-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: r[key] for key in columns} for r in records)
    lines = [
        "# Latest SPADE games: two-seed Qwen3 screen", "",
        (f"Pinned revision `{selection['revision']}`. Games saved at step "
         f"{selection['min_step']} or later; four stochastic plays on each of seeds 0 and 1. "
         "A mixed group has different terminal rewards on repeated plays of the same seed."), "",
        (f"Complete: {len(complete)}/{len(records)} games. Counts among complete games: "
         f"{summary['complete_counts']['mixed_seed_groups']} with a mixed group, "
         f"{summary['complete_counts']['full_on_all']} full on every play, "
         f"{summary['complete_counts']['zero_on_all']} zero on every play, "
         f"{summary['complete_counts']['any_error']} with an error."), "",
        "## Mixed full-win candidates", "",
        "| Game | Step | Mean reward | Full seed 0 | Full seed 1 | Mixed groups | Positive truncations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in candidates:
        lines.append(
            f"| {r['game']} | {r['step']} | {r['mean_reward']:.3f} | "
            f"{r['seed0_full']}/4 | {r['seed1_full']}/4 | {r['mixed_groups']}/2 | "
            f"{r['positive_truncated']}/8 |"
        )
    lines += ["", ("Candidates still require multi-seed probing and reward-code review. "
                   "A zero or full result on both fixed seeds does not prove the game is saturated."), ""]
    (pool / "late-seed1-summary.md").write_text("\n".join(lines))
    print(f"{len(complete)}/{len(records)} late games complete; {len(candidates)} candidates")


if __name__ == "__main__":
    main()
