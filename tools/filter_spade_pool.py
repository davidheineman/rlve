"""Copy selected frozen SPADE games and prompts into a training pool."""

import argparse
import json
import shutil
from pathlib import Path


def filter_pool(source: Path, output: Path, game_list: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    selected = {
        line.strip() for line in game_list.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not selected:
        raise ValueError("Game list is empty")
    manifest = json.loads((source / "selection.json").read_text())
    available = {f"SPADE/{Path(game['path_in_dataset']).stem}" for game in manifest["accepted_games"]}
    unknown = selected - available
    if unknown:
        raise ValueError(f"Games absent from source pool: {sorted(unknown)}")
    if output == source:
        raise ValueError("Output pool must differ from source pool")
    output.mkdir(parents=True, exist_ok=True)
    accepted = []
    for game in manifest["accepted_games"]:
        if f"SPADE/{Path(game['path_in_dataset']).stem}" not in selected:
            continue
        rel_path = Path(game["path_in_dataset"])
        destination = output / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel_path, destination)
        accepted.append(game)
    counts = {}
    for filename in ("train.jsonl", "eval.jsonl"):
        count = 0
        with (source / filename).open() as incoming, (output / filename).open("w") as outgoing:
            for line in incoming:
                row = json.loads(line)
                if row["metadata"]["environment"] not in selected:
                    continue
                old_game = Path(row["metadata"]["game_file"]).resolve()
                row["metadata"]["game_file"] = str((output / old_game.relative_to(source)).resolve())
                outgoing.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        counts[filename] = count
    summary = {**manifest, "accepted_games": accepted, "rejected_games": [],
               "selected_environments": sorted(selected), "source_pool": str(source),
               "num_train_prompts": counts["train.jsonl"],
               "num_eval_prompts": counts["eval.jsonl"]}
    (output / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/SPADE-concise"))
    parser.add_argument("--output", type=Path, default=Path("data/SPADE-frontier"))
    parser.add_argument("--game-list", type=Path,
                        default=Path("configs/spade-qwen15b-frontier.txt"))
    args = parser.parse_args()
    summary = filter_pool(args.source, args.output, args.game_list)
    print(f"Prepared {len(summary['accepted_games'])} games, "
          f"{summary['num_train_prompts']} train prompts, "
          f"{summary['num_eval_prompts']} eval prompts in {args.output}")


if __name__ == "__main__":
    main()
