"""Shard frozen seed-1 prompts for a second screen of recent SPADE games."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("data/SPADE-all-qwen3"))
    parser.add_argument("--min-step", type=int, default=380)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("Need at least one shard")
    selection = json.loads((args.pool / "selection.json").read_text())
    accepted = {
        Path(game["path_in_dataset"]).stem: game
        for game in selection["accepted_games"]
    }
    rows = [json.loads(line) for line in (args.pool / "eval-seed1.jsonl").open()]
    rows = [row for row in rows if row["metadata"]["step"] >= args.min_step]
    rows.sort(key=lambda row: row["metadata"]["environment"])
    if not rows:
        raise ValueError("No games at this step threshold")
    for row in rows:
        metadata = row["metadata"]
        name = metadata["environment"].removeprefix("SPADE/")
        game = accepted[name]
        if metadata["seed"] != 1 or metadata["game_sha256"] != game["sha256"]:
            raise ValueError(f"Frozen seed-1 prompt differs from catalog: {name}")
    for shard in range(args.shards):
        path = args.pool / f"late-seed1-shard-{shard:02d}.jsonl"
        with path.open("w") as stream:
            for row in rows[shard::args.shards]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": selection["dataset"], "revision": selection["revision"],
        "min_step": args.min_step, "seed": 1, "shards": args.shards,
        "games": [row["metadata"]["environment"] for row in rows],
    }
    (args.pool / "late-seed1-selection.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Prepared {len(rows)} late games across {args.shards} seed-1 shards")


if __name__ == "__main__":
    main()
