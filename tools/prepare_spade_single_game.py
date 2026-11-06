"""Expand one pinned final-step SPADE game into a seeded training pool."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spade_games.protocol import actor_prompt


def prepare(source: Path, output: Path, game_name: str, train_seeds: int,
            eval_seeds: int, no_think: bool = False, test_seeds: int = 0) -> dict:
    if train_seeds < 1 or eval_seeds < 1 or test_seeds < 0:
        raise ValueError("Need positive train and eval seed counts and nonnegative test seeds")
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Source and output must differ")
    manifest = json.loads((source / "selection.json").read_text())
    matches = [g for g in manifest["accepted_games"]
               if Path(g["path_in_dataset"]).stem == game_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one game named {game_name}, found {len(matches)}")
    game = matches[0]
    rel_path = Path(game["path_in_dataset"])
    source_game = source / rel_path
    game_bytes = source_game.read_bytes()
    game_hash = hashlib.sha256(game_bytes).hexdigest()
    if game_hash != game["sha256"]:
        raise ValueError("Game file hash differs from pinned manifest")
    destination = output / rel_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_game, destination)
    seeds = list(range(train_seeds + eval_seeds + test_seeds))
    result = subprocess.run(
        [sys.executable, "-m", "spade_games.inspect_game", str(destination),
         *map(str, seeds)],
        capture_output=True, text=True, timeout=120, check=True,
    )
    observations = json.loads(result.stdout)
    if len(observations) != len(seeds) or not all(isinstance(x, str) and x for x in observations):
        raise ValueError("Game reset returned invalid observations")
    repeated = subprocess.run(
        [sys.executable, "-m", "spade_games.inspect_game", str(destination), *map(str, seeds)],
        capture_output=True, text=True, timeout=120, check=True,
    )
    if json.loads(repeated.stdout) != observations:
        raise ValueError("Game reset observations are not reproducible across processes")
    splits = [("train.jsonl", 0, train_seeds),
              ("eval.jsonl", train_seeds, train_seeds + eval_seeds)]
    if test_seeds:
        splits.append(("test.jsonl", train_seeds + eval_seeds, len(seeds)))
    for name, first, last in splits:
        with (output / name).open("w") as stream:
            for seed in range(first, last):
                observation = observations[seed]
                prompt = actor_prompt(observation, "concise")
                if no_think:
                    prompt += " /no_think"
                row = {
                    "user_prompt": prompt,
                    "metadata": {
                        "environment": f"SPADE/{game_name}",
                        "game_file": str(destination),
                        "game_sha256": game_hash,
                        "seed": seed,
                        "step": game["step"],
                        "skill": game.get("skill"),
                        "initial_observation": observation,
                        "no_think": no_think,
                    },
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": manifest["dataset"], "revision": manifest["revision"],
        "game": game, "source_pool": str(source), "train_seeds": train_seeds,
        "eval_seeds": eval_seeds, "test_seeds": test_seeds,
        "seed_ranges": {name: [first, last] for name, first, last in splits},
        "prompt_style": "concise",
        "no_think": no_think,
    }
    (output / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/SPADE-concise"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--game-name", default="game_03298_012_pattern_recognition")
    parser.add_argument("--train-seeds", type=int, default=800)
    parser.add_argument("--eval-seeds", type=int, default=32)
    parser.add_argument("--test-seeds", type=int, default=0)
    parser.add_argument("--no-think", action="store_true")
    args = parser.parse_args()
    summary = prepare(args.source, args.output, args.game_name,
                      args.train_seeds, args.eval_seeds, args.no_think, args.test_seeds)
    print(f"Prepared {summary['game']['filename']}: "
          f"{summary['train_seeds']} train seeds, {summary['eval_seeds']} eval seeds, "
          f"{summary['test_seeds']} test seeds")


if __name__ == "__main__":
    main()
