"""Freeze a cross-game SPADE evaluation set from validated catalog games."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spade_games.protocol import actor_prompt


def prepare(source: Path, output: Path, game_names: list[str], seed_start: int,
            seeds_per_game: int) -> dict:
    if not game_names or len(set(game_names)) != len(game_names):
        raise ValueError("Select distinct games")
    if seed_start < 0 or seeds_per_game < 1:
        raise ValueError("Need nonnegative seed start and positive seeds per game")
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Source and output must differ")
    selection = json.loads((source / "selection.json").read_text())
    accepted = {Path(g["path_in_dataset"]).stem: g for g in selection["accepted_games"]}
    missing = sorted(set(game_names) - accepted.keys())
    if missing:
        raise ValueError(f"Games not in validated catalog: {missing}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    games = []
    for name in game_names:
        game = accepted[name]
        relative = Path(game["path_in_dataset"])
        source_file = source / relative
        game_bytes = source_file.read_bytes()
        if hashlib.sha256(game_bytes).hexdigest() != game["sha256"]:
            raise ValueError(f"Catalog game changed: {name}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)
        seeds = list(range(seed_start, seed_start + seeds_per_game))
        result = subprocess.run(
            [sys.executable, "-m", "spade_games.inspect_game", str(destination), *map(str, seeds)],
            capture_output=True, text=True, timeout=120, check=True,
        )
        observations = json.loads(result.stdout)
        if len(observations) != len(seeds) or not all(isinstance(x, str) and x for x in observations):
            raise ValueError(f"Invalid reset observations for {name}")
        repeated = subprocess.run(
            [sys.executable, "-m", "spade_games.inspect_game", str(destination), *map(str, seeds)],
            capture_output=True, text=True, timeout=120, check=True,
        )
        if json.loads(repeated.stdout) != observations:
            raise ValueError(f"Reset observations are not reproducible for {name}")
        for seed, observation in zip(seeds, observations):
            rows.append({
                "user_prompt": actor_prompt(observation, "concise") + " /no_think",
                "metadata": {
                    "environment": f"SPADE/{name}", "game_file": str(destination),
                    "game_sha256": game["sha256"], "seed": seed,
                    "step": game["step"], "skill": game.get("skill"),
                    "initial_observation": observation, "no_think": True,
                },
            })
        games.append(game)
    with (output / "test.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "dataset": selection["dataset"], "revision": selection["revision"],
        "source_pool": str(source), "games": games, "seed_start": seed_start,
        "seeds_per_game": seeds_per_game, "num_prompts": len(rows),
        "prompt_style": "concise", "no_think": True,
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/SPADE-all-qwen3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--game", action="append", required=True)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--seeds-per-game", type=int, default=128)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.game, args.seed_start,
                     args.seeds_per_game)
    print(f"Prepared {result['num_prompts']} fixed prompts from {len(result['games'])} games")


if __name__ == "__main__":
    main()
