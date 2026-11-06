"""Download a pinned SPADE end-of-run game pool and freeze training prompts."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spade_games.protocol import actor_prompt

DATASET = "spade-rl/SPADE-Environments-Qwen3-30B-Games"
REVISION = "dad8b18457a4c061b411669f5d8d0bb72147d987"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"


def download(path: str) -> bytes:
    with urlopen(f"{BASE_URL}/{path}", timeout=60) as response:
        return response.read()


def prepare(output_dir: Path, last_n_steps: int, seeds_per_game: int,
            prompt_style: str = "spade", only_steps: list[int] | None = None) -> dict:
    if last_n_steps < 1 or seeds_per_game < 2:
        raise ValueError("last_n_steps must be positive and seeds_per_game must be at least 2")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = download("manifest.json")
    manifest = json.loads(manifest_bytes)
    available_steps = {game["step"] for game in manifest["games"]}
    steps = sorted(set(only_steps)) if only_steps else sorted(available_steps)[-last_n_steps:]
    if not set(steps) <= available_steps:
        raise ValueError(f"Requested missing steps: {sorted(set(steps) - available_steps)}")
    selected = [game for game in manifest["games"] if game["step"] in steps]
    output_rows = []
    accepted = []
    rejected = []
    seeds = list(range(seeds_per_game))
    for game in selected:
        rel_path = Path(game["path_in_dataset"])
        if rel_path.is_absolute() or ".." in rel_path.parts or rel_path.suffix != ".py":
            raise ValueError(f"Invalid game path in manifest: {rel_path}")
        local_path = output_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        source = download(rel_path.as_posix())
        local_path.write_bytes(source)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "spade_games.inspect_game", str(local_path), *map(str, seeds)],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except subprocess.TimeoutExpired:
            rejected.append({"game_file": rel_path.as_posix(), "reason": "validation timeout"})
            continue
        if result.returncode:
            rejected.append({"game_file": rel_path.as_posix(), "reason": result.stderr[-1000:]})
            continue
        try:
            observations = json.loads(result.stdout)
            assert len(observations) == len(seeds)
            assert all(isinstance(obs, str) and obs for obs in observations)
        except (ValueError, AssertionError):
            rejected.append({"game_file": rel_path.as_posix(), "reason": "invalid reset output"})
            continue
        game_hash = hashlib.sha256(source).hexdigest()
        accepted.append({**game, "sha256": game_hash})
        for seed, observation in zip(seeds, observations):
            output_rows.append({
                "user_prompt": actor_prompt(observation, prompt_style),
                "metadata": {
                    "environment": f"SPADE/{rel_path.stem}",
                    "game_file": str(local_path.resolve()),
                    "game_sha256": game_hash,
                    "seed": seed,
                    "step": game["step"],
                    "skill": game.get("skill"),
                    "initial_observation": observation,
                },
            })
    if not output_rows:
        raise RuntimeError(f"No valid games at steps {steps}; see {rejected}")
    for name, predicate in (
        ("train.jsonl", lambda row: row["metadata"]["seed"] < seeds_per_game - 1),
        ("eval.jsonl", lambda row: row["metadata"]["seed"] == seeds_per_game - 1),
    ):
        with (output_dir / name).open("w") as stream:
            for row in output_rows:
                if predicate(row):
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": DATASET,
        "revision": REVISION,
        "steps": steps,
        "seeds_per_game": seeds_per_game,
        "prompt_style": prompt_style,
        "accepted_games": accepted,
        "rejected_games": rejected,
        "num_train_prompts": len(output_rows) - len(accepted),
        "num_eval_prompts": len(accepted),
    }
    (output_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/SPADE"))
    parser.add_argument("--last-n-steps", type=int, default=1)
    parser.add_argument("--only-step", type=int, action="append",
                        help="Select specific captured steps instead of --last-n-steps")
    parser.add_argument("--seeds-per-game", type=int, default=8)
    parser.add_argument("--prompt-style", choices=("spade", "concise"), default="spade")
    args = parser.parse_args()
    summary = prepare(args.output_dir, args.last_n_steps, args.seeds_per_game,
                      args.prompt_style, args.only_step)
    print(f"Prepared {len(summary['accepted_games'])} games and {summary['num_train_prompts']} training prompts "
          f"from steps {summary['steps']} ({len(summary['rejected_games'])} rejected)")


if __name__ == "__main__":
    main()
