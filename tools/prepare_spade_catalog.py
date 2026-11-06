"""Download and validate every pinned SPADE game for a Qwen3 difficulty screen."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spade_games.protocol import actor_prompt
from tools.prepare_spade_games import BASE_URL, DATASET, REVISION, download


def _read_game(game: dict, output: Path, seeds: tuple[int, ...],
               source_dir: Path | None) -> tuple[dict, list[str] | None, str | None]:
    relative = Path(game["path_in_dataset"])
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
        return game, None, "invalid manifest path"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_dir is not None:
        source_path = source_dir / relative
        if not source_path.is_file():
            return game, None, "missing from pinned local checkout"
        shutil.copy2(source_path, destination)
        source = destination.read_bytes()
    elif destination.exists():
        source = destination.read_bytes()
    else:
        for attempt in range(4):
            try:
                source = download(relative.as_posix())
                destination.write_bytes(source)
                break
            except (HTTPError, URLError, TimeoutError) as error:
                if attempt == 3:
                    return game, None, f"download failed: {error}"
                time.sleep(2 ** attempt)
    if not source:
        return game, None, "empty source"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "spade_games.inspect_game", str(destination), *map(str, seeds)],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except subprocess.TimeoutExpired:
        return game, None, "validation timeout"
    if result.returncode:
        return game, None, f"validation failed: {result.stderr[-500:]}"
    try:
        observations = json.loads(result.stdout)
        if len(observations) != len(seeds) or not all(isinstance(x, str) and x for x in observations):
            raise ValueError("reset returned invalid observations")
    except (ValueError, TypeError) as error:
        return game, None, f"invalid reset output: {error}"
    return {**game, "sha256": hashlib.sha256(source).hexdigest()}, observations, None


def prepare(output: Path, workers: int, shards: int, limit: int | None = None,
            source_dir: Path | None = None) -> dict:
    if workers < 1 or shards < 1:
        raise ValueError("workers and shards must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if source_dir is not None:
        source_dir = source_dir.resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != REVISION:
            raise ValueError(f"Local checkout revision {revision} differs from pinned {REVISION}")
        if subprocess.check_output(
            ["git", "-C", str(source_dir), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError("Local dataset checkout has uncommitted changes")
    manifest_path = output / "manifest.json"
    if source_dir is not None:
        shutil.copy2(source_dir / "manifest.json", manifest_path)
    elif not manifest_path.exists():
        manifest_path.write_bytes(download("manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    games = manifest["games"][:limit]
    seeds = (0, 1)
    results = [None] * len(games)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_read_game, game, output, seeds, source_dir): index
                   for index, game in enumerate(games)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    accepted, rejected = [], []
    shard_rows = [[] for _ in range(shards)]
    eval_rows = []
    for game, observations, error in results:
        if error:
            rejected.append({"game_file": game["path_in_dataset"], "step": game["step"],
                             "reason": error})
            continue
        accepted.append(game)
        for seed, observation in zip(seeds, observations):
            row = {
                "user_prompt": actor_prompt(observation, "concise") + " /no_think",
                "metadata": {
                    "environment": f"SPADE/{Path(game['path_in_dataset']).stem}",
                    "game_file": str((output / game["path_in_dataset"]).resolve()),
                    "game_sha256": game["sha256"], "seed": seed, "step": game["step"],
                    "skill": game.get("skill"), "initial_observation": observation,
                    "no_think": True,
                },
            }
            if seed == 0:
                shard_rows[(len(accepted) - 1) % shards].append(row)
            else:
                eval_rows.append(row)
    for index, rows in enumerate(shard_rows):
        with (output / f"scan-shard-{index:02d}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "eval-seed1.jsonl").open("w") as stream:
        for row in eval_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": DATASET, "revision": REVISION, "url": BASE_URL,
        "manifest_games": len(manifest["games"]), "processed_games": len(games),
        "accepted_games": accepted, "rejected_games": rejected,
        "scan_seeds": [0], "held_out_seed": 1, "shards": shards,
        "prompt_style": "concise", "no_think": True,
    }
    (output / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/SPADE-all-qwen3"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Smoke test using the first N manifest games")
    parser.add_argument("--source-dir", type=Path, help="Pinned local Git checkout")
    args = parser.parse_args()
    summary = prepare(args.output, args.workers, args.shards, args.limit, args.source_dir)
    print(f"Prepared {len(summary['accepted_games'])}/{summary['processed_games']} games; "
          f"{len(summary['rejected_games'])} rejected; {summary['shards']} scan shards")


if __name__ == "__main__":
    main()
