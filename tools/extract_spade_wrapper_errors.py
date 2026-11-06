"""Map loader-caught SPADE game errors in Slurm logs back to probe episodes."""

import argparse
import json
import re
from pathlib import Path


def extract(pairs: list[tuple[Path, Path]]) -> list[dict]:
    errors = []
    for log_path, probe_path in pairs:
        episodes = [json.loads(line) for line in probe_path.open() if line.strip()]
        pending = None
        for line in log_path.open(errors="replace"):
            if "ERROR:spade_games.synthetic_game_env:Error in step: " in line:
                pending = line.split("Error in step: ", 1)[1].strip()
            elif pending:
                match = re.match(r"(\d+)/(\d+) (SPADE/\S+) reward=", line)
                if match:
                    episode = episodes[int(match.group(1)) - 1]
                    if episode["environment"] != match.group(3):
                        raise ValueError(f"Log and probe disagree at {log_path}:{match.group(1)}")
                    errors.append({
                        "environment": episode["environment"],
                        "seed": episode["seed"],
                        "repeat": episode["repeat"],
                        "error": pending,
                        "slurm_log": str(log_path),
                    })
                    pending = None
        if pending:
            raise ValueError(f"Unmatched error at end of {log_path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", metavar=("LOG", "PROBE"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = [(Path(log), Path(probe)) for log, probe in args.pair]
    errors = extract(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(errors, indent=2) + "\n")
    print(f"Recorded {len(errors)} wrapper-caught game errors in {args.output}")


if __name__ == "__main__":
    main()
