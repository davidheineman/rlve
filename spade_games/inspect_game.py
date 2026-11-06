"""Run game validation in a bounded child process during dataset preparation."""

import json
import sys

from spade_games.synthetic_game_env import make_synthetic_env


def main() -> None:
    game_file = sys.argv[1]
    seeds = [int(seed) for seed in sys.argv[2:]]
    env = make_synthetic_env(game_file, max_turns=12, respect_game_max_turns=True)
    try:
        observations = [env.reset(seed=seed)[0] for seed in seeds]
        print(json.dumps(observations))
    finally:
        env.close()


if __name__ == "__main__":
    main()
