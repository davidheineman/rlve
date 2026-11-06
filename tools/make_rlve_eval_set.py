"""Build small, fixed eval sets for in-loop evaluation during RLVE training.

slime's eval path (slime/rollout/sglang_rollout.py::eval_rollout_single_dataset) reads a
JSON list of records and scores them with `--rm-type rlve`, which re-creates the problem
from `metadata["environment"]` and `metadata["config"]`. So an eval set is just a frozen
list of generated problems - same format as data/HELD-OUT_ENVIRONMENTS/test_128.json:

    {"user_prompt": <problem text>, "metadata": "<json string with environment + config>"}

Two modes:

  generate   Freeze N problems from one or more *training* environments, spread over a
             range of difficulty levels. Seeds start high (default 10^6) so they never
             collide with the seeds RLVEManager draws during training (it counts up from 0).
             Spanning difficulties matters: training starts at difficulty 0 and ramps itself
             up, so a set frozen at one high level reads 0.0 for the whole run, and one
             frozen at one low level saturates.

  subsample  Take K problems per environment from an existing eval file, e.g. the 50
             held-out environments in data/HELD-OUT_ENVIRONMENTS/test.json. Gives breadth
             (many environments, few rollouts each) for a cheap in-loop generalization check.
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Importable no matter how it is invoked: running "python3 tools/..." puts tools/ on the
# path, not the repo root, and Gym lives at the root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


ANSWER_MARKERS = (r"\boxed{", r"}")


def _survives_json(identifier, config):
    """Does this problem still score correctly after a round trip through JSON?

    An eval set is a file, so every config is JSON-encoded and decoded before the reward
    model rebuilds the problem from it. Some environments keep non-string keys in their
    parameters (BucketSorting's `value2count` is keyed by int), and JSON turns those into
    strings, so the scorer no longer finds them: the reference answer scores as wrong and
    the environment reads a flat 0.0 for the whole run. Training never hits this - there the
    config travels through ray as live Python objects - so it only shows up as an eval that
    refuses to move. Check it here rather than discover it in a wandb chart.

    Returns True if the check passes or cannot be made (no reference answer to test with).
    """
    from Gym.environments import identifier2environment

    reference = config.get("parameter", {}).get("reference_answer")
    if reference is None:
        return True

    problem = identifier2environment[identifier](answer_markers=ANSWER_MARKERS)
    problem.set_config(json.loads(json.dumps(config)))
    try:
        return problem.verifier(r"\boxed{" + str(reference) + "}")["accuracy"] == 1
    except Exception:
        return False


def _records_for_environment(identifier, difficulty, num_problems, seed_start, param_seed, seen_prompts=None, allow_fewer=False):
    from Gym.environments import identifier2environment
    from Gym.parameter_controllers import identifier2controller

    assert identifier in identifier2environment, f"unknown environment {identifier}"

    # Walk the controller up to `difficulty`, exactly like RLVEManager does, but freeze the
    # problems at that one level instead of sampling over a sliding window.
    controller = identifier2controller[identifier]()
    for _ in range(difficulty):
        controller.update()
    parameter_list = controller.get_parameter_list()

    rng = random.Random(param_seed)
    records, seed, attempts, unscoreable = [], seed_start, 0, 0
    while len(records) < num_problems and attempts < 20 * num_problems:
        attempts += 1
        problem = identifier2environment[identifier]()
        if not problem.generator(seed=seed, parameter=rng.choice(parameter_list)):
            seed += 1
            continue
        seed += 1
        config = problem.get_config()
        if not _survives_json(identifier, config):
            unscoreable += 1
            continue
        prompt = problem.prompt_generator()
        if seen_prompts is not None:
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
        records.append(
            {
                "user_prompt": prompt,
                "metadata": json.dumps({"environment": identifier, "config": config}),
            }
        )

    if unscoreable and not records:
        raise SystemExit(
            f"{identifier}: every generated problem fails to score after a JSON round trip, so "
            f"it cannot be used as a frozen eval set (its parameters probably use non-string "
            f"dict keys). Train on it if you like, but evaluate on something else."
        )
    if unscoreable:
        print(f"  {identifier}: skipped {unscoreable} problem(s) that do not survive JSON")

    if allow_fewer:
        assert records, f"no unique prompts generated for {identifier}"
    else:
        assert len(records) == num_problems, f"only generated {len(records)}/{num_problems} for {identifier}"
    return records


def cmd_generate(args):
    records, seed = [], args.seed_start
    for identifier in args.environments:
        seen_prompts = set() if getattr(args, "unique_prompts", False) else None
        # Split the per-environment budget evenly over the difficulty levels, remainder
        # going to the easiest ones.
        per_level, extra = divmod(args.num_problems_per_environment, len(args.difficulties))
        for i, difficulty in enumerate(args.difficulties):
            count = per_level + (1 if i < extra else 0)
            if count == 0:
                continue
            records += _records_for_environment(
                identifier,
                difficulty=difficulty,
                num_problems=count,
                seed_start=seed,
                param_seed=args.param_seed + difficulty,
                seen_prompts=seen_prompts,
            )
            seed += 1000  # keep each level's seeds in its own block
    return records


def cmd_subsample(args):
    source = json.load(open(args.source))
    by_environment = {}
    for record in source:
        metadata = record["metadata"]
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        by_environment.setdefault(metadata["environment"], []).append(record)

    rng = random.Random(args.param_seed)
    environments = sorted(by_environment)
    rng.shuffle(environments)
    if args.num_environments is not None:
        environments = environments[: args.num_environments]

    records, dropped = [], []
    for identifier in sorted(environments):
        candidates = by_environment[identifier]
        if args.max_prompt_chars is not None:
            # Long prompts cost generation time twice over (prefill + a longer answer);
            # in-loop eval is supposed to be cheap.
            short = [r for r in candidates if len(r["user_prompt"]) <= args.max_prompt_chars]
            candidates = short or candidates
        # Same JSON round-trip check as `generate`: an environment that cannot score its own
        # reference answer would sit at 0.0 forever and drag the average down with it.
        scoreable = []
        for record in candidates:
            metadata = record["metadata"]
            metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
            if _survives_json(identifier, metadata["config"]):
                scoreable.append(record)
        if not scoreable:
            dropped.append(identifier)
            continue
        records += rng.sample(scoreable, min(args.num_problems_per_environment, len(scoreable)))

    if dropped:
        print(f"  dropped {len(dropped)} environment(s) that cannot score after a JSON round trip: "
              + ", ".join(dropped))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--param-seed", type=int, default=0, help="seed for parameter/record choice")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    generate = subparsers.add_parser("generate", help="freeze problems from RLVE environments")
    generate.add_argument("--environments", nargs="+", required=True)
    generate.add_argument("--difficulties", type=int, nargs="+", default=[0, 1, 2, 3])
    generate.add_argument("--num-problems-per-environment", type=int, default=16)
    generate.add_argument("--seed-start", type=int, default=1_000_000)
    generate.add_argument(
        "--unique-prompts",
        action="store_true",
        help="deduplicate prompts within each environment across difficulty levels",
    )
    generate.set_defaults(func=cmd_generate)

    subsample = subparsers.add_parser("subsample", help="subsample an existing eval file")
    subsample.add_argument("--source", default="data/HELD-OUT_ENVIRONMENTS/test.json")
    subsample.add_argument("--num-environments", type=int, default=None)
    subsample.add_argument("--num-problems-per-environment", type=int, default=1)
    subsample.add_argument("--max-prompt-chars", type=int, default=None)
    subsample.set_defaults(func=cmd_subsample)

    args = parser.parse_args()
    records = args.func(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(output, "w"), indent=1)
    environments = {json.loads(r["metadata"])["environment"] for r in records}
    print(f"wrote {len(records)} problems from {len(environments)} environment(s) to {output}")
    if args.mode == "generate":
        print(f"  difficulties: {args.difficulties}")


if __name__ == "__main__":
    main()
