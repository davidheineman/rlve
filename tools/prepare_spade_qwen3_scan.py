"""Add Qwen3's concise /no_think instruction to all frozen final-step games."""

import argparse
import json
from pathlib import Path


def prepare(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Source and output must differ")
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((source / "selection.json").read_text())
    counts = {}
    for name in ("train.jsonl", "eval.jsonl"):
        count = 0
        with (source / name).open() as incoming, (output / name).open("w") as outgoing:
            for line in incoming:
                row = json.loads(line)
                row["user_prompt"] += " /no_think"
                row["metadata"]["no_think"] = True
                outgoing.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        counts[name] = count
    selection.update(source_pool=str(source), no_think=True, prompt_style="concise",
                     num_train_prompts=counts["train.jsonl"],
                     num_eval_prompts=counts["eval.jsonl"])
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Prepared {len(selection['accepted_games'])} games, "
          f"{counts['train.jsonl']} train seeds and {counts['eval.jsonl']} eval seeds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/SPADE-concise"))
    parser.add_argument("--output", type=Path, default=Path("data/SPADE-qwen3-scan"))
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
