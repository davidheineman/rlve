"""Actor prompt and boxed-action parsing used by the released SPADE games."""

import re


def actor_prompt(observation: str, style: str = "spade") -> str:
    prefix = (
        "You are playing a language game. Make valid actions to win.\n"
        f"Observation: {observation}\n"
    )
    if style == "spade":
        # Matches SPADE's qwen3_game actor template.
        return prefix + "Please reason step by step, and put your final answer within \\boxed{}."
    if style == "concise":
        return prefix + "Reply with one next action inside \\boxed{}. No explanation."
    raise ValueError(f"Unknown actor prompt style: {style}")


def parse_action(response: str) -> str:
    """Extract the first boxed action as SPADE's boxed actor path does."""
    for pattern in (
        r"\\boxed\{([^{}]+)\}",
        r"\\boxed\{\{([^{}]+)\}\}",
        r"\\boxed\{\{\{([^{}]+)\}\}\}",
    ):
        match = re.search(pattern, response)
        if match:
            return rf"\boxed{{{match.group(1).strip()}}}"
    return response.strip()


def observation_suffix_ids(tokenizer, observation: str) -> list[int]:
    """Render the next user turn in the model's own chat format."""
    marker = "__spade_assistant_content_marker__"
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "previous observation"},
            {"role": "assistant", "content": marker},
            {"role": "user", "content": observation},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if rendered.count(marker) != 1:
        raise ValueError("Could not locate assistant turn in chat template")
    suffix = rendered.split(marker, 1)[1]
    eos = tokenizer.eos_token
    if not eos or not suffix.startswith(eos):
        raise ValueError("Chat template does not separate turns with model EOS")
    return tokenizer(suffix[len(eos):], add_special_tokens=False)["input_ids"]
