"""Exact string reward with forgiving final-answer extraction."""

from __future__ import annotations

import ast
import json
import re


def _objects(text: str):
    """Yield balanced brace substrings, including objects inside code fences."""
    stack = []
    quote = None
    escaped = False
    for i, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif stack and char in "\"'":
            quote = char
        elif char == "{":
            stack.append(i)
        elif char == "}" and stack:
            yield text[stack.pop(): i + 1]


def extract_output(response: str) -> str | None:
    """Use the last valid JSON/Python object, then explicit plain-text finals."""
    values = []
    for candidate in _objects(response):
        for parse in (json.loads, ast.literal_eval):
            try:
                item = parse(candidate)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(item, dict) and isinstance(item.get("output"), str):
                values.append(item["output"])
                break
    if values:
        return values[-1]

    boxed = re.findall(r"\\boxed\{([^{}]*)\}", response)
    if boxed:
        return _unquote(boxed[-1].strip())

    # Permit a quoted or bare string on the final line, or after an explicit
    # final-answer marker. Never execute model text or take arbitrary prose.
    lines = [line.strip() for line in response.splitlines() if line.strip() and not line.strip().startswith("```")]
    if not lines:
        return None
    line = lines[-1]
    match = re.fullmatch(r"(?i)(?:final answer|output)\s*[:=]\s*(.*)", line)
    if match:
        line = match.group(1).strip()
    if (line.startswith(('"', "'")) and line.endswith(('"', "'"))) or re.fullmatch(r"[A-Za-z0-9!#-]*", line):
        return _unquote(line)
    return None


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, str):
                return parsed
        except (ValueError, SyntaxError):
            return value[1:-1]
    return value


async def reward(args, sample, **kwargs):
    extracted = extract_output(sample.response)
    correct = float(extracted is not None and extracted == sample.label)
    return {"reward": correct, "accuracy": correct}
