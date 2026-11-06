"""Dolci IF constraint scores, ported from tinker-mopd's audited adapter."""

import ast
import json
import os
import random
import sys
from pathlib import Path


def registry():
    runtime = os.environ.get("MODULAR_RL_RUNTIME")
    if runtime:
        dependencies = str(Path(runtime) / "if_deps")
        if dependencies not in sys.path:
            sys.path.insert(0, dependencies)
        import nltk

        resources = str(Path(runtime) / "if_nltk")
        if resources not in nltk.data.path:
            nltk.data.path.insert(0, resources)
    from langdetect import DetectorFactory

    from .vendor.ifevalg.instructions_registry import INSTRUCTION_DICT

    DetectorFactory.seed = 0
    return INSTRUCTION_DICT


def if_target(raw):
    for _ in range(6):
        if isinstance(raw, list) and len(raw) == 1:
            raw = raw[0]
        elif isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                raw = ast.literal_eval(raw)
        else:
            break
    if not isinstance(raw, dict):
        raise TypeError("IF target must decode to a constraint bundle")
    ids, kwargs = raw.get("instruction_id"), raw.get("kwargs")
    if (not isinstance(ids, list) or not ids or not isinstance(kwargs, list)
            or len(ids) != len(kwargs) or any(not isinstance(i, str) for i in ids)
            or any(k is not None and not isinstance(k, dict) for k in kwargs)):
        raise ValueError("IF IDs and argument dictionaries must have equal nonzero lengths")
    return raw


class InclusiveCopySpan:
    """The Dolci prompt explicitly includes both endpoints; zero is valid."""

    def __init__(self, arguments):
        text, start, end = (arguments[k] for k in ("prompt_to_repeat", "n_start", "n_end"))
        if (not isinstance(text, str) or type(start) is not int or type(end) is not int
                or not 0 <= start <= end < len(text)):
            raise ValueError("Invalid inclusive copy-span target")
        self.expected = text[start:end + 1].strip().lower()

    def check_following(self, value):
        return value.strip().lower() == self.expected


def if_checkers(target):
    target = if_target(target)
    available = registry()
    checkers = []
    for name, arguments in zip(target["instruction_id"], target["kwargs"], strict=True):
        if name not in available:
            raise ValueError(f"Unsupported IF instruction: {name}")
        arguments = {k: v for k, v in (arguments or {}).items() if v is not None}
        if name == "new:copy_span_idx":
            checkers.append(InclusiveCopySpan(arguments))
            continue
        checker = available[name](name)
        state = random.getstate()
        try:
            random.seed(json.dumps([name, arguments], sort_keys=True))
            checker.build_description(**arguments)
        finally:
            random.setstate(state)
        checkers.append(checker)
    return checkers


def grade_if(text, target):
    target = if_target(target)
    # Match the starting adapter's response-only handling of explicit thinking.
    text = text.rsplit("</think>", 1)[-1].strip()
    for marker in ("<|endoftext|>", "<|im_end|>"):
        text = text.removesuffix(marker).rstrip()
    checks = [bool(text) and bool(c.check_following(text)) for c in if_checkers(target)]
    return {"reward": sum(checks) / len(checks), "accuracy": float(all(checks)),
            "constraint_passes": checks, "constraints": target["instruction_id"]}


async def reward(args, sample, **kwargs):
    if sample.metadata["task"] != "if":
        raise ValueError("IF reward requires IF metadata")
    return grade_if(sample.response, sample.metadata["target"])
