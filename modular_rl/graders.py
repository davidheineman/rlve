"""Task rewards. Reference answers and test failures are audited before training."""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


def numeric_answer(text: str, *, reference=False):
    """Read an explicit final answer, never a number from an unfinished derivation."""
    text = text.rsplit("</think>", 1)[-1]
    if reference:
        if "####" not in text:
            raise ValueError("GSM reference has no #### answer")
        text = text.rsplit("####", 1)[1].strip()
    else:
        boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
        tags = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if boxed:
            text = boxed[-1]
        elif tags:
            text = tags[-1]
        else:
            # Permit a bare numeric final line, but not explanatory prose.
            lines = text.strip().splitlines()
            text = lines[-1] if lines else ""
    text = text.strip().rstrip(".").strip("$ ").replace(",", "")
    text = re.sub(r"\\(?:d?frac)\{(-?\d+)\}\{(-?\d+)\}", r"\1/\2", text)
    text = text.removesuffix(r"\%").removesuffix("%").strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", text):
        return None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def grade_math(response, answer):
    expected = numeric_answer(answer, reference=True)
    if expected is None:
        raise ValueError(f"Invalid GSM answer: {answer!r}")
    actual = numeric_answer(response)
    # Dataset decimals sometimes round repeating fractions.
    correct = actual is not None and abs(float(actual - expected)) <= 1e-6
    return {"reward": float(correct), "accuracy": float(correct)}


def extract_code(response):
    response = response.rsplit("</think>", 1)[-1].strip()
    # Bare Python answers can end with a tokenizer transport marker. Passing
    # that marker to compile() turns an otherwise valid solution into a syntax
    # error, whereas fenced answers already discard text after their block.
    while True:
        previous = response
        for marker in ("<|im_end|>", "<|endoftext|>"):
            response = response.removesuffix(marker).rstrip()
        if response == previous:
            break
    fenced = re.findall(r"```([^\n`]*)\n(.*?)```", response, re.DOTALL)
    blocks = [
        code.strip()
        for language, code in fenced
        if language.strip().lower() in ("", "python", "python3", "py")
    ]
    # Answers often append example calls and output after the implementation.
    # Prefer the last implementation block, so a final corrected solution wins.
    for code in reversed(blocks):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        ):
            return code
    return (blocks[0] if blocks else response).strip()


def grade_code(response, target, timeout=5, *, diagnostics=False):
    """Fresh, resource-limited Python process; no inherited credentials or CUDA."""
    worker = Path(__file__).with_name("code_worker.py")
    with tempfile.TemporaryDirectory(prefix="modular-mbpp-") as directory:
        process = subprocess.Popen(
            [sys.executable, "-I", str(worker)],
            cwd=directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={
                "PATH": os.defpath,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONHASHSEED": "0",
            },
        )
        try:
            stdout, stderr = process.communicate(
                json.dumps(
                    {
                        "code": extract_code(response),
                        "target": target,
                        "diagnostics": diagnostics,
                    }
                ),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return {"reward": 0.0, "accuracy": 0.0, "error": "timeout"}
    if process.returncode < 0:
        return {
            "reward": 0.0,
            "accuracy": 0.0,
            "error": f"signal:{-process.returncode}",
        }
    if process.returncode:
        if any(marker in stderr for marker in ("TimeoutError: CPU limit", "MemoryError", "RecursionError")):
            return {"reward": 0.0, "accuracy": 0.0, "error": "worker_resource_limit"}
        raise RuntimeError(f"MBPP executor failed: {stderr[-2000:]}")
    result = json.loads(stdout)
    correct = bool(result["passes"]) and all(result["passes"])
    return {"reward": float(correct), "accuracy": float(correct), **result}
