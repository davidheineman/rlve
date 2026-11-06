"""MBPP executor, following tinker-mopd's subprocess/reliability-guard pattern.

This is a reliability guard for research samples, not a security sandbox.
"""

import builtins
import contextlib
import io
import json
import os
import resource
import signal
import sys


def timeout(signum, frame):
    raise TimeoutError("CPU limit")


def run(payload):
    signal.signal(signal.SIGXCPU, timeout)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (3, 4))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    # Disable process creation and destructive filesystem operations, as in minieval.
    import shutil
    import subprocess

    for name in (
        "kill",
        "killpg",
        "system",
        "fork",
        "forkpty",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "rename",
        "renames",
        "replace",
        "truncate",
        "chmod",
        "chown",
        "chroot",
        "setuid",
        "putenv",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    subprocess.Popen = None
    shutil.rmtree = shutil.move = shutil.chown = None
    builtins.exit = builtins.quit = None
    for name in ("ctypes", "socket", "resource", "psutil", "joblib", "multiprocessing"):
        sys.modules[name] = None
    namespace = {"__name__": "__main__"}
    target = payload["target"]
    passes = []
    diagnostics = payload.get("diagnostics", False)
    test_results = []
    try:
        # Keep executable test objects and the executor private before generated
        # code can mutate builtins.compile/exec in this subprocess.
        execute = exec
        setup_code = compile(target.get("setup", ""), "<setup>", "exec")
        solution_code = compile(payload["code"], "<completion>", "exec")
        test_codes = [compile(test, "<test>", "exec") for test in target["tests"]]
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            execute(setup_code, namespace)
            execute(solution_code, namespace)
            for test, test_code in zip(target["tests"], test_codes, strict=True):
                try:
                    execute(test_code, namespace)
                    passes.append(True)
                    if diagnostics:
                        test_results.append({"test": test, "passed": True})
                except BaseException as exc:  # noqa: BLE001 - generated code can call sys.exit
                    passes.append(False)
                    if diagnostics:
                        test_results.append(
                            {
                                "test": test,
                                "passed": False,
                                "error": type(exc).__name__,
                                "message": str(exc)[:500],
                            }
                        )
    except BaseException as exc:  # noqa: BLE001 - generated code failures are policy errors
        return {"passes": [False] * len(target["tests"]), "error": type(exc).__name__}
    result = {"passes": passes}
    if diagnostics:
        result["test_results"] = test_results
    return result


def main():
    payload = json.load(sys.stdin)
    try:
        result = run(payload)
    except TimeoutError:
        # SIGXCPU can arrive after execution returns but before its result is
        # serialized. It is a failed sample, not a failed training allocation.
        result = {"passes": [False] * len(payload['target']['tests']), "error": "cpu_timeout"}
    finally:
        signal.signal(signal.SIGXCPU, signal.SIG_IGN)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
