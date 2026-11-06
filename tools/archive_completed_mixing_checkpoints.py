"""Export completed mixing runs to HF weights, then discard restart-only state."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from safetensors import safe_open


def tensor_shapes(directory):
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        filenames = set(weight_map.values())
    else:
        filenames = {path.name for path in directory.glob("*.safetensors")}
        weight_map = None
    if not filenames:
        raise RuntimeError(f"No safetensors in {directory}")
    shapes = {}
    total_bytes = 0
    for filename in filenames:
        path = directory / filename
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():
                if name in shapes:
                    raise RuntimeError(f"Duplicate tensor {name}")
                if weight_map is not None and weight_map.get(name) != filename:
                    raise RuntimeError(f"Index mismatch for {name}")
                shapes[name] = tuple(source.get_slice(name).get_shape())
        total_bytes += path.stat().st_size
    if weight_map is not None and set(shapes) != set(weight_map):
        raise RuntimeError(f"Incomplete safetensors index in {directory}")
    return shapes, total_bytes


def validate_export(output, origin, reference_shapes):
    if json.loads((output / "config.json").read_text()) != json.loads((origin / "config.json").read_text()):
        raise RuntimeError("Exported model config differs from the original")
    shapes, total_bytes = tensor_shapes(output)
    for name, expected in reference_shapes.items():
        if shapes.get(name) != expected:
            raise RuntimeError(f"Missing or mismatched trained tensor {name}: {shapes.get(name)} != {expected}")
    if set(shapes) - set(reference_shapes) - {"lm_head.weight"}:
        raise RuntimeError("Unexpected tensors in exported model")
    if total_bytes < 2_000_000_000:
        raise RuntimeError(f"Export is unexpectedly small: {total_bytes} bytes")
    return len(shapes), total_bytes


def archive_run(run, origin, reference_shapes, vocab_size, final_rollout_id=299,
                require_final_eval=True):
    lock_path = run / "run.lock"
    if not lock_path.exists():
        return "skipped (no run lock)", 0
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "skipped (running)", 0
        checkpoint = run / "ckpt"
        output = run / "hf-final"
        marker = run / "weights_only_complete.json"
        if marker.exists():
            tensor_count, weight_bytes = validate_export(output, origin, reference_shapes)
            if checkpoint.exists():
                shutil.rmtree(checkpoint)
            return f"already archived {tensor_count} tensors ({weight_bytes / 1024**3:.2f} GiB)", 0
        state_file = checkpoint / "rlve_checkpoint_state.json"
        if not state_file.exists():
            return "skipped (no checkpoint state)", 0
        state = json.loads(state_file.read_text())
        if (not state["completed"] or state["rollout_id"] != final_rollout_id
                or (require_final_eval and state["evaluated_rollout_id"] != final_rollout_id)):
            return "skipped (not fully completed)", 0
        model = checkpoint / f"iter_{final_rollout_id:07d}"
        temporary = run / "hf-final.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        if not output.exists():
            if not model.exists():
                raise RuntimeError("Final model checkpoint is missing")
            command = [
                sys.executable, "-u", "tools/convert_torch_dist_to_hf.py",
                "--input-dir", str(model), "--output-dir", str(temporary),
                "--origin-hf-dir", str(origin), "--vocab-size", str(vocab_size),
            ]
            subprocess.run(command, check=True)
            tensor_count, weight_bytes = validate_export(temporary, origin, reference_shapes)
            os.replace(temporary, output)
        else:
            tensor_count, weight_bytes = validate_export(output, origin, reference_shapes)
        if not marker.exists():
            marker_tmp = run / "weights_only_complete.json.tmp"
            marker_tmp.write_text(json.dumps({
                "checkpoint_step": final_rollout_id,
                "terminal_step": state["rollout_id"],
                "early_stop_reason": state.get("early_stop_reason"),
                "format": "hf-safetensors",
                "path": "hf-final",
                "tensor_count": tensor_count,
                "weight_bytes": weight_bytes,
            }, sort_keys=True) + "\n")
            os.replace(marker_tmp, marker)
        before = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
        shutil.rmtree(checkpoint)
        return f"archived {tensor_count} tensors ({weight_bytes / 1024**3:.2f} GiB)", before - weight_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--origin-hf-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--run-pattern", default="*-mix-*")
    parser.add_argument("--final-rollout-id", type=int, default=299)
    parser.add_argument("--allow-early-stop", action="store_true")
    parser.add_argument("--allow-unevaluated-final", action="store_true",
                        help="Archive a completed final update when eval cadence did not land on it")
    args = parser.parse_args()
    reference_shapes, _ = tensor_shapes(args.origin_hf_dir)
    saved = 0
    failures = 0
    runs = sorted((args.sweep_dir / "resumable").glob(args.run_pattern))
    if args.run_index is not None:
        run = args.sweep_dir / "resumable" / f"{args.run_index:02d}-mix-{args.run_index:02d}"
        if run not in runs:
            raise RuntimeError(f"Run directory does not exist: {run}")
        runs = [run]
    for run in runs:
        try:
            final_rollout_id = args.final_rollout_id
            if args.allow_early_stop:
                state_path = run / "ckpt" / "rlve_checkpoint_state.json"
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    if state["completed"] and 0 <= state["rollout_id"] <= args.final_rollout_id:
                        final_rollout_id = state["rollout_id"]
            result, reclaimed = archive_run(
                run, args.origin_hf_dir, reference_shapes, args.vocab_size, final_rollout_id,
                # A single-update run saves checkpoint 0, but cannot run a
                # post-update evaluation at that zero-based iteration.
                require_final_eval=not args.allow_unevaluated_final and final_rollout_id != 0,
            )
            saved += reclaimed
            print(f"{run.name}: {result}", flush=True)
        except Exception as error:
            failures += 1
            print(f"{run.name}: FAILED: {error}", file=sys.stderr, flush=True)
    print(f"Reclaimed {saved / 1024**3:.2f} GiB; failures: {failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
