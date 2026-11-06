"""Validate a final model-only checkpoint, export to HF, and remove the source copy."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from archive_completed_mixing_checkpoints import tensor_shapes, validate_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--origin-hf-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    assert (run / "completed").is_file(), "Training did not finish successfully"
    marker_path = run / "weights_only_complete.json"
    checkpoint = run / "final-ckpt"
    if marker_path.exists():
        validate_export(run / "hf-final", args.origin_hf_dir, tensor_shapes(args.origin_hf_dir)[0])
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        return
    iteration = int((checkpoint / "latest_checkpointed_iteration.txt").read_text())
    temporary = run / "hf-final.incomplete"
    if temporary.exists():
        shutil.rmtree(temporary)
    vocab_size = json.loads((args.origin_hf_dir / "config.json").read_text())["vocab_size"]
    if not (run / "hf-final").exists():
        subprocess.run([
            sys.executable, "tools/convert_torch_dist_to_hf.py",
            "--input-dir", str(checkpoint / f"iter_{iteration:07d}"),
            "--output-dir", str(temporary), "--origin-hf-dir", str(args.origin_hf_dir),
            "--vocab-size", str(vocab_size),
        ], check=True)
        validate_export(temporary, args.origin_hf_dir, tensor_shapes(args.origin_hf_dir)[0])
        os.replace(temporary, run / "hf-final")
    tensor_count, weight_bytes = validate_export(run / "hf-final", args.origin_hf_dir, tensor_shapes(args.origin_hf_dir)[0])
    marker = {
        "format": "hf-safetensors", "path": "hf-final", "checkpoint_step": iteration,
        "weight_bytes": weight_bytes, "tensor_count": tensor_count,
    }
    temporary_marker = marker_path.with_suffix(".json.tmp")
    temporary_marker.write_text(json.dumps(marker, indent=2) + "\n")
    os.replace(temporary_marker, marker_path)
    shutil.rmtree(checkpoint)
    print(f"Exported terminal update {iteration + 1}; retained {weight_bytes / 1024**3:.2f} GiB of final weights")


if __name__ == "__main__":
    main()
