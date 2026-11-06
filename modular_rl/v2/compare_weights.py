"""Check saved policy parameters against their original Megatron reference."""

import argparse
import json
from pathlib import Path

from modular_rl.prepare import write_json


def compare(reference, checkpoint, output):
    import torch
    import torch.distributed.checkpoint as dcp

    torch.set_num_threads(1)
    names = (
        "decoder.layers.self_attention.linear_proj.weight",
        "decoder.layers.self_attention.linear_qkv.weight",
    )

    def first_layer(path, name):
        metadata = dcp.FileSystemReader(path).read_metadata().state_dict_metadata[name]
        state = {name: torch.empty(metadata.size, dtype=metadata.properties.dtype)}
        dcp.load(state, checkpoint_id=path, no_dist=True)
        return state[name][0].float().clone()

    probes = []
    for name in names:
        before, after = first_layer(reference, name), first_layer(checkpoint, name)
        if before.shape != after.shape:
            raise ValueError(f"Checkpoint tensor shape differs: {name}")
        difference = after - before
        probes.append(
            {
                "parameter": name,
                "layer": 0,
                "elements": before.numel(),
                "changed_elements": int(torch.count_nonzero(difference)),
                "max_absolute_delta": float(difference.abs().max()),
                "l2_delta": float(difference.norm()),
                "relative_l2_delta": float(difference.norm() / before.norm()),
            }
        )
    result = {
        "reference": str(reference),
        "checkpoint": str(checkpoint),
        "parameters_changed": any(probe["changed_elements"] > 0 for probe in probes),
        "probes": probes,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["parameters_changed"]:
        raise RuntimeError("The saved attention parameter probes did not change")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    compare(**vars(parser.parse_args()))
