"""Validate exported OPD teacher metadata before loading its weights."""


def validate_teacher_marker(marker, environment, directory):
    step = marker.get("checkpoint_step")
    if (
        type(step) is not int
        or step < 0
        or marker.get("format") != "hf-safetensors"
        or marker.get("path") != "hf-final"
        or not isinstance(marker.get("weight_bytes"), int)
        or marker["weight_bytes"] <= 0
    ):
        raise RuntimeError(f"Incomplete teacher for {environment}: {directory}")
