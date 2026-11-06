"""Select allocated GPUs even when NVML exposes the whole node."""

import re


def allocated_rows(rows, allocation, *, expected=4):
    records = [tuple(value.strip() for value in row.split(",")) for row in rows]
    if any(len(record) != 3 for record in records):
        raise RuntimeError("GPU allocation inventory contains malformed records")
    if len(records) == expected:
        # On nodes with NVML device isolation this is already the confined
        # allocation. CUDA ordinals may differ from NVML physical indices.
        return rows
    identifiers = []
    for part in allocation.split(",") if allocation else []:
        part = part.strip()
        if re.fullmatch(r"\d+-\d+", part):
            first, last = map(int, part.split("-"))
            if last < first or last - first > 1024:
                raise RuntimeError("GPU allocation inventory has an invalid ID range")
            identifiers.extend(str(i) for i in range(first, last + 1))
        else:
            identifiers.append(part)
    if len(identifiers) != expected or len(set(identifiers)) != expected:
        raise RuntimeError(
            f"GPU allocation inventory exposes {len(rows)} devices; need an explicit "
            f"{expected}-GPU SLURM_JOB_GPUS mask, got {allocation!r}"
        )
    by_id = {}
    for row, record in zip(rows, records):
        by_id[record[0]] = row
        by_id[record[1]] = row
    if any(identifier not in by_id for identifier in identifiers):
        raise RuntimeError(
            "GPU allocation inventory does not contain the Slurm GPU IDs"
        )
    selected = [by_id[identifier] for identifier in identifiers]
    if len(set(selected)) != expected:
        raise RuntimeError("GPU allocation inventory maps multiple IDs to one device")
    return selected
