"""Retire obsolete exports without making NFS open-file cleanup fatal."""

import errno
import logging
import os
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def remove_retired(path):
    """NFS may keep .nfs files alive until an engine releases its memory map."""
    path = Path(path)
    try:
        shutil.rmtree(path)
        return True
    except FileNotFoundError:
        return True
    except OSError as error:
        if error.errno not in (errno.EBUSY, errno.ENOTEMPTY):
            raise
        with os.scandir(path) as entries:
            if any(not entry.name.startswith(".nfs") or not entry.is_file(follow_symlinks=False) for entry in entries):
                raise
        # Move busy directories to the back of the next bounded collection pass.
        path.touch()
        logger.info("Deferring cleanup of memory-mapped NFS snapshot: %s", path)
        return False


def retire_snapshot(root, source):
    """Only retired versions are collected; current and retained versions stay put."""
    root, source = Path(root), Path(source)
    relative = source.relative_to(root)
    if len(relative.parts) != 3 or relative.parts[1] not in ("weights", "sampler_weights"):
        raise ValueError(f"Refusing to retire a path outside a model checkpoint: {source}")
    retired = root / ".retired" / relative.parts[0]
    retired.mkdir(parents=True, exist_ok=True)
    destination = retired / uuid.uuid4().hex
    try:
        source.rename(destination)
    except FileNotFoundError:
        pass
    else:
        remove_retired(destination)
    # Each model slot owns its retirement directory. Limit metadata work per update.
    candidates = sorted(retired.iterdir(), key=lambda path: path.stat().st_mtime_ns)
    for path in candidates[:4]:
        if path != destination:
            remove_retired(path)
