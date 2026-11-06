"""Align MOPD traces to the committed checkpoint before a Slurm restart."""

import argparse
import json
import time
from pathlib import Path

from modular_rl.v2.recovery import archive_attempt


def reconcile(run):
    run=Path(run)
    state_file=run/'ckpt/rlve_checkpoint_state.json'
    if not state_file.exists():return None
    state=json.loads(state_file.read_text())
    if state['completed']:return None
    archive=run/'resume-audit'/f'attempt-{time.time_ns()}'
    return archive_attempt(run,archive)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    print(json.dumps(reconcile(parser.parse_args().run)),flush=True)
