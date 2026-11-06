"""Preserve cancelled MOPD model/RNG payloads while retiring superseded Adam state."""

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from compact_finished_optimizer_checkpoints import atomic, identities, retain_payloads, verify_existing


def compact_cancelled(root):
    halt=json.loads((root/'HALTED_BY_USER.json').read_text())
    jobs=json.loads((root/'students/jobs.json').read_text())
    states={}
    result=subprocess.check_output(['sacct','-n','-P','-X','-j',','.join(j['job_id'] for j in jobs),
                                   '--format=JobIDRaw,State'],text=True)
    for line in result.splitlines():
        fields=line.split('|')
        if len(fields)>1:states[fields[0]]=fields[1]
    audit=root/'cancelled-storage-compaction';audit.mkdir(exist_ok=True)
    for job in jobs:
        if job['job_id'] not in halt['jobs'] or not states.get(job['job_id'],'').startswith('CANCELLED'):
            raise ValueError('Require an explicitly user-cancelled, terminal job')
        policy=Path(job['output'])/'policy'
        if not policy.resolve().is_relative_to(root.resolve()/'students/runs') or policy.is_symlink():
            raise ValueError('Policy outside the cancelled experiment')
        if (policy/'optimizer_state_retired.json').exists():continue
        with (policy/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=json.loads((policy/'ckpt/rlve_checkpoint_state.json').read_text())
            step=int((policy/'ckpt/latest_checkpointed_iteration.txt').read_text())
            if state['completed'] or step!=state['rollout_id'] or step<1:
                raise ValueError('Require a committed partial checkpoint')
            native=policy/'ckpt'/f'iter_{step:07d}'
            target=native.with_name(native.name+'.model-only-incomplete')
            retired=native.with_name(native.name+'.optimizer-retired')
            if target.exists() or retired.exists():raise ValueError('Inspect interrupted compaction first')
            before=identities(native);original_bytes=sum(v[1] for v in before.values())
            record={'policy':str(policy),'native':str(native),'job_id':job['job_id'],'slurm_state':states[job['job_id']],
                    'checkpoint_state':state,'original_bytes':original_bytes,'status':'copying','started_at':time.time()}
            key=hashlib.sha256(str(native).encode()).hexdigest()[:16];receipt=audit/(key+'.json');atomic(receipt,record)
            retain_payloads(native,target)
            proof=verify_existing(native,target)
            if identities(native)!=before:raise ValueError('Original checkpoint changed')
            record.update(proof,status='verified',retained_bytes=sum(v[1] for v in identities(target).values()))
            atomic(receipt,record)
            native.rename(retired);target.rename(native)
            record.update(status='model_committed');atomic(receipt,record)
            shutil.rmtree(retired)
            record.update(status='complete',bytes_reclaimed=original_bytes-record['retained_bytes'],finished_at=time.time())
            atomic(receipt,record)
            atomic(policy/'optimizer_state_retired.json',{'status':'model_only','audit':str(receipt),
                'model_tensors_identical':True,'rng_and_extra_state_retained':True,'exact_adam_resume_available':False,
                'reason':'User cancelled this run and requested a fresh baseline-SFT restart with eval32'})
            print('COMPACTED_CANCELLED',job['name'],'reclaimed_GB',record['bytes_reclaimed']/1e9,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    compact_cancelled(parser.parse_args().root.resolve())
