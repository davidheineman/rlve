import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import cleanup_storage_review_20261026 as cleanup


class ParentCleanupTest(unittest.TestCase):
    def test_cache_receipt_path_is_not_overwritten_by_inventory_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scratch = root / 'scratch';out = root / 'archive';out.mkdir()
            (scratch / 'images').mkdir(parents=True)
            for name in ['slime-v0.5.0rc0-cu126.sif', 'miles-dev-202610231907.sif']:
                (scratch / 'images' / name).write_bytes(b'image')
            targets = [scratch / name for name in ['cache/apptainer-miles', 'nano/uv_cache', 'nano/hf_cache']]
            for target in targets:
                target.mkdir(parents=True);(target / 'cached').write_bytes(b'derived')
            measured = root / 'data/STORAGE-REVIEW-20261026/du-other.tsv'
            measured.parent.mkdir(parents=True)
            measured.write_text(''.join(f'4096\t{target}\n' for target in targets) + f'99\t{scratch}/wbenv\n')
            with patch.object(cleanup, 'ROOT', root), patch.object(cleanup, 'SCRATCH', scratch), \
                    patch.object(cleanup, 'OUT', out), patch.object(cleanup, 'BUILD_CACHES', targets), \
                    patch.object(cleanup, 'queue', return_value={}), patch.object(cleanup, 'command', return_value=''):
                result = cleanup.clean_build_caches()
            self.assertEqual(len(result['removed']), 3)
            self.assertTrue((out / 'build-cache-cleanup.json').is_file())
            self.assertTrue(all(not list(target.iterdir()) for target in targets))

    def test_keeps_all_active_parents_including_computed_future_branches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'slurm_jobs.json').write_text('{}')
            state = {'devices': [
                {'active': {'parent': {'rollouts': 400}, 'result': {'rollouts': 4000}}},
                {'active': {'parent': {'rollouts': 800}}},
                {'active': None},
            ]}
            self.assertEqual(cleanup.protected_parents(root, state, {}), {'400', '800'})

    def test_live_teacher_submit_journal_gap_preserves_parent(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'slurm_jobs.json').write_text('{}')
            task = root / 'tasks/teacher-01-g0003'
            task.mkdir(parents=True)
            (task / 'task.json').write_text(json.dumps({'parent': {'rollouts': 1200}}))
            token = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
            live = {'1': {'comment': f'rlve-async:{token}:teacher-01-g0003'}}
            self.assertEqual(cleanup.protected_parents(root, {'devices': []}, live), {'1200'})

    def test_unknown_live_teacher_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'slurm_jobs.json').write_text(json.dumps({'teacher-01-g0003': {'kind': 'teacher', 'job': '1'}}))
            with self.assertRaisesRegex(RuntimeError, 'metadata'):
                cleanup.protected_parents(root, {'devices': []}, {'1': {'comment': ''}})

    def test_coordinator_resumes_when_pruning_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'monitor.json').write_text(json.dumps({'coordinator_job': '1'}))
            (root / 'coordinator-lease.json').write_text(json.dumps({'timestamp': 10}))
            queue = {'1': {'state': 'RUNNING', 'name': 'scaling-async-n32', 'comment': ''}}
            with patch.object(cleanup, 'queue', return_value=queue), \
                    patch.object(cleanup, 'command', return_value=f'JobId=1 WorkDir={root}') as command, \
                    patch.object(cleanup.time, 'sleep'), \
                    patch.object(cleanup.subprocess, 'Popen'), \
                    patch.object(cleanup.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'test failure'):
                    with cleanup.quiesce(root):
                        raise RuntimeError('test failure')
                self.assertIn(['scancel', '--signal=STOP', '--full', '1'], [call.args[0] for call in command.call_args_list])
                self.assertEqual(run.call_args.args[0], ['scancel', '--signal=CONT', '--full', '1'])

    def test_gpu_job_cannot_be_suspended(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'monitor.json').write_text(json.dumps({'coordinator_job': '1'}))
            queue = {'1': {'state': 'RUNNING', 'name': 'scaling-async-n32', 'comment': ''}}
            with patch.object(cleanup, 'queue', return_value=queue), \
                    patch.object(cleanup, 'command', return_value=f'JobId=1 WorkDir={root} gres/gpu=1'), \
                    self.assertRaisesRegex(RuntimeError, 'GPU job'):
                with cleanup.quiesce(root):
                    self.fail('Must not enter pruning context')


if __name__ == '__main__':
    unittest.main()
