import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[2] / "scripts/slurm/run_one.sh"


class LauncherSnapshotTest(unittest.TestCase):
    def test_editing_source_cannot_change_a_running_launchers_unread_tail(self):
        source = SOURCE.read_text()
        guard = source.split("# BEGIN immutable launcher snapshot", 1)[1].split("# END immutable launcher snapshot", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "launcher.sh"
            script.write_text('#!/bin/bash\nset -euo pipefail\n' + guard +
                              '\nprintf "ready|%s\\n" "$RLVE_LAUNCHER_SNAPSHOT"\n'
                              'read -r continue\nprintf "original finalization\\n"\n')
            process = subprocess.Popen(["bash", str(script)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                ready = process.stdout.readline().strip()
                self.assertTrue(ready.startswith("ready|"), ready)
                snapshot = Path(ready.split("|", 1)[1])
                self.addCleanup(snapshot.unlink, missing_ok=True)
                # Reproduce the in-place edit that used to corrupt long-running shells.
                script.write_text('#!/bin/bash\nprintf "unterminated edited script')
                output, error = process.communicate("continue\n", timeout=10)
                self.assertEqual(process.returncode, 0, error)
                self.assertEqual(output.strip(), "original finalization")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
