"""Concurrent job shells must hold different shared port reservations."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/slurm/port_slot.sh"


class PortSlotTest(unittest.TestCase):
    def start_reservation(self, directory, private_tmp):
        private_tmp.mkdir()
        process = subprocess.Popen(
            ["bash", "-c", 'set -euo pipefail; source "$1"; '
             'rlve_reserve_port_slot "$2" 0; printf "%s\\n" "$RLVE_RESERVED_PORT_SLOT"; read -r release',
             "bash", str(SCRIPT), str(directory)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "TMPDIR": str(private_tmp)},
        )
        self.addCleanup(self.stop, process)
        slot = process.stdout.readline().strip()
        self.assertTrue(slot.isdigit(), process.stderr.read() if process.poll() is not None else slot)
        return process, int(slot)

    @staticmethod
    def stop(process):
        if process.poll() is None:
            process.communicate("release\n", timeout=10)

    def test_concurrent_jobs_with_private_tmp_reserve_distinct_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, first_slot = self.start_reservation(root / "shared", root / "job1-tmp")
            second, second_slot = self.start_reservation(root / "shared", root / "job2-tmp")
            self.assertNotEqual(first_slot, second_slot)
            self.stop(first)
            third, third_slot = self.start_reservation(root / "shared", root / "job3-tmp")
            self.assertEqual(first_slot, third_slot)
            self.stop(second)
            self.stop(third)
