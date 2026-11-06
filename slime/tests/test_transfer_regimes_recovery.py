"""Allocation isolation and recovery from node-local scheduling failures."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from transfer_regimes.monitor import check, durable_rollouts, update_placement
from transfer_regimes.placement import allocated_rows

from async_branching.io import read_json, write_json


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.rows = [f"{i}, GPU-{i}, 40960" for i in range(7)]

    def test_unconfined_nvml_selects_only_the_slurm_allocation(self):
        self.assertEqual(
            allocated_rows(self.rows, "1,3,5,6"), [self.rows[i] for i in [1, 3, 5, 6]]
        )

    def test_compressed_gpu_range_is_supported(self):
        self.assertEqual(allocated_rows(self.rows, "3-6"), self.rows[3:])

    def test_uuid_mask_is_supported(self):
        self.assertEqual(
            allocated_rows(self.rows, "GPU-0,GPU-2,GPU-4,GPU-6"), self.rows[::2]
        )

    def test_confined_nvml_preserves_existing_ordinal_mapping(self):
        rows = self.rows[::2]
        self.assertEqual(allocated_rows(rows, "0,1,2,3"), rows)

    def test_unconfined_inventory_without_an_unambiguous_mask_is_rejected(self):
        for mask in ["", "0,1,2,9", "0,1,1,2", "0,1,2", "3-1", "GPU-0,0,1,2"]:
            with self.subTest(mask=mask), self.assertRaises(RuntimeError):
                allocated_rows(self.rows, mask)

    def test_node_exclusion_expands_slurm_hostlists_before_adding_node(self):
        def output(command, **_):
            if command[2] == "job":
                return "JobId=123 ExcNodeList=jagupard[28,30],tiger[6,8]"
            self.assertEqual(
                command, ["scontrol", "show", "hostnames", "jagupard[28,30],tiger[6,8]"]
            )
            return "jagupard28\njagupard30\ntiger6\ntiger8\n"

        with (
            patch(
                "transfer_regimes.monitor.subprocess.check_output", side_effect=output
            ),
            patch("transfer_regimes.monitor.subprocess.run") as run,
        ):
            update_placement("123", exclude_node="tiger7")
        self.assertIn(
            "ExcNodeList=jagupard28,jagupard30,tiger6,tiger7,tiger8",
            run.call_args.args[0],
        )

    def test_retry_progress_includes_checkpoints_during_a_teacher_barrier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "tasks/teacher0/latest.json",
                {"resume": "/bundle/checkpoints/u00050/state"},
            )
            write_json(root / "tasks/teacher1/result.json", {"rollouts": 800})
            self.assertEqual(durable_rollouts(root), 1200)

    def test_replacement_job_is_polled_instead_of_expired_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "sweep.json",
                {
                    "array_job_id": "100",
                    "experiments": [
                        {"index": 4, "job_id": "200_4", "experiment": str(root / "run")}
                    ],
                },
            )
            with patch(
                "transfer_regimes.monitor.state",
                return_value={
                    "state": "RUNNING",
                    "node": "healthy",
                    "reason": "healthy",
                },
            ) as state:
                report = check(root)
            state.assert_called_once_with("200_4")
            self.assertEqual(report["experiments"][0]["job_id"], "200_4")

    def test_interrupted_recovery_releases_only_its_recorded_held_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "sweep.json",
                {
                    "array_job_id": "100",
                    "experiments": [{"index": 0, "experiment": str(root / "run")}],
                },
            )
            process = SimpleNamespace(returncode=0, stderr="")
            with (
                patch(
                    "transfer_regimes.monitor.state",
                    return_value={"state": "FAILED", "node": "tiger7", "reason": "1:0"},
                ),
                patch(
                    "transfer_regimes.monitor.tail",
                    return_value="Need 12 GiB of free node-local /tmp",
                ),
                patch("transfer_regimes.monitor.subprocess.run", return_value=process),
                patch(
                    "transfer_regimes.monitor.update_placement",
                    side_effect=subprocess.CalledProcessError(1, ["scontrol"]),
                ),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    check(root)
            self.assertIn("100_0", read_json(root / "recovery-pending.json"))
            with (
                patch(
                    "transfer_regimes.monitor.state",
                    return_value={
                        "state": "PENDING",
                        "node": "",
                        "reason": "job requeued in held state",
                    },
                ),
                patch("transfer_regimes.monitor.update_placement"),
                patch(
                    "transfer_regimes.monitor.subprocess.run", return_value=process
                ) as run,
            ):
                report = check(root)
            run.assert_called_once_with(
                ["scontrol", "release", "100_0"], check=True, timeout=20
            )
            self.assertTrue(report["experiments"][0]["requeued"])
            self.assertEqual(read_json(root / "recovery-pending.json"), {})


if __name__ == "__main__":
    unittest.main()
