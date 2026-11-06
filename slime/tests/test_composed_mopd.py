"""Frozen composed assignments feed the existing causal async trainer."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clustering_mopd.composed import COMPOSED_SEARCH, assignments_from_composed_search
from clustering_mopd.estimate import calibrate_completed, completed_range
from clustering_mopd.plan import build_plan, configurations, metadata, partition
from clustering_mopd.schedule import ClusterSimulation, initial_state

from async_branching.io import read_json, write_json
from async_branching.replay import ReplayBackend
from async_branching.runner import Runner


class ComposedAssignmentTests(unittest.TestCase):
    def fixture(self, root):
        groups = [list(range(i, i + 8)) for i in range(0, 64, 8)]
        other = [[i + 8 * j for j in range(8)] for i in range(8)]
        write_json(
            root / "plan.json",
            {
                "environments": [f"env-{i}" for i in range(64)],
                "random_seeds": [1234, 4321],
            },
        )
        write_json(
            root / "assignments.json",
            {
                "used_confirmation": False,
                "frozen_at": 123,
                "candidates": [
                    {
                        "k": 8,
                        "best": {"groups": groups},
                        "worst": {"groups": other},
                        "random_groups": [other, groups],
                    }
                ],
            },
        )
        return groups, other

    def test_import_uses_frozen_groups_without_reading_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            high, low = self.fixture(root)
            (root / "completed.json").write_text("confirmation must never be read")
            selected = assignments_from_composed_search(root)
            original = copy.deepcopy(selected)
            self.assertEqual(partition(8, "high-transfer", selected, 99), high)
            self.assertEqual(partition(8, "low-transfer", selected, 99), low)
            self.assertEqual(partition(8, "random", selected, 99999), low)
            self.assertEqual(selected["provenance"]["random_seed"], 1234)
            self.assertEqual(selected, original)

    def test_confirmation_selected_or_incomplete_partitions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            frozen = read_json(root / "assignments.json")
            frozen["used_confirmation"] = True
            write_json(root / "assignments.json", frozen)
            with self.assertRaisesRegex(ValueError, "without confirmation"):
                assignments_from_composed_search(root)
            frozen["used_confirmation"] = False
            frozen["candidates"][0]["best"]["groups"][0][0] = 63
            write_json(root / "assignments.json", frozen)
            with self.assertRaisesRegex(ValueError, "exactly once"):
                assignments_from_composed_search(root)

    def test_random_replicate_must_be_predeclared(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "predeclared"):
                assignments_from_composed_search(root, random_replicate=2)

    def test_multiple_counts_use_their_own_frozen_mixing_groups(self):
        selected = assignments_from_composed_search(COMPOSED_SEARCH, counts=(4, 16))
        frozen = read_json(COMPOSED_SEARCH / "assignments.json")
        self.assertEqual(
            configurations(64, composed=True, composed_counts=(4, 16)),
            [
                (k, a)
                for k in (4, 16)
                for a in ("high-transfer", "random", "low-transfer")
            ],
        )
        plans = []
        for count, assignment in configurations(
            64, composed=True, composed_counts=(4, 16)
        ):
            source = next(r for r in frozen["candidates"] if r["k"] == count)
            expected = {
                "high-transfer": source["best"]["groups"],
                "random": source["random_groups"][0],
                "low-transfer": source["worst"]["groups"],
            }[assignment]
            plan, inputs = build_plan(
                count, assignment, devices=8, size=64, assignment_selection=selected
            )
            plans.append(plan)
            self.assertEqual(
                [c["environment_indices"] for c in plan["clusters"]],
                sorted(sorted(g) for g in expected),
            )
            self.assertEqual(plan["teacher_worker_count"], min(7, count))
            self.assertEqual(plan["adapter_slots"], min(7, count) + 1)
            self.assertEqual(plan["simulated_idle_devices"], max(0, 7 - count))
            self.assertEqual(plan["simulated_clock_budget"], 16000)
            self.assertEqual(len(inputs[plan["id_eval"]]), 64)
            for cluster in plan["clusters"]:
                self.assertEqual(len(cluster["environments"]), 64 // count)
                panel = inputs[cluster["eval"]]
                self.assertEqual(len(panel), 32)
                for env in cluster["environments"]:
                    self.assertEqual(
                        sum(metadata(r)["environment"] == env for r in panel),
                        count // 2,
                    )
        self.assertTrue(
            all(
                p["initial_student_reference"] == plans[0]["initial_student_reference"]
                for p in plans
            )
        )
        for counts in [(), (4, 4), (3,)]:
            with self.assertRaises(ValueError):
                configurations(64, composed=True, composed_counts=counts)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            assignments_from_composed_search(COMPOSED_SEARCH, counts=(64,))
        with self.assertRaisesRegex(ValueError, "unique"):
            assignments_from_composed_search(COMPOSED_SEARCH, counts=(4, 4))

    def test_real_k8_triplet_matches_frozen_composed_search_and_old_recipe(self):
        selected = assignments_from_composed_search(COMPOSED_SEARCH)
        discovery = read_json(COMPOSED_SEARCH / "plan.json")
        probes = read_json(Path(discovery["evaluation"]["selection"]["path"]))

        def probe_key(row):
            value = metadata(row)
            return value["environment"], value["config"]["seed"]

        selection_probes = {probe_key(r) for r in probes}
        plans = []
        self.assertEqual(
            configurations(64, composed=True),
            [(8, "high-transfer"), (8, "random"), (8, "low-transfer")],
        )
        for count, assignment in configurations(64, composed=True):
            plan, inputs = build_plan(
                count,
                assignment,
                devices=8,
                size=64,
                assignment_selection=selected,
                student_project="rl-clustering-students-v2",
                teacher_project="rl-clustering-teachers-v2",
            )
            plans.append(plan)
            self.assertEqual(
                [c["environment_indices"] for c in plan["clusters"]],
                selected["candidates"][0]["groups_by_assignment"][assignment],
            )
            self.assertEqual(plan["teacher_worker_count"], 7)
            self.assertEqual(plan["adapter_slots"], 8)
            self.assertEqual(plan["simulated_clock_budget"], 16000)
            self.assertEqual(plan["student_project"], "rl-clustering-students-v2")
            self.assertEqual(plan["teacher_project"], "rl-clustering-teachers-v2")
            self.assertEqual(plan["project"], "rl-clustering-teachers-v2")
            self.assertEqual(len(inputs[plan["id_eval"]]), 64)
            self.assertTrue(
                selection_probes.isdisjoint(
                    probe_key(r) for r in inputs[plan["id_eval"]]
                )
            )
            for cluster in plan["clusters"]:
                panel = inputs[cluster["eval"]]
                self.assertEqual(len(panel), 32)
                self.assertTrue(
                    selection_probes.isdisjoint(probe_key(r) for r in panel)
                )
                self.assertEqual(len(cluster["environments"]), 8)
                for environment in cluster["environments"]:
                    self.assertEqual(
                        sum(metadata(r)["environment"] == environment for r in panel), 4
                    )
            old = read_json(Path(plan["reference_async_experiment"]) / "plan.json")
            for field in [
                "teacher_config",
                "student_config",
                "teacher_branch_rollouts",
                "student_rollouts_per_teacher_version",
                "teacher_recovery_fraction",
                "student_quantum_rollouts",
                "lora_rank",
                "lora_alpha",
            ]:
                self.assertEqual(plan[field], old[field])
        self.assertEqual(
            plans[0]["initial_student_reference"], plans[1]["initial_student_reference"]
        )
        self.assertEqual(
            plans[1]["initial_student_reference"], plans[2]["initial_student_reference"]
        )
        self.assertEqual(
            len(
                {
                    tuple(tuple(c["environment_indices"]) for c in p["clusters"])
                    for p in plans
                }
            ),
            3,
        )

    def test_eight_clusters_use_seven_causal_teacher_lanes(self):
        plan = {
            "clusters": [{"id": f"cluster-{i:02d}"} for i in range(8)],
            "student_rollouts": 2400,
            "simulated_clock_budget": 2400,
            "teacher_worker_count": 7,
            "schedule_seed": 1234,
            "parent_policy": "latest-available",
            "teacher_device_rollout_horizon": None,
        }
        state = initial_state(plan, {"hf": "base", "resume": "state"})
        simulation = ClusterSimulation(
            state, teacher_cap=400, merge_cap=200, quantum=80
        )
        backend = ReplayBackend(teacher_rollouts=400)
        with tempfile.TemporaryDirectory() as temporary:
            Runner(simulation, backend, Path(temporary)).run()
        self.assertEqual(state.clock, 2400)
        self.assertEqual(len(state.devices), 7)
        teachers = [t for t in backend.tasks if t["kind"] == "teacher"]
        self.assertEqual(len({t["environment"] for t in teachers}), 8)
        self.assertLessEqual(
            sum(state.teacher_rollout_counts().values()) + state.student_rollouts,
            8 * state.clock,
        )
        for task in teachers:
            self.assertLessEqual(task["parent"]["clock"], task["start_clock"])
            self.assertLessEqual(
                task["parent"]["rollouts"], task["teacher_start_rollouts"]
            )


class CompletedCalibrationTests(unittest.TestCase):
    def test_k4_and_k16_can_scale_the_completed_eight_device_calibration(self):
        calibration = {
            "runs": [
                {
                    "label": "n64-k8",
                    "teacher_rollouts": 112000,
                    "allocated_hours": 14,
                    "student_rollouts": 7600,
                    "student_seconds_per_update_median": 14,
                    "id_eval_prompts": 64,
                    "test_eval_prompts": 32,
                    "student_evaluation_points": 20,
                }
            ],
            "seconds_per_256_sample_gate_median": 160,
        }
        results = []
        for k, work in [(4, 64000), (16, 112000)]:
            plan = {"cluster_count": k, "simulated_devices": 8, "id_eval_prompts": 64}
            result = completed_range(plan, calibration, work, 12000, 30)
            lo, hi = result["physical_hours_range"]
            self.assertGreater(lo, 0)
            self.assertGreater(hi, lo)
            results.append(lo)
        self.assertGreater(results[1], results[0])

    def test_allocated_time_excludes_queue_and_covers_resubmitted_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiments = []
            for i in range(2):
                e = root / f"run-{i}"
                write_json(
                    e / "completed.json",
                    {"simulated_clock": 16000, "student": {"rollouts": 6400}},
                )
                write_json(
                    e / "plan.json",
                    {"simulated_devices": 8, "physical_gpus": 4, "id_eval_prompts": 32},
                )
                write_json(
                    e / "job.json",
                    {"job_id": str(i + 10), "previous_job_ids": [str(i + 20)]},
                )
                write_json(e / "status.json", {"physical_elapsed_seconds": 99 * 3600})
                p = e / "workers/student/metrics.jsonl"
                p.parent.mkdir(parents=True)
                p.write_text(
                    json.dumps(
                        {
                            "train/seconds_per_update": 16,
                            "clustering/val_envs_seconds": 120,
                        }
                    )
                    + "\n"
                )
                p = e / "tasks/teacher-cluster-00-g0001/metrics.jsonl"
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps({"david/rollouts": 4000}) + "\n")
                experiments.append(
                    {"cluster_count": 8, "label": f"run-{i}", "experiment": str(e)}
                )
            write_json(root / "sweep.json", {"experiments": experiments})

            def accounting(command, **kwargs):
                job = command[command.index("-j") + 1]
                return f"{job}|3600|COMPLETED\n"

            with patch(
                "clustering_mopd.estimate.subprocess.check_output",
                side_effect=accounting,
            ):
                result = calibrate_completed(root)
            self.assertEqual([r["allocated_hours"] for r in result["runs"]], [2, 2])
            self.assertTrue(result["queue_gaps_excluded"])

    def test_larger_id_panel_adds_evaluation_time(self):
        calibration = {
            "runs": [
                {
                    "label": "baseline",
                    "teacher_rollouts": 112000,
                    "allocated_hours": 12,
                    "student_rollouts": 6400,
                    "student_seconds_per_update_median": 16,
                    "id_eval_prompts": 32,
                }
            ],
            "seconds_per_256_sample_gate_median": 120,
        }
        plan = {"cluster_count": 8, "simulated_devices": 8, "id_eval_prompts": 32}
        old = completed_range(plan, calibration, 112000, 12000, 30)
        plan["id_eval_prompts"] = 64
        new = completed_range(plan, calibration, 112000, 12000, 30)
        self.assertGreater(
            new["physical_hours_range"][0], old["physical_hours_range"][0]
        )
        self.assertGreater(
            new["physical_hours_range"][1], old["physical_hours_range"][1]
        )
        plan["simulated_devices"] = 9
        with self.assertRaisesRegex(ValueError, "eight simulated"):
            completed_range(plan, calibration, 112000, 12000, 30)


if __name__ == "__main__":
    unittest.main()
