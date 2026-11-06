"""N64 panel identity, compatible reuse, and leakage-free merge controls."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from lora_clustering.balanced_pipeline import confirmation_assignments
from lora_clustering.plan import build_clustering_plan, validate_evaluation
from lora_clustering.reuse import describe_reuse, import_reuse
from lora_teachers.plan import sha256


class N64Test(unittest.TestCase):
    def test_balanced_pipeline_completes_and_resumes_with_cached_selection(self):
        import asyncio

        try:
            import torch  # noqa: F401
            from lora_clustering.balanced_pipeline import run_balanced
            from lora_clustering.evaluation import Evaluator  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(
                f"Pipeline runtime checks need the pinned image: {error}"
            )

        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary)
            output = experiment / "plots"
            plan = build_clustering_plan(size=64)
            plan.update(
                size=4,
                environments=plan["environments"][:4],
                teachers=plan["teachers"][:4],
                plot_output=str(output),
            )
            plan["clustering"]["balanced_restarts"] = 2
            rows = {}
            for split in ("selection", "confirmation"):
                all_rows = json.loads((ROOT / plan["clustering"][split]).read_text())
                selected = [
                    r
                    for r in all_rows
                    if json.loads(r["metadata"])["environment"] in plan["environments"]
                ]
                rows[split] = {
                    name: [
                        r
                        for r in selected
                        if json.loads(r["metadata"])["environment"] == name
                    ]
                    for name in plan["environments"]
                }
                path = experiment / "source/rlve" / plan["clustering"][split]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(selected))
                plan["clustering"][f"{split}_sha256"] = sha256(path)
            (experiment / "plan.json").write_text(json.dumps(plan))

            class FakeEvaluator:
                def __init__(self, *args):
                    pass

                def cached(self, split, label, environment):
                    path = (
                        experiment
                        / "clustering/evaluations"
                        / split
                        / label
                        / f"{environment}.json"
                    )
                    return json.loads(path.read_text()) if path.exists() else None

                async def sampler(self, label, **kwargs):
                    return label

                async def cell(self, split, label, environment, sampler):
                    path = (
                        experiment
                        / "clustering/evaluations"
                        / split
                        / label
                        / f"{environment}.json"
                    )
                    if path.exists():
                        return json.loads(path.read_text())
                    target = plan["environments"].index(environment)
                    donor = (
                        int(label.removeprefix("individual-"))
                        if label.startswith("individual-")
                        else -10
                    )
                    correct = int(donor // 2 == target // 2)
                    total = plan["clustering"][f"{split}_samples_per_prompt"]
                    cell = {
                        "eval_sha256": plan["clustering"][f"{split}_sha256"],
                        "accuracy": correct / total,
                        "observations": [
                            {
                                "prompt_id": hashlib.sha256(
                                    r["user_prompt"].encode()
                                ).hexdigest(),
                                "correct": correct,
                                "total": total,
                            }
                            for r in rows[split][environment]
                        ],
                    }
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(cell))
                    return cell

                async def vector(self, split, label, **kwargs):
                    return [
                        (await self.cell(split, label, name, label))["accuracy"]
                        for name in plan["environments"]
                    ]

            results = [
                {
                    "hf": str(experiment / "dummy"),
                    "environment": name,
                    "training_rollouts": 8000,
                    "accuracy": 0.5,
                    "stop_reason": "budget",
                    "wandb_url": "test",
                }
                for name in plan["environments"]
            ]
            with (
                patch("lora_clustering.evaluation.Evaluator", FakeEvaluator),
                patch(
                    "lora_clustering.pipeline.cluster_adapter",
                    return_value=experiment / "dummy",
                ),
                patch.dict(
                    "os.environ", {"LORA_CHECKPOINT_ROOT": str(experiment / "local")}
                ),
            ):
                first = asyncio.run(
                    run_balanced(experiment, plan, results, "base", "fake")
                )
                with patch(
                    "analyze_balanced_lora_clustering.analyze",
                    side_effect=AssertionError("Must reuse frozen selection"),
                ):
                    second = asyncio.run(
                        run_balanced(experiment, plan, results, "base", "fake")
                    )
            self.assertEqual(first, second)
            self.assertEqual(first["selection"]["selected"]["k"], 2)
            self.assertTrue((output / "confirmation.json").is_file())
            self.assertTrue((output / "balanced/k_selection.png").is_file())

    def test_original_n64_panel_and_recipe_are_preserved(self):
        plan = validate_evaluation(build_clustering_plan(size=64))
        old = build_clustering_plan()
        runs = json.loads((ROOT / "data/DATA-SCALING-V1/run_plan.json").read_text())[
            "runs"
        ]
        original = next(r for r in runs if r["label"] == "n64-mopd")
        self.assertEqual(plan["environments"], original["environments"])
        self.assertEqual(plan["environments"][:32], old["environments"])
        self.assertEqual(len(plan["teachers"]), 64)
        self.assertEqual(plan["teacher_config"], old["teacher_config"])
        self.assertEqual(plan["teacher_rollouts"], 8000)
        self.assertEqual(plan["teacher_validation_threshold"], 0.98)
        self.assertEqual(plan["clustering"]["mode"], "balanced")
        self.assertEqual(plan["clustering"]["balanced_candidate_k"], [2, 4, 8, 16, 32])
        for split in ("selection", "confirmation"):
            before = json.loads((ROOT / old["clustering"][split]).read_text())
            after = json.loads((ROOT / plan["clustering"][split]).read_text())
            self.assertEqual(after[: len(before)], before)

    def test_merged_transfer_controls_exclude_target_and_match_size(self):
        groups = [list(range(i, i + 4)) for i in range(0, 64, 4)]
        result = confirmation_assignments(groups, 1234)
        self.assertEqual(result, confirmation_assignments(groups, 1234))
        for key in ("selected-loo", "random-loo", "outside-matched"):
            for i, members in enumerate(result["controls"][key]):
                self.assertNotIn(i, members)
                self.assertEqual(len(members), 3)
                self.assertEqual(len(set(members)), 3)
                if key == "outside-matched":
                    self.assertFalse(set(members) & set(groups[i // 4]))
        for i, members in enumerate(result["controls"]["selected-full"]):
            self.assertIn(i, members)
            self.assertEqual(len(members), 4)


class ReuseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "old"
        self.target = Path(self.temp.name) / "new"
        self.source.mkdir()
        self.target.mkdir()
        self.old = {
            key: 1
            for key in (
                "optimizer_eps",
                "grad_clip_norm",
                "tis_clip_low",
                "tis_clip_high",
            )
        }
        self.old.update(
            project="test",
            base_model="base",
            teacher_config={"lr": 2e-5},
            lora_rank=32,
            lora_alpha=64,
            target_modules="attn,mlp",
            seed=1234,
            teacher_rollouts=8000,
            teacher_validation_threshold=0.98,
        )
        self.old["environments"] = ["A", "B"]
        self.old["teachers"] = [
            {
                "environment": name,
                "source_label": f"teacher-{i:02d}",
                "eval_sha256": name,
            }
            for i, name in enumerate(self.old["environments"])
        ]
        self.new = {**self.old, "environments": ["A", "B", "C", "D"]}
        self.new["teachers"] = self.old["teachers"] + [
            {
                "environment": name,
                "source_label": f"teacher-{i:02d}",
                "eval_sha256": name,
            }
            for i, name in enumerate(("C", "D"), start=2)
        ]
        for plan, experiment in ((self.old, self.source), (self.new, self.target)):
            plan["clustering"] = {}
            for split in ("selection", "confirmation"):
                relative = f"data/{split}.json"
                path = experiment / "source/rlve" / relative
                rows = [
                    {
                        "user_prompt": f"{split}-{name}",
                        "metadata": {"environment": name},
                    }
                    for name in plan["environments"]
                ]
                self.write(path, rows)
                plan["clustering"].update(
                    {
                        split: relative,
                        f"{split}_sha256": sha256(path),
                        f"{split}_samples_per_prompt": 2,
                    }
                )
            self.write(experiment / "plan.json", plan)
        for teacher in self.old["teachers"]:
            directory = self.source / "teacher" / teacher["source_label"]
            adapter = directory / "final/adapter"
            self.write(adapter / "adapter_config.json", {"r": 32})
            self.write(
                adapter / "META.json",
                {
                    "tenant_digest": hashlib.sha256(
                        f"tml-async-{self.source.name}".encode()
                    ).hexdigest()
                },
            )
            self.write(
                adapter.parent / "reference.json",
                {
                    "sampler_uri": f"tinker://{teacher['source_label']}/sampler_weights/final",
                    "state_uri": None,
                },
            )
            (adapter / "adapter_model.safetensors").write_bytes(
                b"sentinel adapter bytes"
            )
            self.write(
                directory / "completed.json",
                {
                    "hf": str(adapter),
                    "environment": teacher["environment"],
                    "training_rollouts": 8000,
                    "accuracy": 0.5,
                    "resume": None,
                },
            )
        for split in ("selection", "confirmation"):
            for name in self.old["environments"]:
                self.write(
                    self.source
                    / "clustering/evaluations"
                    / split
                    / "base"
                    / f"{name}.json",
                    {
                        "adapter": "base",
                        "eval_sha256": self.old["clustering"][f"{split}_sha256"],
                        "observations": [
                            {
                                "prompt_id": hashlib.sha256(
                                    f"{split}-{name}".encode()
                                ).hexdigest(),
                                "correct": 1,
                                "total": 2,
                            }
                        ],
                    },
                )
        self.write(self.source / "completed.json", {"stage": "complete"})
        self.write(self.target / "input_sha256.json", {})
        self.new["reuse"] = describe_reuse(self.source, self.new)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_import_copies_adapters_and_preserves_cell_provenance(self):
        record = import_reuse(self.target, self.source, self.new)
        self.assertEqual(
            record["evaluation_cells"], {"selection": 2, "confirmation": 2}
        )
        cell = json.loads(
            (self.target / "clustering/evaluations/selection/base/A.json").read_text()
        )
        self.assertEqual(
            cell["eval_sha256"], self.new["clustering"]["selection_sha256"]
        )
        self.assertEqual(
            cell["reused_from"]["eval_sha256"],
            self.old["clustering"]["selection_sha256"],
        )
        result = json.loads(
            (self.target / "teacher/teacher-00/completed.json").read_text()
        )
        self.assertTrue(result["hf"].startswith(str(self.target)))
        adapter = Path(result["hf"])
        original = self.source / "teacher/teacher-00/final/adapter"
        self.assertEqual(
            sha256(adapter / "adapter_model.safetensors"),
            sha256(original / "adapter_model.safetensors"),
        )
        self.assertTrue((adapter.parent / "reference.json").is_file())
        self.assertEqual(
            json.loads((adapter / "META.json").read_text())["tenant_digest"],
            hashlib.sha256(f"tml-async-{self.target.name}".encode()).hexdigest(),
        )
        manifest = json.loads((self.target / "input_sha256.json").read_text())
        for relative, expected in manifest.items():
            self.assertEqual(sha256(self.target / relative), expected)

    def test_changed_training_recipe_cannot_reuse(self):
        self.new["teacher_config"] = {"lr": 1e-4}
        with self.assertRaisesRegex(ValueError, "same training recipe"):
            describe_reuse(self.source, self.new)

    def test_changed_probe_input_cannot_reuse_cells(self):
        path = self.target / "source/rlve/data/selection.json"
        rows = json.loads(path.read_text())
        rows[0]["user_prompt"] = "different prompt"
        self.write(path, rows)
        self.new["clustering"]["selection_sha256"] = sha256(path)
        with self.assertRaisesRegex(ValueError, "Per-task selection inputs changed"):
            import_reuse(self.target, self.source, self.new)


if __name__ == "__main__":
    unittest.main()
