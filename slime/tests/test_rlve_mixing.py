import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

module_spec = importlib.util.spec_from_file_location(
    "rlve_mixing", Path(__file__).parents[1] / "utils" / "rlve_mixing.py"
)
rlve_mixing = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(rlve_mixing)
FrozenPromptMix = rlve_mixing.FrozenPromptMix
load_candidate = rlve_mixing.load_candidate


class FrozenPromptMixTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.pool_path = self.folder / "pool.json"
        self.pool_path.write_text(json.dumps([
            {"user_prompt": f"{environment} {index}", "metadata": json.dumps({"environment": environment, "config": {"seed": index}})}
            for environment in ("A", "B") for index in range(3)
        ]))
        self.pool_hash = hashlib.sha256(self.pool_path.read_bytes()).hexdigest()
        self.manifest_path = self.folder / "mixes.json"
        self.manifest_path.write_text(json.dumps({
            "environments": ["A", "B"],
            "pool_sha256": self.pool_hash,
            "candidates": [{"alpha": 0.5, "weights": {"A": 0.8, "B": 0.2}}],
        }))

    def test_fixed_pool_sampling_and_checkpoint_state(self):
        manifest, candidate = load_candidate(self.manifest_path, 0, ["A", "B"], self.pool_path)
        first = FrozenPromptMix(self.pool_path, ["A", "B"], candidate["weights"], 7, manifest["pool_sha256"])
        draws = [first.draw() for _ in range(1000)]
        self.assertTrue(all(prompt.startswith(environment) for environment, prompt, _ in draws))
        self.assertTrue(all(config["seed"] in range(3) for _, _, config in draws))
        self.assertTrue(740 < sum(environment == "A" for environment, _, _ in draws) < 860)
        state = first.get_state()
        expected = [first.draw() for _ in range(20)]
        resumed = FrozenPromptMix(self.pool_path, ["A", "B"], candidate["weights"], 7, manifest["pool_sha256"])
        resumed.set_state(state)
        self.assertEqual([resumed.draw() for _ in range(20)], expected)

    def test_mismatched_pool_is_rejected(self):
        self.pool_path.write_text("[]")
        with self.assertRaisesRegex(ValueError, "differs"):
            load_candidate(self.manifest_path, 0, ["A", "B"], self.pool_path)


if __name__ == "__main__":
    unittest.main()
