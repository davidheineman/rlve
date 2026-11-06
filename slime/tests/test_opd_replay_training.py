"""CPU integration of the real rollout controller, using mocked GPU/Ray calls."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from slime.ray.buffer import RolloutController
from slime.tests.test_opd_replay import trace
from slime.utils.opd_replay import OPDReplayBuffer
from slime.utils.types import Sample


class ReplayControllerTest(unittest.TestCase):
    def test_generate_scores_only_fresh_environment_and_preserves_cached_scores(self):
        cls = RolloutController.__ray_metadata__.modified_class
        controller = cls.__new__(cls)
        controller.args = SimpleNamespace(
            opd_replay_fraction=.5, rollout_batch_size=8, n_samples_per_prompt=1,
            environment_list=["B"], seed=123, load_debug_rollout_data=None,
            wandb_always_use_train_step=True,
            global_batch_size=8, loss_type="custom_loss", save_debug_rollout_data=None,
            save_rollouts_jsonl=None, opd_teacher_dir="bank", use_wandb=False,
            advantage_estimator="none", reward_key="reward", rewards_normalization=False,
        )
        controller.opd_replay = OPDReplayBuffer(2)
        controller.opd_replay.add([trace("A")])
        controller.data_source = None
        controller.custom_reward_post_process_func = None
        generated = []
        scored = []

        def generate(args, step, source, evaluation=False):
            generated.append(args.opd_fresh_batch_size)
            data = []
            for index in range(args.opd_fresh_batch_size):
                value = trace("B", index)
                value["teacher_log_probs"] = None
                data.append([Sample.from_dict(value)])
            return data

        def score(trajectories):
            scored.extend(trajectories)
            return [(index, [-.25, -.5]) for index, _, _, _ in trajectories]

        controller.generate_rollout = generate
        controller.teacher_scorers = [SimpleNamespace(score=SimpleNamespace(remote=score))]
        with patch("slime.ray.buffer.ray.get", side_effect=lambda x: x), \
             patch("slime.ray.buffer.ray.put", side_effect=lambda x: x), \
             patch("slime.ray.buffer.log_rollout_data"), redirect_stdout(io.StringIO()):
            data = controller.generate(0).inner
        self.assertEqual(generated, [4])
        self.assertEqual(len(scored), 4)
        self.assertTrue(all(env == "B" for _, env, _, _ in scored))
        self.assertEqual(data["opd_replay_mask"], [[0., 0.]]*4 + [[1., 1.]]*4)
        self.assertEqual(data["teacher_log_probs"], [[-.25, -.5]]*4 + [[-.5, -2.]]*4)
        self.assertEqual(data["rollout_log_probs"], [[-1., -1.]]*8)
        self.assertEqual(data["loss_masks"], [[0, 1]]*8)
        self.assertEqual(controller.opd_replay.size, 3)  # A retained, latest two B.
        self.assertIsNone(controller.args.opd_fresh_batch_size)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"snapshot.gz"
            controller.save_opd_replay(str(path))
            loaded = OPDReplayBuffer.load(path, 2)
            self.assertEqual(loaded.size, 3)


if __name__ == "__main__":
    unittest.main()
