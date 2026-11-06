"""Exercise the actual training loop with fake actor/Ray boundaries, without GPUs."""

import ast
import contextlib
import io
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch


# Import just the driver function: the full module imports GPU runtime packages.
source = Path(__file__).resolve().parents[2] / "train.py"
function = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "train")
compiled = compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec")


class FinalWeightsOnlyTest(unittest.TestCase):
    def run_training(self, scores, final_only=True, offload=True):
        actor, rollout = Mock(), Mock()
        actor.async_init.return_value = [0]
        args = SimpleNamespace(
            transactional_checkpoints=False, save="final-ckpt", save_interval=2,
            opd_teacher_dir=None, dpo_online=False, num_rollout=4, kl_coef=0,
            use_kl_loss=False, loss_type="policy_loss", start_rollout_id=None,
            rollout_global_dataset=False, rlve=True, offload=offload,
            fixed_rollout_policy=False, n_samples_per_prompt=8,
            rollout_anneal_thresholds=None, eval_interval=2, eval_on_last_rollout=True,
            early_stop_patience_evals=None, early_stop_val_accuracy=.98,
            require_early_stop=False, use_wandb=False,
        )
        namespace = {
            "os": os, "time": time, "ray": SimpleNamespace(get=lambda value: value),
            "create_placement_groups": lambda args: {"actor": None, "rollout": None},
            "init_wandb_primary": lambda args: None,
            "create_actor_group": lambda *args, **kwargs: actor,
            "create_rollout_manager": lambda *args, **kwargs: rollout,
            "evaluate_student": Mock(side_effect=scores),
            "log_step_perf": Mock(), "is_final_ood_update": lambda *args: False,
            "GPU_MEMORY_TYPE_WEIGHTS": "weights", "GPU_MEMORY_TYPE_KV_CACHE": "kv",
        }
        exec(compiled, namespace)
        with patch.dict(os.environ, {"RLVE_FINAL_WEIGHTS_ONLY": "1" if final_only else "0"}), contextlib.redirect_stdout(io.StringIO()):
            namespace["train"](args)
        return actor, rollout

    def test_budget_stop_saves_once_and_never_saves_loader(self):
        actor, rollout = self.run_training([.1, .2, .3])
        self.assertEqual(actor.async_train.call_count, 4)
        actor.async_save_model.assert_called_once_with(3, with_optimizer=False)
        rollout.controller.save.remote.assert_not_called()
        rollout.controller.save_runtime.remote.assert_not_called()
        self.assertLess(actor.mock_calls.index(call.async_onload()),
                        actor.mock_calls.index(call.async_save_model(3, with_optimizer=False)))

    def test_early_stop_saves_the_early_terminal_weights(self):
        actor, rollout = self.run_training([.1, .98])
        self.assertEqual(actor.async_train.call_count, 2)
        actor.async_save_model.assert_called_once_with(1, with_optimizer=False)
        rollout.controller.save.remote.assert_not_called()

    def test_non_offloaded_actor_saves_final_weights(self):
        actor, _ = self.run_training([.1, .2, 1.], offload=False)
        actor.async_save_model.assert_called_once_with(3, with_optimizer=False)
        actor.async_onload.assert_not_called()

    def test_existing_periodic_checkpoint_mode_still_saves_loader(self):
        actor, rollout = self.run_training([.1, .2, .3], final_only=False)
        self.assertEqual(actor.async_save_model.call_args_list,
                         [call(1, with_optimizer=True), call(3, with_optimizer=True)])
        self.assertEqual(rollout.controller.save.remote.call_args_list, [call(1), call(3)])


if __name__ == "__main__":
    unittest.main()
