import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import torch


spec = spec_from_file_location("opd", Path(__file__).parents[1] / "utils" / "opd.py")
opd = module_from_spec(spec)
spec.loader.exec_module(opd)


class ToyTeacher:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, input_ids, use_cache):
        assert input_ids.tolist() == [[1, 2, 3, 1]]
        assert use_cache is False
        return SimpleNamespace(logits=self.logits)


class OPDTest(unittest.TestCase):
    def test_teacher_scores_only_student_response_tokens(self):
        logits = torch.tensor([[[0., 1., 2., 3.], [1., 2., 3., 4.], [4., 3., 2., 1.], [0., 0., 0., 0.]]])
        teacher = ToyTeacher(logits)
        scores = opd.score_response_log_probs(teacher, [1, 2, 3, 1], 2, torch.device("cpu"))
        expected = [
            torch.log_softmax(logits[0, 1], dim=-1)[3].item(),
            torch.log_softmax(logits[0, 2], dim=-1)[1].item(),
        ]
        for actual, target in zip(scores, expected):
            self.assertAlmostEqual(actual, target)

    def test_reverse_kl_gradient_prefers_teacher_likely_tokens(self):
        student = torch.tensor([-1., -1.], requires_grad=True)
        behavior = torch.tensor([-1., -1.])
        teacher = torch.tensor([-0.5, -2.])
        loss, metrics = opd.sampled_reverse_kl_loss(student, behavior, teacher, torch.mean)
        loss.backward()
        self.assertTrue(torch.allclose(student.grad, torch.tensor([-0.25, 0.5])))
        self.assertAlmostEqual(metrics["opd_reverse_kl"].item(), 0.25)

    def test_eos_transport_changes_only_response_targets_and_preserves_context(self):
        logits = torch.tensor([[[0., 1., 2., 3.], [1., 2., 3., 4.], [4., 3., 2., 1.], [0., 0., 0., 0.]]])
        teacher = ToyTeacher(logits)
        normal = opd.score_response_log_probs(teacher, [1, 2, 3, 1], 2, torch.device("cpu"))
        aligned = opd.score_response_log_probs(teacher, [1, 2, 3, 1], 2, torch.device("cpu"),
                                               stop_token_permutation=(1, 0))
        self.assertEqual(normal[0], aligned[0])
        self.assertAlmostEqual(aligned[1], logits[0, 2].log_softmax(-1)[0].item())
        # ToyTeacher asserts the original prompt and response input IDs; none
        # are replaced in the conditioning history, including prompt token 1.

    def test_eos_transport_is_a_normalized_permutation_including_both_aliases(self):
        logits = torch.tensor([[[4., -10., 0., -2.], [0., 0., 0., 0.]]])
        class Teacher:
            def __call__(self, input_ids, use_cache):
                return SimpleNamespace(logits=logits)
        values = [opd.score_response_log_probs(Teacher(), [2, target], 1, torch.device('cpu'),
                    stop_token_permutation=(1, 0))[0] for target in range(4)]
        probabilities = torch.tensor(values).exp()
        self.assertTrue(torch.allclose(probabilities.sum(), torch.tensor(1.)))
        self.assertTrue(torch.allclose(probabilities, logits[0, 0].softmax(-1)[torch.tensor([1, 0, 2, 3])]))

    def test_stop_alias_alignment_corrects_the_direction_of_the_stop_gradient(self):
        for teacher_logp, expected_sign in [(-25., 1), (-.01, -1)]:
            student = torch.tensor([-1.], requires_grad=True)
            loss, _ = opd.sampled_reverse_kl_loss(student, student.detach(), torch.tensor([teacher_logp]), torch.mean)
            loss.backward()
            self.assertEqual(int(student.grad.sign().item()), expected_sign)

    def test_eos_transport_rejects_unrecognized_intermediate_stop_tokens(self):
        logits = torch.zeros(1, 4, 4)
        with self.assertRaisesRegex(ValueError, 'stop at either EOS'):
            opd.score_response_log_probs(ToyTeacher(logits), [1, 2, 3, 1], 2, torch.device('cpu'),
                                         stop_token_permutation=(3, 0))

    def test_matching_teacher_has_zero_gradient(self):
        student = torch.tensor([-1., -2.], requires_grad=True)
        behavior = student.detach().clone()
        loss, _ = opd.sampled_reverse_kl_loss(student, behavior, behavior, torch.mean)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))

    def test_misaligned_scores_fail(self):
        with self.assertRaises(ValueError):
            opd.sampled_reverse_kl_loss(torch.zeros(2), torch.zeros(1), torch.zeros(2), torch.mean)

    def test_replay_retains_signed_advantages_instead_of_sft(self):
        student = torch.tensor([-1., -1.], requires_grad=True)
        behavior = torch.tensor([-1., -1.])
        teacher = torch.tensor([-.5, -2.])
        loss, _ = opd.sampled_reverse_kl_loss(student, behavior, teacher, torch.mean,
                                             replay_mask=torch.ones(2))
        loss.backward()
        self.assertTrue(torch.allclose(student.grad, torch.tensor([-.25, .5])))

    def test_stale_replay_uses_behavior_ratio_and_clips_only_replay(self):
        student = torch.tensor([-1., -1., -1.], requires_grad=True)
        behavior = torch.tensor([-2., -2., -2.])
        teacher = torch.tensor([-1., -3., -1.])
        loss, _ = opd.sampled_reverse_kl_loss(student, behavior, teacher, torch.mean,
                                             replay_mask=torch.tensor([1., 1., 0.]))
        loss.backward()
        # Positive replay is clipped; negative replay is discouraged; fresh OPD
        # retains its original unclipped gradient even at a large ratio.
        self.assertEqual(student.grad[0].item(), 0)
        self.assertAlmostEqual(student.grad[1].item(), torch.exp(torch.tensor(1.)).item()/3, places=6)
        self.assertAlmostEqual(student.grad[2].item(), -torch.exp(torch.tensor(1.)).item()/3, places=6)

    def test_answer_forcing_mask_zeroes_replayed_prefix_gradients(self):
        student = torch.tensor([-1., -1.], requires_grad=True)
        behavior = torch.tensor([-1., -1.])
        teacher = torch.tensor([-2., -.5])
        mask = torch.tensor([0., 1.])
        loss, _ = opd.sampled_reverse_kl_loss(student, behavior, teacher,
                                             lambda x: (x*mask).sum()/mask.sum(),
                                             replay_mask=torch.ones(2))
        loss.backward()
        self.assertTrue(torch.allclose(student.grad, torch.tensor([0., -.5])))


if __name__ == "__main__":
    unittest.main()
