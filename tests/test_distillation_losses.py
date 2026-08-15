import unittest

import torch
import torch.nn.functional as F

from opsd_trainer import OPSDTrainer


class TopKForwardKLTests(unittest.TestCase):
    def test_teacher_is_truncated_but_student_keeps_full_vocab_denominator(self):
        student_logits = torch.tensor([[[0.0, 0.0, 10.0]]])
        teacher_logits = torch.tensor([[[10.0, 9.0, -10.0]]])

        actual = OPSDTrainer.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            beta=0,
            top_k=2,
            reduction="sum",
        )

        _, indices = torch.topk(teacher_logits, k=2, dim=-1)
        teacher_top_k_log_probs = F.log_softmax(torch.gather(teacher_logits, -1, indices), dim=-1)
        student_full_log_probs = F.log_softmax(student_logits, dim=-1)
        expected = F.kl_div(
            torch.gather(student_full_log_probs, -1, indices),
            teacher_top_k_log_probs,
            reduction="sum",
            log_target=True,
        )
        student_truncated_log_probs = F.log_softmax(torch.gather(student_logits, -1, indices), dim=-1)
        incorrectly_renormalized = F.kl_div(
            student_truncated_log_probs,
            teacher_top_k_log_probs,
            reduction="sum",
            log_target=True,
        )

        torch.testing.assert_close(actual, expected)
        self.assertGreater((actual - incorrectly_renormalized).abs().item(), 1.0)

    def test_top_k_rejects_non_forward_kl(self):
        logits = torch.zeros((1, 1, 3))
        with self.assertRaisesRegex(ValueError, "forward KL"):
            OPSDTrainer.generalized_jsd_loss(
                student_logits=logits,
                teacher_logits=logits,
                beta=0.5,
                top_k=2,
            )

    def test_top_k_rejects_values_larger_than_vocabulary(self):
        logits = torch.zeros((1, 1, 3))
        with self.assertRaisesRegex(ValueError, "vocabulary size"):
            OPSDTrainer.generalized_jsd_loss(
                student_logits=logits,
                teacher_logits=logits,
                beta=0,
                top_k=4,
            )


if __name__ == "__main__":
    unittest.main()
