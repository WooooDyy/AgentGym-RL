import unittest

import numpy as np
import torch

from verl.agent_trainer.ppo.wmc_erc import (
    apply_wmc_erc,
    compute_dynamic_mask,
    compute_h_wm,
    compute_s_star,
    compute_turn_boundaries,
)


class MockBatch:
    def __init__(self, tensors):
        self.batch = tensors


class TestComputeSStar(unittest.TestCase):
    def test_single_turn(self):
        probs = torch.tensor([0.8, 0.6, 0.9, 0.7], dtype=torch.float32)
        old_log_probs = torch.zeros(1, 6, dtype=torch.float32)
        old_log_probs[0, :4] = torch.log(probs)
        entropys = torch.tensor([[1.0, 1.5, 0.5, 1.2, 0.0, 0.0]], dtype=torch.float32)
        response_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.float32)

        boundaries = compute_turn_boundaries(response_mask)
        result = compute_s_star(old_log_probs, entropys, response_mask, boundaries)

        expected = torch.mean(probs * (entropys[0, :4] + torch.log(probs)))
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        self.assertAlmostEqual(result[0][0].item(), expected.item(), places=5)


class TestComputeHWM(unittest.TestCase):
    def test_uses_observation_mask(self):
        entropys = torch.tensor([[0.5, 0.6, 5.0, 3.0, 7.0, 0.0]], dtype=torch.float32)
        observation_mask = torch.tensor([[0, 0, 0, 1, 0, 0]], dtype=torch.float32)
        boundaries = [[(0, 2)]]

        result = compute_h_wm(entropys, observation_mask, boundaries)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        self.assertAlmostEqual(result[0][0].item(), 3.0, places=5)


class TestComputeDynamicMask(unittest.TestCase):
    def test_asymmetric_behavior(self):
        s_star = [[torch.tensor(15.0)], [torch.tensor(5.0)]]
        h_wm = [[torch.tensor(1.0)], [torch.tensor(1.0)]]

        high_blocked = compute_dynamic_mask(
            s_star,
            h_wm,
            mu_base=0.1,
            mu_exp=10.0,
            eta_wm=1.0,
            lambda_wm=0.0,
            s_bar=10.0,
            sigma=5.0,
        )
        self.assertEqual(high_blocked[0], [0.0])
        self.assertEqual(high_blocked[1], [1.0])

        low_blocked = compute_dynamic_mask(
            s_star,
            h_wm,
            mu_base=10.0,
            mu_exp=0.1,
            eta_wm=1.0,
            lambda_wm=0.0,
            s_bar=10.0,
            sigma=5.0,
        )
        self.assertEqual(low_blocked[0], [1.0])
        self.assertEqual(low_blocked[1], [0.0])


class TestApplyWmcErc(unittest.TestCase):
    def _make_batch(
        self,
        advantages,
        response_mask,
        old_log_probs,
        attention_mask,
        observation_mask=None,
    ):
        tensors = {
            "advantages": advantages.clone(),
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "attention_mask": attention_mask,
        }
        if observation_mask is not None:
            tensors["observation_mask"] = observation_mask
        return MockBatch(tensors)

    def test_batch_mode_uses_batch_stats(self):
        response_mask = torch.ones(4, 4, dtype=torch.float32)
        advantages = torch.ones(4, 4, dtype=torch.float32) * 2.0
        old_log_probs = torch.tensor(
            [[np.log(0.3)] * 4, [np.log(0.3)] * 4, [np.log(0.3)] * 4, [np.log(0.9)] * 4],
            dtype=torch.float32,
        )
        entropys = torch.tensor([[2.0] * 4, [2.0] * 4, [2.0] * 4, [3.0] * 4], dtype=torch.float32)
        attention_mask = torch.ones(4, 4, dtype=torch.float32)
        batch = self._make_batch(advantages, response_mask, old_log_probs, attention_mask)

        config = {"enable": True, "mu_base": 0.1, "mu_exp": 10.0, "eta_wm": 1.0, "lambda_wm": 1.0, "clipping_type": "batch"}
        running_stats = {"s_bar": 100.0, "s_std": 1.0, "h_bar": 1.0, "initialized": True}

        batch, metrics = apply_wmc_erc(batch, entropys, config, running_stats)

        self.assertIn("wmc_erc/mask_ratio", metrics)
        self.assertTrue((batch.batch["advantages"][3] == 0).all())

    def test_global_mode_uses_running_stats(self):
        response_mask = torch.ones(4, 4, dtype=torch.float32)
        advantages = torch.ones(4, 4, dtype=torch.float32) * 2.0
        old_log_probs = torch.tensor([[np.log(0.3)] * 4] * 4, dtype=torch.float32)
        entropys = torch.tensor([[2.0] * 4] * 4, dtype=torch.float32)
        attention_mask = torch.ones(4, 4, dtype=torch.float32)
        batch = self._make_batch(advantages, response_mask, old_log_probs, attention_mask)

        config = {"enable": True, "mu_base": 10.0, "mu_exp": 0.1, "eta_wm": 1.0, "lambda_wm": 0.0, "clipping_type": "global"}
        running_stats = {"s_bar": 10.0, "s_std": 1.0, "h_bar": 1.0, "initialized": True}

        batch, _ = apply_wmc_erc(batch, entropys, config, running_stats)

        self.assertTrue((batch.batch["advantages"] == 0).all())

    def test_observation_mask_drives_h_wm_and_wm_nll(self):
        response_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)
        advantages = torch.ones(1, 5, dtype=torch.float32)
        old_log_probs = torch.tensor([[np.log(0.5), np.log(0.5), np.log(0.5), np.log(0.2), np.log(0.9)]], dtype=torch.float32)
        entropys = torch.tensor([[1.0, 1.0, 1.0, 7.0, 8.0]], dtype=torch.float32)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.float32)
        explicit_observation_mask = torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.float32)
        batch = self._make_batch(advantages,
                                 response_mask,
                                 old_log_probs,
                                 attention_mask,
                                 observation_mask=explicit_observation_mask)

        config = {"enable": True, "mu_base": 10.0, "mu_exp": 10.0, "eta_wm": 1.0, "lambda_wm": 0.0, "clipping_type": "batch"}
        running_stats = {"initialized": False}

        _, metrics = apply_wmc_erc(batch, entropys, config, running_stats)

        self.assertAlmostEqual(metrics["wmc_erc/batch_h_bar"], 7.0, places=5)
        self.assertAlmostEqual(metrics["wmc_erc/wm_nll"], -np.log(0.2), places=5)


if __name__ == "__main__":
    unittest.main()
