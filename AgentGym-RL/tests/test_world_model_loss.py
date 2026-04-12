import sys
import types
import unittest

import torch
from omegaconf import OmegaConf
from tensordict import TensorDictBase


if not hasattr(torch, "compile"):
    torch.compile = lambda fn, **kwargs: fn  # type: ignore[attr-defined]
else:
    torch.compile = lambda fn, **kwargs: fn  # type: ignore[assignment]


flash_attn_module = types.ModuleType("flash_attn")
bert_padding_module = types.ModuleType("flash_attn.bert_padding")


def _unsupported(*args, **kwargs):
    raise NotImplementedError("flash_attn stubs in test should not be called")


bert_padding_module.pad_input = _unsupported
bert_padding_module.unpad_input = _unsupported
bert_padding_module.rearrange = _unsupported
bert_padding_module.index_first_axis = _unsupported
flash_attn_module.bert_padding = bert_padding_module
sys.modules.setdefault("flash_attn", flash_attn_module)
sys.modules.setdefault("flash_attn.bert_padding", bert_padding_module)


from verl import DataProto
from verl.agent_trainer.ppo.world_model_loss import compute_observation_mask, compute_world_model_loss
from verl.workers.agent_actor.dp_actor import DataParallelPPOActor


class DummyActorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


class TestWorldModelLossHelpers(unittest.TestCase):
    def test_compute_observation_mask_fallback(self):
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.float32)
        response_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)

        observation_mask = compute_observation_mask(attention_mask=attention_mask, response_mask=response_mask)

        self.assertTrue(torch.equal(observation_mask, torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.float32)))

    def test_compute_observation_mask_prefers_explicit(self):
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.float32)
        response_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)
        explicit_observation_mask = torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.float32)

        observation_mask = compute_observation_mask(attention_mask=attention_mask,
                                                    response_mask=response_mask,
                                                    observation_mask=explicit_observation_mask)

        self.assertTrue(torch.equal(observation_mask, explicit_observation_mask))

    def test_compute_world_model_loss(self):
        log_prob = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.float32)
        response_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)

        wm_sft_loss, observation_mask = compute_world_model_loss(log_prob=log_prob,
                                                                 attention_mask=attention_mask,
                                                                 response_mask=response_mask)

        self.assertTrue(torch.equal(observation_mask, torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)))
        self.assertAlmostEqual(wm_sft_loss.item(), -3.5, places=5)


class TestWorldModelLossActorIntegration(unittest.TestCase):
    def _build_actor(self, world_model_coeff: float):
        config = OmegaConf.create({
            "use_remove_padding": False,
            "ulysses_sequence_parallel_size": 1,
            "grad_clip": 1.0,
            "clip_ratio": 0.2,
            "entropy_coeff": 0.0,
            "use_kl_loss": False,
            "world_model_coeff": world_model_coeff,
            "ppo_mini_batch_size": 1,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_epochs": 1,
            "shuffle": False,
            "use_dynamic_bsz": False,
        })
        actor_module = DummyActorModule()
        optimizer = torch.optim.SGD(actor_module.parameters(), lr=0.1)
        actor = DataParallelPPOActor(config=config, actor_module=actor_module, actor_optimizer=optimizer)

        def fake_forward_micro_batch(micro_batch, temperature):
            del temperature, micro_batch
            base = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32, device=actor_module.weight.device)
            log_prob = actor_module.weight * base
            entropy = torch.zeros_like(log_prob)
            return entropy, log_prob

        actor._forward_micro_batch = fake_forward_micro_batch
        return actor

    def _build_data(self):
        tensors = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long),
            "old_log_probs": torch.zeros((1, 4), dtype=torch.float32),
            "advantages": torch.zeros((1, 4), dtype=torch.float32),
            "responses": torch.tensor([[20, 21, 22, 23]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
            "observation_mask": torch.tensor([[0, 0, 1, 0]], dtype=torch.float32),
        }
        return DataProto.from_dict(tensors=tensors, meta_info={"temperature": 1.0})

    def test_update_policy_uses_explicit_observation_mask(self):
        actor = self._build_actor(world_model_coeff=1.0)
        data = self._build_data()

        original_tensor_cuda = torch.Tensor.cuda
        original_tensordict_cuda = TensorDictBase.cuda
        try:
            torch.Tensor.cuda = lambda self, *args, **kwargs: self
            TensorDictBase.cuda = lambda self, *args, **kwargs: self
            metrics = actor.update_policy(data)
        finally:
            torch.Tensor.cuda = original_tensor_cuda
            TensorDictBase.cuda = original_tensordict_cuda

        self.assertAlmostEqual(metrics["actor/wm_sft_loss"], -3.0, places=5)
        self.assertIn("actor/world_model_coeff", metrics)
        self.assertGreater(actor.actor_module.weight.item(), 1.0)

    def test_update_policy_ignores_world_model_loss_when_disabled(self):
        actor = self._build_actor(world_model_coeff=0.0)
        data = self._build_data()

        original_tensor_cuda = torch.Tensor.cuda
        original_tensordict_cuda = TensorDictBase.cuda
        try:
            torch.Tensor.cuda = lambda self, *args, **kwargs: self
            TensorDictBase.cuda = lambda self, *args, **kwargs: self
            metrics = actor.update_policy(data)
        finally:
            torch.Tensor.cuda = original_tensor_cuda
            TensorDictBase.cuda = original_tensordict_cuda

        self.assertNotIn("actor/wm_sft_loss", metrics)
        self.assertAlmostEqual(actor.actor_module.weight.item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
