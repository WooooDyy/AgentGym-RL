# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Megatron PPO actor.
"""

from functools import partial
from typing import Iterable

import torch
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.utils.megatron.pipeline_parallel import compute_transformers_input_shapes, make_batch_generator
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import broadcast_dict_tensor, split_dict_tensor_into_batches
from verl.workers.agent_actor import BasePPOActor
import verl.utils.torch_functional as verl_F

__all__ = ["MegatronPPOActor"]


class MegatronPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        model_config,
        megatron_config,
        actor_module,
        actor_optimizer=None,
        actor_optimizer_config=None,
    ):
        super().__init__(config=config)
        self.model_config = model_config
        self.megatron_config = megatron_config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.actor_optimizer_config = actor_optimizer_config

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  epochs=self.config.ppo_epochs,
                                  dataloader_kwargs={"shuffle": self.config.shuffle, "drop_last": True})

    def _set_mode(self, train: bool) -> None:
        for module in self.actor_module:
            if train:
                module.train()
            else:
                module.eval()

    def _broadcast_batch(self, batch):
        batch = batch.contiguous()
        broadcast_dict_tensor(batch,
                              src=mpu.get_pipeline_model_parallel_last_rank(),
                              group=mpu.get_pipeline_model_parallel_group())
        return batch

    def _run_forward_batches(self, batch, micro_batch_size: int, forward_step, forward_only: bool):
        batch = self._broadcast_batch(batch)
        batch["attention_mask"] = batch["attention_mask"].to(bool)
        batches = split_dict_tensor_into_batches(batch, batch_size=micro_batch_size)
        n_micro_batch = len(batches)
        seq_len = batches[0]["input_ids"].shape[1]
        input_shapes = compute_transformers_input_shapes(
            batches,
            meta_info={
                "sequence_parallel": self.megatron_config.sequence_parallel,
                "hidden_size": self.model_config.hidden_size,
            },
        )
        batch_generator = make_batch_generator(batches, vpp_size=len(self.actor_module))
        forward_backward_func = get_forward_backward_func()

        common_kwargs = {
            "forward_step_func": forward_step,
            "data_iterator": batch_generator,
            "model": self.actor_module,
            "num_microbatches": n_micro_batch,
            "seq_length": micro_batch_size * seq_len,
            "hidden_size": self.model_config.hidden_size,
            "micro_batch_size": 1,
            "forward_only": forward_only,
        }
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            common_kwargs["input_shapes"] = input_shapes

        return forward_backward_func(**common_kwargs)

    def compute_log_prob(self, data: DataProto):
        self._set_mode(train=False)

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        return_entropy = data.meta_info.get("return_entropy", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        batch_size = batch.batch_size[0]
        response_length = batch["responses"].shape[1]

        def loss_func(micro_batch, output):
            logits = output.logits[:, -response_length - 1:-1, :]
            logits.div_(temperature)
            log_probs = vocab_parallel_log_probs_from_logits(logits=logits, labels=micro_batch["responses"])
            metrics = {"log_probs": log_probs}
            if return_entropy:
                metrics["entropys"] = vocab_parallel_entropy(logits)
            # Forward-only path ignores the scalar loss.
            return torch.zeros((), device=logits.device, dtype=logits.dtype), metrics

        def forward_step(batch_iter, model):
            micro_batch = next(batch_iter)
            output = model(input_ids=micro_batch["input_ids"],
                           attention_mask=micro_batch["attention_mask"],
                           position_ids=micro_batch["position_ids"])
            return output, partial(loss_func, micro_batch)

        losses_reduced = self._run_forward_batches(batch=batch,
                                                   micro_batch_size=micro_batch_size,
                                                   forward_step=forward_step,
                                                   forward_only=True)

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            log_probs = torch.cat([item["log_probs"] for item in losses_reduced], dim=0)
            entropys = torch.cat([item["entropys"] for item in losses_reduced], dim=0) if return_entropy else None
        else:
            log_probs = torch.empty((batch_size, response_length),
                                    dtype=torch.float32,
                                    device=batch["input_ids"].device)
            entropys = torch.empty_like(log_probs) if return_entropy else None

        torch.distributed.broadcast(log_probs,
                                    src=mpu.get_pipeline_model_parallel_last_rank(),
                                    group=mpu.get_pipeline_model_parallel_group(),
                                    async_op=False)
        if return_entropy:
            torch.distributed.broadcast(entropys,
                                        src=mpu.get_pipeline_model_parallel_last_rank(),
                                        group=mpu.get_pipeline_model_parallel_group(),
                                        async_op=False)
            return log_probs, entropys
        return log_probs

    def _optimizer_step(self):
        step_output = self.actor_optimizer.step()
        if isinstance(step_output, tuple):
            if len(step_output) >= 2:
                grad_norm = step_output[1]
            elif len(step_output) == 1:
                grad_norm = step_output[0]
            else:
                grad_norm = None
        else:
            grad_norm = step_output
        return grad_norm

    def update_policy(self, dataloader):
        self._set_mode(train=True)

        metrics = {}
        world_model_coeff = self.config.get("world_model_coeff", 0.0)
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        grad_accum_steps = self.config.ppo_mini_batch_size // micro_batch_size

        for mini_batch in dataloader:
            batch = mini_batch.batch
            temperature = mini_batch.meta_info.get("temperature", 1.0)
            self.actor_optimizer.zero_grad()

            response_length = batch["responses"].shape[1]

            def loss_func(micro_batch, output):
                logits = output.logits[:, -response_length - 1:-1, :]
                logits.div_(temperature)
                log_prob = vocab_parallel_log_probs_from_logits(logits=logits, labels=micro_batch["responses"])
                entropy = vocab_parallel_entropy(logits)

                response_mask = micro_batch["response_mask"]
                old_log_prob = micro_batch["old_log_probs"]
                advantages = micro_batch["advantages"]

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                             log_prob=log_prob,
                                                                             advantages=advantages,
                                                                             eos_mask=response_mask,
                                                                             cliprange=self.config.clip_ratio)
                entropy_loss = verl_F.masked_mean(entropy, response_mask)
                policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff

                metric_dict = {
                    "actor/entropy_loss": entropy_loss.detach().item(),
                    "actor/pg_loss": pg_loss.detach().item(),
                    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    "actor/ppo_kl": ppo_kl.detach().item(),
                }

                if self.config.use_kl_loss:
                    ref_log_prob = micro_batch["ref_log_prob"]
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = verl_F.masked_mean(kld, response_mask)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metric_dict["actor/kl_loss"] = kl_loss.detach().item()
                    metric_dict["actor/kl_coef"] = self.config.kl_loss_coef

                if world_model_coeff > 0:
                    response_length_local = response_mask.shape[1]
                    observation_mask = micro_batch["attention_mask"][:, -response_length_local:].float() * (
                        1.0 - response_mask.float()
                    )
                    if observation_mask.any().item():
                        wm_sft_loss = -verl_F.masked_mean(log_prob, observation_mask)
                        policy_loss = policy_loss + world_model_coeff * wm_sft_loss
                        metric_dict["actor/wm_sft_loss"] = wm_sft_loss.detach().item()
                        metric_dict["actor/world_model_coeff"] = world_model_coeff

                return policy_loss / grad_accum_steps, metric_dict

            def forward_step(batch_iter, model):
                micro_batch = next(batch_iter)
                output = model(input_ids=micro_batch["input_ids"],
                               attention_mask=micro_batch["attention_mask"],
                               position_ids=micro_batch["position_ids"])
                return output, partial(loss_func, micro_batch)

            losses_reduced = self._run_forward_batches(batch=batch,
                                                       micro_batch_size=micro_batch_size,
                                                       forward_step=forward_step,
                                                       forward_only=False)

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                for reduced in losses_reduced:
                    append_to_dict(metrics, reduced)

            grad_norm = self._optimizer_step()
            if grad_norm is not None and mpu.is_pipeline_last_stage(ignore_virtual=True):
                append_to_dict(metrics, {"actor/grad_norm": float(grad_norm)})

            self.actor_optimizer.zero_grad()

        return metrics
