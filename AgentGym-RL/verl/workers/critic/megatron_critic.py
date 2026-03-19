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
Megatron PPO critic.
"""

from functools import partial

import torch
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.utils.megatron.pipeline_parallel import compute_transformers_input_shapes, make_batch_generator
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import broadcast_dict_tensor, masked_mean, split_dict_tensor_into_batches
from verl.workers.agent_critic import BasePPOCritic

__all__ = ["MegatronPPOCritic"]


class MegatronPPOCritic(BasePPOCritic):

    def __init__(
        self,
        config,
        model_config,
        megatron_config,
        critic_module,
        critic_optimizer,
        critic_optimizer_config=None,
    ):
        super().__init__(config=config)
        self.model_config = model_config
        self.megatron_config = megatron_config
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.critic_optimizer_config = critic_optimizer_config

    def make_minibatch_iterator(self, data: DataProto):
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  epochs=self.config.ppo_epochs,
                                  dataloader_kwargs={"shuffle": self.config.shuffle, "drop_last": True})

    def _set_mode(self, train: bool) -> None:
        for module in self.critic_module:
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
        batch_generator = make_batch_generator(batches, vpp_size=len(self.critic_module))
        forward_backward_func = get_forward_backward_func()

        common_kwargs = {
            "forward_step_func": forward_step,
            "data_iterator": batch_generator,
            "model": self.critic_module,
            "num_microbatches": n_micro_batch,
            "seq_length": micro_batch_size * seq_len,
            "hidden_size": self.model_config.hidden_size,
            "micro_batch_size": 1,
            "forward_only": forward_only,
        }
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            common_kwargs["input_shapes"] = input_shapes

        return forward_backward_func(**common_kwargs)

    def compute_values(self, data: DataProto) -> torch.Tensor:
        self._set_mode(train=False)

        micro_batch_size = data.meta_info.get("micro_batch_size", data.batch.batch_size[0])
        select_keys = ["input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        batch_size = batch.batch_size[0]
        seq_len = batch["input_ids"].shape[1]

        def loss_func(_, output):
            return torch.zeros((), device=output.logits.device, dtype=output.logits.dtype), {"values": output.logits}

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
            values = torch.cat([item["values"] for item in losses_reduced], dim=0)
        else:
            values = torch.empty((batch_size, seq_len), dtype=torch.float32, device=batch["input_ids"].device)

        torch.distributed.broadcast(values,
                                    src=mpu.get_pipeline_model_parallel_last_rank(),
                                    group=mpu.get_pipeline_model_parallel_group(),
                                    async_op=False)
        values = values * batch["attention_mask"]
        return values

    def _optimizer_step(self):
        step_output = self.critic_optimizer.step()
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

    def update_critic(self, dataloader):
        self._set_mode(train=True)

        metrics = {}
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        grad_accum_steps = self.config.ppo_mini_batch_size // micro_batch_size

        for mini_batch in dataloader:
            batch = mini_batch.batch
            self.critic_optimizer.zero_grad()

            def loss_func(micro_batch, output):
                vpreds = output.logits
                values = micro_batch["values"]
                returns = micro_batch["returns"]
                eos_mask = micro_batch["response_mask"]

                vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                     values=values,
                                                                     returns=returns,
                                                                     eos_mask=eos_mask,
                                                                     cliprange_value=self.config.cliprange_value)
                metric_dict = {
                    "critic/vf_loss": vf_loss.detach().item(),
                    "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                    "critic/vpred_mean": masked_mean(vpreds, eos_mask).detach().item(),
                }
                return vf_loss / grad_accum_steps, metric_dict

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
                append_to_dict(metrics, {"critic/grad_norm": float(grad_norm)})

            self.critic_optimizer.zero_grad()

        return metrics
