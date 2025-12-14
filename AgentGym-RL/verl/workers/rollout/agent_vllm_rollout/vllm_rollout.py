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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from logging import getLogger
import os
import time
from typing import List, Optional
from omegaconf import DictConfig
from tensordict import TensorDict
import torch
from torch import int32 as torch_int32
import torch.distributed
from torch.distributed import get_rank, get_world_size
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from tqdm import tqdm
from vllm import SamplingParams
from verl import DataProto
from verl.third_party.vllm import LLM, vllm_version, parallel_state as vllm_ps
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length
from verl.utils.agentgym.client import init_env_client
from verl.utils.memory import MemoryBank
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import RolloutHandler, Message, _pre_process_inputs


LOGGER = getLogger(__name__)


# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, rollout_config: DictConfig, agentgym_config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = rollout_config
        self.agentgym_config = agentgym_config
        assert not (not rollout_config.enforce_eager and rollout_config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= (world_size := get_world_size()), f"{tensor_parallel_size = } should be less than or equal to {world_size = }"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=rollout_config.dtype,
            enforce_eager=rollout_config.enforce_eager,
            gpu_memory_utilization=rollout_config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            load_format=rollout_config.load_format,
            disable_log_stats=rollout_config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=rollout_config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=rollout_config.max_tokens,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in rollout_config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = rollout_config.get(k)
        kwargs["n"] = 1  # because we have repeated task n times

        LOGGER.info(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer

        # Initialize memory bank if enabled
        self.memory_enabled = rollout_config.get('memory', {}).get('enabled', False)
        self.memory_bank: Optional[MemoryBank] = None
        if self.memory_enabled:
            memory_config = rollout_config.get('memory', {})
            self.memory_k = memory_config.get('k', 3)
            self.memory_min_reward = memory_config.get('min_reward', 0.5)
            self.memory_encoder = memory_config.get('encoder', 'sentence-transformers/all-MiniLM-L6-v2')
            self.memory_task_specific = memory_config.get('task_specific', True)
            self.memory_save_path = memory_config.get('save_path', None)

            # Initialize memory bank
            self.memory_bank = MemoryBank(
                encoder_name=self.memory_encoder,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                min_reward=self.memory_min_reward,
                task_specific=self.memory_task_specific,
            )

            # Load existing memory bank if path provided
            if self.memory_save_path and os.path.exists(self.memory_save_path + '.pkl'):
                try:
                    self.memory_bank = MemoryBank.load(self.memory_save_path)
                    LOGGER.info(f"Loaded memory bank from {self.memory_save_path} with {len(self.memory_bank)} experiences")
                except Exception as e:
                    LOGGER.warning(f"Failed to load memory bank from {self.memory_save_path}: {e}")

            LOGGER.info(f"Memory bank initialized: k={self.memory_k}, min_reward={self.memory_min_reward}, "
                       f"task_specific={self.memory_task_specific}, encoder={self.memory_encoder}")
        else:
            LOGGER.info("Memory bank disabled")


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def preprocess_prompt_to_rollout_handler(self, prompts: DataProto, n: int) -> List[RolloutHandler]:
        assert "raw_prompt" in prompts.non_tensor_batch.keys(), "raw_prompt is not in non_tensor_batch, need to set data.return_raw_chat=True"
        handler_list = []
        for i, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for _ in range(n):
                # only keep not pad part
                input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch['input_ids'][i])
                attention_mask = _pre_process_inputs(0, prompts.batch['attention_mask'][i])
                position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
                task_name_item_id_string = prompts.non_tensor_batch["item_id"][i].split("_")
                input_ids_list = list(input_ids)
                attention_mask_list = list(attention_mask)
                position_ids_list = list(position_ids)
                handler = RolloutHandler(
                    messages=[
                        Message(role=prompt["role"], content=prompt["content"]) for prompt in raw_prompt
                    ],
                    task_name=task_name_item_id_string[0],
                    item_id=int(task_name_item_id_string[-1]),
                    score=0,
                    done=False,
                    input_ids=input_ids_list,
                    prompt_ids=input_ids_list,
                    response_ids=[],
                    attention_mask=attention_mask_list,
                    prompt_attention_mask=attention_mask_list,
                    response_attention_mask=[],
                    position_ids=position_ids_list,
                    prompt_position_ids=position_ids_list,
                    response_position_ids=[],
                    loss_mask=[0] * len(input_ids),
                    prompt_loss_mask=[0] * len(input_ids),
                    response_loss_mask=[],
                    max_response_len=self.config.response_length,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length)
                )
                assert len(handler.input_ids) == len(handler.attention_mask) == len(handler.position_ids) == len(handler.loss_mask), f"RolloutHandler has mismatched length: input_ids={len(handler.input_ids)}, attention_mask={len(handler.attention_mask)}, position_ids={len(handler.position_ids)}, loss_mask={len(handler.loss_mask)}"
                handler_list.append(handler)
        return handler_list


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        global_steps = prompts.meta_info.get('global_steps', None)
        max_rounds = prompts.meta_info.get('max_rounds', 10)
        cur_device = prompts.batch["input_ids"].device

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # repeat for self.config.n times to rollout
        batch_size = prompts.batch['input_ids'].size(0)
        batch_size *= self.config.n
        rollout_handler_ls = self.preprocess_prompt_to_rollout_handler(prompts, n=self.config.n)
        env_clients = [init_env_client(self.agentgym_config) for _ in range(batch_size)]
        time.sleep(self.config.send_interval) # take a break before sendng request
        all_done_flag = False
        for idx, rollout_handler in enumerate(rollout_handler_ls):
            try:
                env_clients[idx].reset(rollout_handler.item_id)
                task = env_clients[idx].observe()
                rollout_handler.add_user_message(self.tokenizer, task)
            except TimeoutError:
                LOGGER.info(f"Reset Timeout: Webarena Env Timeout. item id = {rollout_handler.item_id}")
                rollout_handler.done = True
                rollout_handler.score = 0

        rounds = 0
        task_rounds = [0] * batch_size
        rollout_bar = tqdm(total = max_rounds, desc="Running rounds", disable=torch.distributed.get_rank() != 0)

        # Track experiences for memory storage
        step_experiences: List[List[tuple]] = [[] for _ in range(batch_size)]  # Store (obs_text, action) for each agent

        def agent_step(i, idx):
            content = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)

            # Store observation before action for memory
            if self.memory_enabled and len(rollout_handler_ls[idx].messages) > 0:
                last_msg = rollout_handler_ls[idx].messages[-1]
                if last_msg.role == 'user':
                    obs_text = last_msg.content
                    step_experiences[idx].append((obs_text, content))

            rollout_handler_ls[idx].add_assistant_message(self.tokenizer, content)
            task_rounds[idx] += 1
            try:
                step_output = env_clients[idx].step(content)
                state, rollout_handler_ls[idx].score, rollout_handler_ls[idx].done = (
                    step_output.state,
                    step_output.reward,
                    step_output.done,
                )
                rollout_handler_ls[idx].add_user_message(self.tokenizer, state)

                # Store successful experiences in memory bank when episode is done
                if self.memory_enabled and rollout_handler_ls[idx].done:
                    for obs_text, action in step_experiences[idx]:
                        self.memory_bank.add(
                            obs_text=obs_text,
                            action=action,
                            reward=rollout_handler_ls[idx].score,
                            task_name=rollout_handler_ls[idx].task_name,
                            item_id=rollout_handler_ls[idx].item_id,
                        )
                    step_experiences[idx] = []  # Clear after storing

                return step_output.done
            except Exception as e:
                rollout_handler_ls[idx].score = 0
                rollout_handler_ls[idx].done = True
                LOGGER.info(f"Rollou step Error: {e} item id = {rollout_handler_ls[idx].item_id}")
                step_experiences[idx] = []  # Clear on error
                return True
        while rounds < max_rounds and not all_done_flag:
            # get generation prompt with memory retrieval
            generation_prompt_idxs = []
            not_done_idxs = []
            for idx, rollout_handler in enumerate(rollout_handler_ls):
                if not rollout_handler.done:
                    # Retrieve similar experiences from memory if enabled
                    memory_examples = None
                    if self.memory_enabled and self.memory_bank is not None:
                        # Get current observation (last user message)
                        if len(rollout_handler.messages) > 0 and rollout_handler.messages[-1].role == 'user':
                            current_obs = rollout_handler.messages[-1].content
                            try:
                                retrieved_exps = self.memory_bank.retrieve(
                                    query_text=current_obs,
                                    k=self.memory_k,
                                    task_name=rollout_handler.task_name if self.memory_task_specific else None,
                                )
                                if retrieved_exps:
                                    memory_examples = self.memory_bank.format_as_examples(
                                        retrieved_exps,
                                        self.tokenizer,
                                        format_style='chat',
                                    )
                            except Exception as e:
                                LOGGER.warning(f"Memory retrieval failed: {e}")

                    # Generate prompt with memory examples
                    generation_prompt_idxs.append(
                        rollout_handler.get_generation_prompt(self.tokenizer, memory_examples=memory_examples)
                    )
                    not_done_idxs.append(idx)

            rollout_bar.set_description(f"Rounds {rounds + 1}/{max_rounds} | Active agents per gpu: {len(not_done_idxs)}")
            # users can customize different sampling_params at different run
            with self.update_sampling_params(**kwargs):
                output = self.inference_engine.generate(
                    prompts=None,
                    prompt_token_ids=generation_prompt_idxs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False)
            response_ids = output[0].tolist()
            all_done_flag = True
            time.sleep(self.config.send_interval) # take a break before sendng request
            if len(not_done_idxs) > 0:
                with ThreadPoolExecutor(max_workers=len(not_done_idxs)) as executor:
                    step_dones = list(executor.map(
                        lambda args: agent_step(*args), [(i, idx) for i, idx in enumerate(not_done_idxs)]
                    ))
                    all_done_flag = all(step_dones)
            rounds += 1
            rollout_bar.update(1)

        # process ids
        rollout_bar.close()
        LOGGER.info(f'vLLMRollout.generate_sequences: Finished {rounds} rounds of rollout.')
        # breakpoint()
        response_ids, response_attention_mask, response_position_ids, response_loss_mask = [], [], [], []
        scores, messages = [], []

        for rollout_handler in rollout_handler_ls:
            # check length
            rollout_handler.truncate_output_ids()
            assert len(rollout_handler.input_ids) == len(rollout_handler.attention_mask) == len(rollout_handler.position_ids) == len(rollout_handler.loss_mask), f"""Rollout Handler has different length of {len(rollout_handler.input_ids)=},
            {len(rollout_handler.attention_mask)=}, {len(rollout_handler.position_ids)=}, {len(rollout_handler.loss_mask)=}"""
            assert len(rollout_handler.input_ids) <= self.config.max_model_len, f"Rollout Handler has sequence length {len(rollout_handler.input_ids)} > max_sequence_length {self.config.max_model_len}"

            response_ids.append(torch.tensor(rollout_handler.response_ids, dtype=torch_int32, device=cur_device))
            response_attention_mask.append(torch.tensor(rollout_handler.response_attention_mask, dtype=torch_int32, device=cur_device))
            response_position_ids.append(torch.tensor(rollout_handler.response_position_ids, dtype=torch_int32, device=cur_device))
            response_loss_mask.append(torch.tensor(rollout_handler.response_loss_mask, dtype=torch_int32, device=cur_device))
            scores.append(rollout_handler.score)
            messages.append(rollout_handler.messages)
        LOGGER.info(f'vLLMRollout.generate_sequences: Completed processing {len(rollout_handler_ls)} rollout handlers.')
        # breakpoint()

        # pad to length
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        response_length = response_ids.size(1)
        delta_position_ids = torch.arange(1, response_length + 1, device=cur_device)
        delta_position_ids = delta_position_ids.unsqueeze(0).repeat(batch_size, 1)
        input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        prompt_length = input_ids.size(-1)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        input_ids = input_ids.repeat_interleave(self.config.n, dim=0)
        attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
        position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
        response_position_ids = position_ids[:, -1:] + delta_position_ids

        seq = torch.cat((input_ids, response_ids), dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((position_ids, response_position_ids), dim=-1)
        response_mask = response_loss_mask

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32) # (bs, response_length)
        valid_response_length = attention_mask[:, prompt_length:].sum(dim=-1)
        for i in range(len(scores)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
        LOGGER.info(f'vLLMRollout.generate_sequences: Prepared final tensors for output.')
        # breakpoint() # here.

        if global_steps:
            try:
                os.makedirs(os.path.join(self.config.rollout_log_dir, f"step{global_steps}"), exist_ok=True)
                with open(os.path.join(self.config.rollout_log_dir, f"step{global_steps}/{torch.distributed.get_rank()}.json"), "w") as f:
                    json_msg = []
                    for idx, msgs in enumerate(messages):
                        records = {
                            "item_id": rollout_handler_ls[idx].item_id,
                            "conversations": [msg.to_dict() for msg in msgs],
                            "reward": scores[idx]
                        }
                        json_msg.append(records)
                    json.dump(json_msg, f, ensure_ascii=True, indent=4)
            except Exception as e:
                LOGGER.info(e)
        LOGGER.info(f'vLLMRollout.generate_sequences: Saved rollout logs at step {global_steps}.')
        # breakpoint()

        # close clients
        for client in env_clients:
            try:
                client.close()
            except Exception as e:
                LOGGER.info(f"Error during closing env: {e}")
                breakpoint()

        LOGGER.info(f'vLLMRollout.generate_sequences: Closed all environment clients.')
        # breakpoint()

        batch = TensorDict(
            {
                'prompts': input_ids,
                'responses': response_ids,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'response_mask': response_mask,
                'scores': reward_tensor,
                'task_rounds': torch.tensor(task_rounds, dtype=torch.float32, device=input_ids.device),
                'task_scores': reward_tensor
            },
            batch_size=batch_size)
        LOGGER.info(f'vLLMRollout.generate_sequences: Constructed final output batch tensor dict.')
        # breakpoint()

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
            LOGGER.info(f'vLLMRollout.generate_sequences: Freed vLLM cache engine if applicable.')

        # Save memory bank if enabled
        if self.memory_enabled and self.memory_bank is not None and self.memory_save_path:
            try:
                self.memory_bank.save(self.memory_save_path)
                LOGGER.info(f"Saved memory bank to {self.memory_save_path} with {len(self.memory_bank)} experiences")
            except Exception as e:
                LOGGER.warning(f"Failed to save memory bank: {e}")

        LOGGER.info(f'vLLMRollout.generate_sequences: Completed rollout for all agents.')
        # breakpoint()

        return DataProto(batch=batch)
