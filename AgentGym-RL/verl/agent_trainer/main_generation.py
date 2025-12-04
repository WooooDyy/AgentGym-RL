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
Generate responses given a dataset of prompts
"""
from collections import defaultdict
import json
from logging import getLogger, INFO
import ray
import numpy as np
import hydra
import verl.utils.torch_functional as verl_F
import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.agentgym.client import init_env_client


logger = getLogger(__name__)
logger.setLevel(INFO)


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_json(os.path.join(config.data.path, f"{config.agentgym.task_name}_test.json"))
    item_ids = dataset[config.data.prompt_key].tolist()
    # load sub category test file
    category_files = os.listdir(config.data.path)
    category_files = [f for f in category_files if not f.startswith(f"{config.agentgym.task_name}_test")]
    category_map = {}
    for category_file in category_files:
        path = os.path.join(config.data.path, category_file)
        print(f'{path = }')
        with open(path, "r") as f:
            datas = json.load(f)
            for data in datas:
                category_map[data["item_id"]] = category_file.split(".")[0]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool_name = ''
    max_colocate_count = 1
    # resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    # resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, n_gpus_per_node=self.n_gpus_per_node, name_prefix=resource_pool_name, max_colocate_count=max_colocate_count, detached=False)
    resource_pool = RayResourcePool(process_on_nodes=[(n_gpus_per_node := config.trainer.n_gpus_per_node)] * config.trainer.nnodes, use_gpu=True, n_gpus_per_node=n_gpus_per_node, name_prefix=resource_pool_name, max_colocate_count=max_colocate_count, detached=False)
    # class RayResourcePool(ResourcePool):

    # def __init__(self,
    #             #  process_on_nodes: List[int] = None,
    #              process_on_nodes: list[int] | None,
    #             #  use_gpu: bool = True,
    #              use_gpu: bool,
    #              n_gpus_per_node: int,
    #             #  name_prefix: str = "",
    #              name_prefix: str,
    #             max_colocate_count: int = 5,
    # max_colocate_count: int,
    #             #  detached=False) -> None:
    #              detached: bool) -> None:
    # resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping, n_gpus_per_node=config.trainer.max_colocate_countmax_colocate_count)

    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    num_batch = (total_samples // config_batch_size) + 1
    output_lst = [[] for _ in range(config.data.n_samples)]
    env_client = init_env_client(config.agentgym)

    logger.warning(f'{total_samples = }, {config_batch_size = }, {num_batch = }, {dp_size = }')

    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        start_idx = batch_idx * config_batch_size
        end_idx = min(total_samples, start_idx + config_batch_size)
        batch_item_ids = item_ids[start_idx: end_idx]
        prompt_with_chat_template = ["<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + env_client.conversation_start[0]["value"] + "<|im_end|>\n<|im_start|>assistant\n" + env_client.conversation_start[1]["value"] + "<|im_end|>" for _ in range(len(batch_item_ids))]
        if not prompt_with_chat_template:
            raise ValueError(f'No prompts found in the batch. {batch_idx = }, {start_idx = }, {end_idx = }')
        messages = [[{"role": "user", "content": env_client.conversation_start[0]["value"]},
                     {"role": "assistant", "content": env_client.conversation_start[1]["value"]}] for _ in range(len(batch_item_ids))]

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=tokenizer,
                                                                         max_length=config.data.max_prompt_length,
                                                                         pad_token_id=tokenizer.pad_token_id,
                                                                         left_pad=True)
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        data.meta_info['global_steps'] = 'test_batch_' + str(batch_idx)
        data.meta_info['max_rounds'] = config.agentgym.max_rounds
        data.non_tensor_batch["item_id"] = np.array(batch_item_ids, dtype=object)
        data.non_tensor_batch["raw_prompt"] = np.array(messages, dtype=object)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dp_size != 0:
            dummy_data_size = dp_size - real_batch_size % dp_size
            dummy_data = data[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')

        for i in range(config.data.n_samples):
            output = wg.generate_sequences(data)
            # remove dummy data
            output = output[:real_batch_size]

            output_lst[i].extend(output.batch['task_scores'].sum(dim=-1).tolist())

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_np = np.array(output_lst, dtype=object)
    output_np = np.transpose(output_np, axes=(1, 0))
    output_lst = output_np.tolist()

    print("============Total Task Evaluation============")
    print(f"Avg@{config.data.n_samples}: {np.mean(output_np)}")
    print(f"Pass@{config.data.n_samples}: {np.mean(np.max(output_np, axis=-1) > 0)}")
    print("============Sub Task Evaluation============")

    category_success_bucket = defaultdict(list)
    for item_id, score in zip(item_ids, output_lst):
        category = category_map[item_id]
        category_success_bucket[category].append(score)
    for category_file in category_files:
        category = category_file.split(".")[0]
        print(f"Category: {category}")
        print(f"Avg@{config.data.n_samples}: {np.mean(np.array(category_success_bucket[category]))}")
        print(f"Pass@{config.data.n_samples}: {np.mean(np.max(np.array(category_success_bucket[category]), axis=-1) > 0)}")



if __name__ == '__main__':
    main()
