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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from pprint import pprint
from omegaconf import OmegaConf
import hydra
import ray
from verl.agent_trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker, CriticWorker
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config):
    # print initial config
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    strategy_actor = config.actor_rollout_ref.actor.strategy
    if strategy_actor != 'fsdp':
        raise NotImplementedError('Only FSDP is supported for PPO trainer now.')
    assert strategy_actor == config.critic.strategy
    # ray_worker_group_cls = RayWorkerGroup

    # if config.actor_rollout_ref.actor.strategy == 'fsdp':
    #     assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
    #     from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker, CriticWorker
    #     from verl.single_controller.ray import RayWorkerGroup
    #     ray_worker_group_cls = RayWorkerGroup

    # else:
    #     raise NotImplementedError


    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping, n_gpus_per_node=config.trainer.n_gpus_per_node)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=RayWorkerGroup)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    # import socket
    # print(socket.gethostbyname(socket.gethostname()))
    # socket.sethostname("localhost")
    # print(socket.gethostbyname(socket.gethostname()))
    main()
