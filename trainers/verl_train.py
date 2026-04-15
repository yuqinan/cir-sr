"""Train an LLM using GRPO over Reasoning Gym procedural dataset(s)."""

print("DEBUG: Starting imports...")

print("DEBUG: Importing basic modules...")
from dataclasses import replace

print("DEBUG: Importing hydra...")
import hydra

print("DEBUG: Importing ray...")
import ray
from ray.util import get_node_ip_address

print("DEBUG: Importing omegaconf...")
from omegaconf import OmegaConf

print("DEBUG: Importing custom trainers...")
print("DEBUG: About to import RayGRPOTrainer...")
from trainers.verl_grpo_trainer import RayGRPOTrainer
from trainers.verl_dapo_trainer import RayDAPOTrainer
print("DEBUG: RayGRPOTrainer imported successfully!")
print("DEBUG: About to import VLLMRewardModelWorker...")
from trainers.vllm_reward import VLLMRewardModelWorker
print("DEBUG: VLLMRewardModelWorker imported successfully!")

print("DEBUG: Importing utils...")
from utils.datasets import ReasoningGymDataset, make_dataset, GSM8KDataset, make_gsm8k_dataset, MATHHardDataset, make_math_hard_dataset

print("DEBUG: Importing data and reasoning_gym...")
import data
import reasoning_gym

print("DEBUG: Importing torch and pandas...")
from torch.utils.data import ConcatDataset
import pandas as pd

print("DEBUG: All top-level imports completed successfully!")

print("This node's IP address is:", get_node_ip_address())

def prepare_datasets(config, tokenizer) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    """Prepare training and validation datasets."""
    developer_prompt_setting = config.reasoning_gym.developer_prompt
    developer_prompt = data.template.SYSTEM_PROMPTS[developer_prompt_setting]

    # Create and concatenate data sources first
    train_data_sources = []
    val_data_sources = []
    # Collect all training data sources
    print(f"DEBUG: collecting {len(config.reasoning_gym.datasets)} training data sources")
    for dataset_config in config.reasoning_gym.datasets:
        print(dataset_config)
        train_data_source = reasoning_gym.create_dataset(**dataset_config)
        print(f"DEBUG: collected {len(train_data_source)} training data sources")
        train_data_sources=train_data_source
        print(f"DEBUG: collected {len(train_data_source)} training data sources")

        
    # Collect all validation data sources  
    for dataset_config in config.reasoning_gym.validation_dataset:
        val_data_source = reasoning_gym.create_dataset(**dataset_config)
        val_data_sources.extend(val_data_source)

    print(f"DEBUG: collected {len(train_data_sources)} training data sources and {len(val_data_sources)} validation data sources")
    # Create single datasets from combined data sources
    train_dataset = make_dataset(
        tokenizer, train_data_sources, developer_prompt, 
        max_prompt_length=config.data.max_prompt_length,
        chat=config.data.chat,
        preappend_token=config.data.preappend_token
    )
    val_dataset = make_dataset(
        tokenizer, val_data_sources, developer_prompt, 
        max_prompt_length=config.data.max_prompt_length,
        chat=config.data.chat,
        preappend_token=config.data.preappend_token
    )
    print(f"DEBUG: created {len(train_dataset)} training dataset and {len(val_dataset)} validation dataset")
    
    return train_dataset, val_dataset

def filter_gsm8k_datasets(file_path, size, seed):
    data_source = pd.read_parquet(file_path)
    if size > 0:
        data_source = data_source.sample(size, random_state=seed)
    return data_source

def filter_math_hard_datasets(file_path, size, seed):
    data_source = pd.read_parquet(file_path)
    if size > 0:
        data_source = data_source.sample(size, random_state=seed)
    return data_source

def prepare_math_hard_datasets(config, tokenizer) -> tuple[MATHHardDataset, MATHHardDataset]:
    developer_prompt_setting = config.math_hard.developer_prompt
    developer_prompt = data.template.SYSTEM_PROMPTS[developer_prompt_setting]
    train_data_sources = filter_math_hard_datasets(**config.math_hard.datasets)
    val_data_sources = filter_math_hard_datasets(**config.math_hard.validation_dataset)

    train_dataset = make_math_hard_dataset(
        tokenizer, train_data_sources, developer_prompt,
        max_prompt_length=config.data.max_prompt_length,
        chat=config.data.chat
    )
    val_dataset = make_math_hard_dataset(
        tokenizer, val_data_sources, developer_prompt,
        max_prompt_length=config.data.max_prompt_length,
        chat=config.data.chat
    )
    return train_dataset, val_dataset

def prepare_gsm8k_datasets(config, tokenizer) -> tuple[GSM8KDataset, GSM8KDataset]:
    developer_prompt_setting = config.gsm8k.developer_prompt

    developer_prompt = data.template.SYSTEM_PROMPTS[developer_prompt_setting]
    train_data_sources = filter_gsm8k_datasets(**config.gsm8k.datasets)
    val_data_sources = filter_gsm8k_datasets(**config.gsm8k.validation_dataset)

    train_dataset = make_gsm8k_dataset(
        tokenizer, train_data_sources, developer_prompt, 
        max_prompt_length=config.data.max_prompt_length,    
        chat=config.data.chat
    )
    val_dataset = make_gsm8k_dataset(
        tokenizer, val_data_sources, developer_prompt, 
        max_prompt_length=config.data.max_prompt_length,
        chat=config.data.chat   
    )

    return train_dataset, val_dataset

@ray.remote
def main_task(config):
    from pprint import pprint

    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == "fsdp" or config.actor_rollout_ref.actor.strategy == "fsdp2":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError




    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    if config.reward_model.enable:
        role_worker_mapping[Role.RewardModel] = ray.remote(VLLMRewardModelWorker)

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
        Role.RewardModel: global_pool_id,
    }
    print("-------------------------------- initializing resource pool manager --------------------------------")
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    print(config.data.task)
    print("-------------------------------- loading datasets --------------------------------")
    if config.data.task == "reasoning_gym":
        train_dataset, val_dataset = prepare_datasets(config, tokenizer)
    elif config.data.task == "gsm8k":
        train_dataset, val_dataset = prepare_gsm8k_datasets(config, tokenizer)
    elif config.data.task == "math_hard":
        train_dataset, val_dataset = prepare_math_hard_datasets(config, tokenizer)
    else:
        raise ValueError(f"Invalid task: {config.data.task}")
    print("-------------------------------- datasets loaded --------------------------------")

    trainer = RayGRPOTrainer(
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        max_output_length=config.data.max_response_length,
    )
    
    print("-------------------------------- initializing workers --------------------------------")
    trainer.init_workers()
    print("-------------------------------- initializing workers done --------------------------------")
    trainer.fit()

from omegaconf import DictConfig, ListConfig

@hydra.main(config_path="configs/grpo", config_name="mini_sudoku-sudoku", version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf, DictConfig, ListConfig

    resolved = OmegaConf.to_container(config, resolve=True)  # resolve=True will eval symbol values
    
    OmegaConf.resolve(config)

    pprint(OmegaConf.to_container(config, resolve=True))

    if not ray.is_initialized():

        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}, num_cpus=config.ray_init.num_cpus)
        #ray.init()
        print("This node's IP address is:", get_node_ip_address())
    config.data.preappend_token = "<" + config.data.preappend_token + ">"
    ray.get(main_task.remote(config))


if __name__ == "__main__":
    print("starting training")
    main()
    if ray.is_initialized():
        ray.shutdown()