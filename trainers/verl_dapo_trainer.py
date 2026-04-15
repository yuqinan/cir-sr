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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(
        self,
        config,
        tokenizer,
        train_dataset,
        val_dataset,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        max_output_length: int = 1024,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.max_output_length = max_output_length
        task_types = []
        if config.data.task == "reasoning_gym":
            self.val_path = config.reasoning_gym.val_path
            self.dataset_configs = config.reasoning_gym.datasets
            self.val_dataset_configs = config.reasoning_gym.validation_dataset
            
            task_types = [dataset_config.name for dataset_config in self.dataset_configs]
            task_types.extend([dataset_config.name for dataset_config in self.val_dataset_configs])
            task_types = list(set(task_types))
            
            self.reward_calculator = RewardCalculator(
                task=config.data.task,
                task_type=task_types,
            )
            
            # Pass task_types to reward_model config for vllm_reward.py
            with open_dict(config):
                config.reward_model.task_types = task_types
            
            
        else:
            self.val_path = config.gsm8k.val_path
            self.dataset_configs = config.gsm8k.datasets
            self.val_dataset_configs = config.gsm8k.validation_dataset
            self.reward_calculator = RewardCalculator(
                task=config.data.task,
                task_type=task_types,
            )
        
        self.all_dataset_configs = self.dataset_configs
        self.all_val_dataset_configs = self.val_dataset_configs
        self.config = config
        
        # Parse reward configuration from config
        self.reward_config = getattr(config.reasoning_gym, 'rewards', {'rule_based': 1.0}) if hasattr(config, 'reasoning_gym') else {'rule_based': 1.0}
        self.preappend_token = config.data.preappend_token
        

        train_reward_fn = lambda data: self._score_output(data, num_examine=10, preappend_token=self.preappend_token)
        val_reward_fn = lambda data, **kwargs: self._validate_score(data, **kwargs, preappend_token=self.preappend_token)

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            reward_fn = train_reward_fn,
            val_reward_fn = val_reward_fn,
            collate_fn = collate_fn
        )
    
    def _add_decoded_text_to_batch(self, data: DataProto) -> DataProto:
        """
        Pre-decode prompts and responses using the teacher tokenizer and add to DataProto.
        This ensures the student model gets properly decoded text.
        """
        # print("[DEBUG] Pre-decoding text using teacher tokenizer")
        
        # Store decoded text in non_tensor_batch to pass to reward model
        decoded_prompts = []
        decoded_responses = []
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            decoded_prompts.append(prompt_str)
            decoded_responses.append(response_str)
        
        # Add decoded text to batch
        data.non_tensor_batch["decoded_prompts"] = np.array(decoded_prompts, dtype=object)
        data.non_tensor_batch["decoded_responses"] = np.array(decoded_responses, dtype=object)
      
        return data

    def _compute_rewards_common(self, data: DataProto, is_validation: bool = False, num_examine: int = 10, preappend_token: str = ""):
        """
        Common reward computation logic for both training and validation using weighted rewards from config.
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        data = self._add_decoded_text_to_batch(data)
        decoded_prompts = data.non_tensor_batch["decoded_prompts"]
        decoded_responses = data.non_tensor_batch["decoded_responses"]
        
        responses_by_index = {} if is_validation else None
        
        # Collect all predictions and entries for batch processing
        predictions = []
        entries = []
        
        if "usefulness" in self.reward_config:
            predictions_perturbed = []
            print("-----------before---------------------")
            print(self.tokenizer.decode(data[0].batch['input_ids'].detach().cpu().tolist(), skip_special_tokens=True))

            data_batch_input_ids_ = deepcopy(data.batch["input_ids"])
            data_batch_attention_mask_ = deepcopy(data.batch["attention_mask"])
            data_batch_position_ids_ = deepcopy(data.batch["position_ids"])
            
            batch_perturbed = make_usefulness_perturbed_dataset(
                    original_output = data, 
                    tokenizer = self.tokenizer, 
                    max_prompt_length=self.config.data.max_prompt_length + self.config.data.max_response_length, 
                    truncation=self.config.data.truncation,
                    chosen_mode="shuffle")

            batch_perturbed_output = self.actor_rollout_wg.generate_sequences(batch_perturbed)
            batch_perturbed_output = self._add_decoded_text_to_batch(batch_perturbed_output)
            decoded_responses_perturbed = batch_perturbed_output.non_tensor_batch["decoded_responses"]

            for i in range(len(batch_perturbed_output)):
                data_item = batch_perturbed_output[i]
                response_str = decoded_responses_perturbed[i]
                predictions_perturbed.append(response_str)
        
            print(data.batch.keys())
            data.batch["input_ids"] = data_batch_input_ids_
            data.batch["attention_mask"] = data_batch_attention_mask_
            data.batch["position_ids"] = data_batch_position_ids_
            print("------------after--------------------")
            print(self.tokenizer.decode(batch_perturbed_output[0].batch['input_ids'].detach().cpu().tolist(), skip_special_tokens=True))
            print("-------------------------------------")
            print(self.tokenizer.decode(data[0].batch['input_ids'].detach().cpu().tolist(), skip_special_tokens=True))
        
        for i in range(len(data)):
            data_item = data[i]
            response_str = decoded_responses[i]
            index = data_item.non_tensor_batch["index"]
            
            # Get entry 
            if is_validation:
                entry = self._get_val_entry_for_index(index)
            else:
                entry = self._get_entry_for_index(index)
            
            predictions.append(response_str)
            entries.append(entry)      
        
        # Compute weighted rewards based on config
        reward_components = {}
        
        # Calculate different reward types based on config
        for reward_name, weight in self.reward_config.items():
            if weight <= 0:
                continue
                
            if reward_name == "informativeness":
                # Use batch processing for informativeness rewards
                reward_components[reward_name] = self.reward_calculator.calculate_batch_informativeness_rewards(
                    predictions, entries, preappend_token=preappend_token
                )
                

            elif reward_name == "rule_based":  # rule_based or other types
                # Use individual processing for other reward types
                individual_rewards = []
                for prediction, entry in zip(predictions, entries):
                    reward = self.reward_calculator.calculate_reward(
                        prediction, entry, preappend_token=preappend_token, reward_type=reward_name
                    )
                    individual_rewards.append(reward)
                reward_components[reward_name] = individual_rewards
            
            else:
                individual_rewards_perturbed = []
                for prediction_perturbed, prediction in zip(predictions_perturbed, predictions):
                    #reward = self.reward_calculator.calculate_reward(
                    #    prediction, entry, preappend_token=preappend_token, reward_type=reward_name
                    #)
                    reward = self.reward_calculator.calculate_usefulness_reward(
                        prediction_perturbed, prediction
                    )
                    individual_rewards_perturbed.append(reward)
                reward_components[reward_name] = individual_rewards_perturbed
        
        # Combine weighted rewards
        final_rewards = [0.0] * len(predictions)
        for i in range(len(predictions)):
            weighted_sum = 0.0
            for reward_name, rewards_list in reward_components.items():
                weight = self.reward_config[reward_name]
                weighted_sum += weight * rewards_list[i]
            final_rewards[i] = weighted_sum
        
        # Set rewards in tensor and collect for validation logging
        for i, final_reward in enumerate(final_rewards):
            data_item = data[i]
            index = data_item.non_tensor_batch["index"]
            
            # Set reward at the last token position
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            reward_tensor[i, valid_response_length - 1] = final_reward
            
            # Collect responses for validation JSON logging
            if is_validation:
                entry = entries[i]
                if index not in responses_by_index:
                    responses_by_index[index] = {
                        'entry': entry,
                        'source_data': entry['data_source'],
                        'responses': []
                    }
                
                # Add detailed reward breakdown for logging
                reward_breakdown = {}
                for reward_name, rewards_list in reward_components.items():
                    reward_breakdown[f'{reward_name}_reward'] = rewards_list[i]
                    reward_breakdown[f'{reward_name}_weight'] = self.reward_config[reward_name]
                
                responses_by_index[index]['responses'].append({
                    'response': decoded_responses[i],
                    'final_reward': final_reward,
                    'reward_breakdown': reward_breakdown
                })
            
            if i < 1:
                print(f"-------------------------------- [DEBUG] {i} --------------------------------")
                print(f"First response: {decoded_responses[i]}")
                print(f"Reward config: {self.reward_config}")
                print(f"Final reward: {final_reward}")
                if reward_components:
                    for reward_name, rewards_list in reward_components.items():
                        print(f"{reward_name}: {rewards_list[i]} (weight: {self.reward_config[reward_name]})")
            if reward_components['rule_based'] == 1.0:
                print(f"-------------------------------- correct response--------------------------------")
                print(f"First response: {decoded_responses[i]}")
        
        if is_validation:
            return reward_tensor, responses_by_index, reward_components
        
        return reward_tensor, responses_by_index
    

    def _score_output(self, data: DataProto, num_examine: int = 0, preappend_token: str = "") -> torch.Tensor:
        reward_tensor, _ = self._compute_rewards_common(data, is_validation=False, num_examine=num_examine, preappend_token=preappend_token)
        return reward_tensor
    
    def _validate_score(self, data: DataProto, return_dict = True, preappend_token: str = ""):
        
        reward_tensor, responses_by_index, reward_components = self._compute_rewards_common(data, is_validation=True, preappend_token=preappend_token)
        
        if self.use_rm:
            print("-------------------------------- [DEBUG] using rm --------------------------------")
            batch = self._add_decoded_text_to_batch(batch)           
            # First compute student reward model score
            rm_output = self.rm_wg.compute_rm_score(batch)
            batch = batch.union(rm_output)
            student_reward_tensor = rm_output.batch["rm_scores"]           
            # Also compute teacher rule-based reward
            teacher_reward_tensor = self.reward_fn(batch)           
            # Combine rewards with configurable weights
            teacher_weight = getattr(self.config.reasoning_gym, 'teacher_reward_weight', 0.5)
            student_weight = getattr(self.config.reasoning_gym, 'student_reward_weight', 0.5)
            reward_tensor = teacher_weight * teacher_reward_tensor + student_weight * student_reward_tensor


        for index, data_info in responses_by_index.items():
            source_data = data_info['source_data']
            new_val_path = os.path.join(self.val_path, f"{source_data}.json")
            
            # Save meta_info only once at step 0
            if self.global_steps == 0:
                auto_json.save_entry(
                    filepath=new_val_path,
                    step="meta_info",
                    index=index,
                    response=data_info['entry']['question'],
                    reward=0.0
                )
            
            # Save regular entry with responses and rewards
            #first_response = data_info['question'][0] if data_info['question'] else {'response': '', 'reward': 0.0}
            auto_json.save_entry(
                filepath=new_val_path,
                step=self.global_steps,
                index=index,
                response=data_info['entry']['question'],
                reward=0,
                extra_data={'all_responses': data_info['responses']}
            )
        
        result = {'reward_tensor': reward_tensor}
        result['reward_extra_info'] = {}
        for reward_name, reward_value in reward_components.items():
            result['reward_extra_info'][reward_name] = reward_value

        if self.use_rm:
            result['reward_extra_info']['student_reward'] = student_reward_tensor.sum(-1).cpu().tolist()
            result['reward_extra_info']['teacher_reward'] = teacher_reward_tensor.sum(-1).cpu().tolist()
        
        # Check if n-shot evaluation is enabled
        if (hasattr(self.config, 'reasoning_gym') and 
            getattr(self.config.reasoning_gym, 'n-shot-reward', False)):
            
            n_shot = getattr(self.config.reasoning_gym, 'n', 3)
            # print(f"[DEBUG] Running {n_shot}-shot evaluation...")
            
            # Run few-shot evaluation using reward model worker
            if hasattr(self, 'rm_wg') and self.rm_wg is not None:
                # Pre-decode text using teacher tokenizer before sending to reward model
                data = self._add_decoded_text_to_batch(data)
                
                # Call the few-shot evaluation
                few_shot_output = self.rm_wg.compute_few_shot_reward(data, n_shot=n_shot)
                student_acc = few_shot_output.batch.get("student_acc", 0.0)
                
                # print(f"[DEBUG] Few-shot student accuracy: {student_acc:.3f}")
                result['student-acc'] = float(student_acc)
            else:
                # print("[DEBUG] Reward model worker not available for few-shot evaluation")
                result['student-acc'] = 0.0

        return result

    def _get_entry_for_index(self, index: int) -> dict:
        """Get the data entry for a given index from training dataset."""
        entry = self.train_dataset[index].copy()  # Make a copy to avoid modifying original
        entry['index'] = index
        return entry
    
    def _get_val_entry_for_index(self, index: int) -> dict:
        """Get the data entry for a given index from validation dataset."""
        entry = self.val_dataset[index].copy()  # Make a copy to avoid modifying original
        entry['index'] = index
        return entry

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )


        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # Inject total_training_steps to actor/critic optim_config
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)