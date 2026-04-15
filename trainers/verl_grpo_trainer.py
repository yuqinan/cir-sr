# Adapted version of Bytedance code:
# https://github.com/volcengine/verl/blob/a65c9157bc0b85b64cd753de19f94e80a11bd871/verl/trainer/main_ppo.py

import gc
import uuid
from copy import deepcopy

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import Dataset
from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_data_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.utils.debug import marked_timer
from verl.utils.dataset.rl_dataset import collate_fn

# Import the RewardCalculator
from trainers.reward_calculator import RewardCalculator
from tqdm import tqdm
import os
import json

from data.json_utils import auto_json

class RayGRPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        role_worker_mapping: dict,
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

            # Extract model name from config for format detection
            model_name = getattr(config.actor_rollout_ref.model, 'path', None)

            self.reward_calculator = RewardCalculator(
                task=config.data.task,
                task_type=task_types,
                reward_partial=getattr(config.reasoning_gym, 'reward_partial', False),
                replacement_data_path=getattr(config.reasoning_gym, 'replacement_data_path', None),
                model_name=model_name,
            )
            
            # Pass task_types to reward_model config for vllm_reward.py
            with open_dict(config):
                config.reward_model.task_types = task_types
            
            
        elif config.data.task == "gsm8k":
            self.val_path = config.gsm8k.val_path
            self.dataset_configs = config.gsm8k.datasets
            self.val_dataset_configs = config.gsm8k.validation_dataset

            model_name = getattr(config.actor_rollout_ref.model, 'path', None)

            self.reward_calculator = RewardCalculator(
                task=config.data.task,
                task_type=task_types,
                reward_partial=getattr(config.gsm8k, 'reward_partial', False),
                model_name=model_name,
            )

        elif config.data.task == "math_hard":
            self.val_path = config.math_hard.val_path
            self.dataset_configs = config.math_hard.datasets
            self.val_dataset_configs = config.math_hard.validation_dataset

            model_name = getattr(config.actor_rollout_ref.model, 'path', None)

            self.reward_calculator = RewardCalculator(
                task=config.data.task,
                task_type=task_types,
                reward_partial=False,
                model_name=model_name,
            )

        else:
            raise ValueError(f"Unknown data.task: {config.data.task}")
        
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
    
    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")


    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
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
        # Supported reward types: cot_verifier_accuracy, quality, rule_based, format, cot_importance
        for reward_name, weight in self.reward_config.items():
            #if weight <= 0:
            #    continue

            if reward_name == "cot_verifier_accuracy":
                # Use batch processing for cot_verifier_accuracy rewards
                reward_components[reward_name] = self.reward_calculator.calculate_batch_cot_verifier_rewards(
                    predictions, entries, preappend_token=preappend_token
                )

            elif reward_name == "quality":
                # build examples by loading in json file when the directory is a list
                examples = {}
                #for i in range(len(examples)):
                #    with open(examples[i], 'r') as f:
                #        eval_result = json.loads(f)
                #    examples.append({"input_question": eval_result["metadata"]["input_str"], "thinking_traces": eval_result["teacher_thinking"], "score": eval_result["score"]})
                # Use batch processing for quality rewards
                reward_components[reward_name] = self.reward_calculator.calculate_batch_quality_rewards(
                    predictions, entries, preappend_token=preappend_token, examples=examples
                )

            elif reward_name == "rule_based" or reward_name == "format":
                # Use individual processing for rule_based and format reward types
                individual_rewards = []
                for prediction, entry in zip(predictions, entries):
                    reward = self.reward_calculator.calculate_reward(
                        prediction, entry, preappend_token=preappend_token, reward_type=reward_name
                    )
                    individual_rewards.append(reward)
                reward_components[reward_name] = individual_rewards

            elif reward_name == "cot_importance":
                # CoT importance uses JS divergence to measure importance of reasoning traces
                # Uses trainer-specific implementation with actor_rollout_wg.compute_log_prob()
                # Hardcoded truncation levels: [0, 30, 60] percentages
                reward_components[reward_name] = self.reward_calculator.calculate_batch_cot_importance_rewards_trainer(
                    predictions=predictions,
                    entries=entries,
                    preappend_token=preappend_token,
                    actor_rollout_wg=self.actor_rollout_wg,
                    tokenizer=self.tokenizer
                )

            else:
                # Unsupported reward type
                raise NotImplementedError(
                    f"Reward type '{reward_name}' is not supported. "
                    f"Supported reward types: cot_verifier_accuracy, quality, rule_based, format, cot_importance."
                )
        
        # Combine weighted rewards
        # Only apply cot_importance reward when rule_based reward is 1 (correct answer)
        final_rewards = [0.0] * len(predictions)
        for i in range(len(predictions)):
            weighted_sum = 0.0
            rule_based_reward = reward_components.get('rule_based', [0.0] * len(predictions))[i]

            for reward_name, rewards_list in reward_components.items():
                weight = self.reward_config[reward_name]
                # Only apply cot_importance reward if rule_based reward is 1 (correct answer)
                #if reward_name == 'cot_importance':
                #    if rule_based_reward >= 0.99:  # Correct answer
                #        weighted_sum += weight * rewards_list[i]
                #    # else: skip cot_importance for incorrect answers
                #else:
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
            
            #if i < 1:
            #    print(f"-------------------------------- [DEBUG] {i} --------------------------------")
            #    print(f"First response: {decoded_responses[i]}")
            #    print(f"Reward config: {self.reward_config}")
            #    print(f"Final reward: {final_reward}")
            #    if reward_components:
            #        for reward_name, rewards_list in reward_components.items():
            #            print(f"{reward_name}: {rewards_list[i]} (weight: {self.reward_config[reward_name]})")
            #if reward_components['rule_based'] == 1.0:
            #    print(f"-------------------------------- correct response--------------------------------")
            #    print(f"First response: {decoded_responses[i]}")
        
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
                    response=data_info['entry'].get('question', data_info['entry'].get('problem', '')),
                    reward=0.0
                )

            # Save regular entry with responses and rewards
            auto_json.save_entry(
                filepath=new_val_path,
                step=self.global_steps,
                index=index,
                response=data_info['entry'].get('question', data_info['entry'].get('problem', '')),
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
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
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

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            #pprint(f"Initial validation metrics: {val_metrics}")
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
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
               
            #self.config.reasoning_gym.rewards.usefulness = 0.0
            #self.config.reasoning_gym.rewards.rule_based = 1.0

            for batch_dict in tqdm(self.train_dataloader, desc="Training Progress"):
                metrics = {}
                timing_raw = {}

                # make this into a batch of mini batch ig?
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation: extracting mini batch?
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False

                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    #print(batch.batch["ref_log_probs"][0])
                    # in what case we need a reward model?
                    with marked_timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.

                        if self.use_rm:
                            # Pre-decode text using teacher tokenizer before sending to reward model
                            # This ensures the student model gets properly decoded text
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

                        else:
                            # Use rule-based reward function
                            reward_tensor = self.reward_fn(batch)

                        # Set the final reward scores                      
                        batch.batch["token_level_scores"] = reward_tensor

                        # Clean up GPU memory after reward computation to prevent OOM
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        #print("-------------------------------- kl_metrics --------------------------------")
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            norm_adv_by_std_in_grpo = self.config.actor_rollout_ref.actor.get("norm_adv_by_std_in_grpo", True),
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )
                    
                    #print("-------------------------------- updating critic --------------------------------")
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    print("-------------------------------- updating actor --------------------------------")
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        #print(actor_output_metrics)
                        #print(batch.batch["ref_log_probs"][0])
                        metrics.update(actor_output_metrics)
                    
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, {}, timing_raw, rollout_data_dir)
                        
                    # validate
                    #print("-------------------------------- validating --------------------------------")
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        #print("-------------------------------- val_metrics --------------------------------")
                        print(val_metrics)
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: implement actual tflpo and theoretical tflpo

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    print(f"Final validation metrics: {last_val_metrics}")
                    
                    # Print API cost summary at end of training
                    self.reward_calculator.openai_client.print_cost_summary("Training Complete")
                    
                    return

                self.global_steps += 1
                gc.collect()