"""
Usefulness Metric

Handles Chain-of-Thought (CoT) perturbation evaluation to measure 
the usefulness of reasoning traces through various perturbation techniques.
"""

import json
import logging
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base_metric import BaseMetric

logger = logging.getLogger(__name__)


class UsefulnessMetric(BaseMetric):
    """Handles CoT perturbation evaluation to measure reasoning trace usefulness.
    
    Combines original CoT perturbation functionality with checkpoint tracking
    and analysis capabilities for comprehensive usefulness evaluation.
    """
    
    def __init__(self, config: Dict[str, Any], reward_calculator, prompt_manager):
        """Initialize usefulness metric with checkpoint tracking."""
        super().__init__(config, reward_calculator, prompt_manager)
        
        # Initialize checkpoint tracking from BaseMetrics functionality
        self.checkpoint_results = {}
        self.aggregated_metrics = {}
    
    def _init_metric_config(self):
        """Initialize usefulness metric configuration."""
        requested_metrics = self.config.get('evaluation', {}).get('metrics', [])

        # All supported perturbation types
        self.cot_types = ['truncate', 'filler', 'shuffle', 'remove_thinking', 'incremental_thinking', 'expert_thinking', 'truncate_random', 'truncate_second', 'truncate_first', 'replace']

        # Perturbations with special handling (not using standard perturb_teacher_responses flow)
        self.special_perturbations = {'incremental_thinking'}

        self.requested_cot_metrics = [r for r in requested_metrics if r in self.cot_types]

        # Split into standard and special perturbations
        self.standard_perturbations = [m for m in self.requested_cot_metrics if m not in self.special_perturbations]
        self.special_requested = [m for m in self.requested_cot_metrics if m in self.special_perturbations]
    
    def can_run(self) -> bool:
        """Check if CoT perturbation evaluation should be run."""
        return len(self.requested_cot_metrics) > 0
    
    def evaluate(self, teacher_responses: List[Dict[str, Any]], model_manager=None, 
                openai_client=None, **kwargs) -> List[Dict[str, Any]]:
        """Run CoT perturbation evaluation on teacher responses."""
        checkpoint_path = kwargs.get('checkpoint_path', '')
        accuracy_metric = kwargs.get('accuracy_metric', None)
        
        if not accuracy_metric:
            raise ValueError("AccuracyMetric instance required for CoT perturbation evaluation")
        
        return self._run_cot_perturbation(checkpoint_path, teacher_responses, accuracy_metric, model_manager, openai_client)
    
    def _run_cot_perturbation(self, checkpoint_path: str, teacher_responses: List[Dict[str, Any]], 
                             accuracy_metric, model_manager=None, openai_client=None) -> List[Dict[str, Any]]:
        """Run CoT perturbation on existing teacher responses."""
        from ..utils.cot_perturbation_utils import perturb_teacher_responses
        
        logger.info("Running CoT perturbation...")
        
        # Load expert thinking if requested
        if "expert_thinking" in self.requested_cot_metrics:
            expert_thinking_config = self.config['evaluation']['expert_thinking_dir']

            if not isinstance(expert_thinking_config, dict):
                logger.error(f"expert_thinking_dir must be a dictionary mapping model_name -> path")
                raise ValueError("expert_thinking_dir must be a dictionary format")

            # Load expert thinking traces for each model
            for model_name, expert_path in expert_thinking_config.items():
                with open(expert_path, 'r') as f:
                    expert_thinking = json.load(f)
                    for idx, item in enumerate(expert_thinking):
                        if idx < len(teacher_responses):
                            # Store multiple expert traces with model name as key
                            if 'expert_thinking_traces' not in teacher_responses[idx]:
                                teacher_responses[idx]['expert_thinking_traces'] = {}
                            #teacher_responses[idx]['expert_thinking_traces'][model_name] = item['teacher_thinking']
                            teacher_responses[idx]['expert_thinking_traces'][model_name] = item['teacher_answer']

        # Load replacement thinking if requested
        if "replace" in self.requested_cot_metrics:
            import random

            replacement_thinking_config = self.config['evaluation']['replacement_thinking_dir']

            if not isinstance(replacement_thinking_config, dict):
                logger.error(f"replacement_thinking_dir must be a dictionary mapping replacement_id -> path")
                raise ValueError("replacement_thinking_dir must be a dictionary format")

            # Load replacement thinking traces for each replacement source
            for replacement_id, replacement_path in replacement_thinking_config.items():
                with open(replacement_path, 'r') as f:
                    replacement_thinking = json.load(f)

                    # Randomly sample replacements for each teacher response
                    for idx in range(len(teacher_responses)):
                        # Use a fixed seed based on (idx, replacement_id) for reproducibility
                        random.seed(hash((idx, replacement_id)) % (2**32))

                        # Randomly sample one item from the replacement thinking list
                        sampled_item = random.choice(replacement_thinking)
                        sampled_idx = replacement_thinking.index(sampled_item)

                        # Store multiple replacement traces with replacement_id as key
                        if 'replacement_thinking_traces' not in teacher_responses[idx]:
                            teacher_responses[idx]['replacement_thinking_traces'] = {}
                        teacher_responses[idx]['replacement_thinking_traces'][replacement_id] = {
                            'thinking': sampled_item.get('teacher_thinking', sampled_item.get('teacher_answer', '')),
                            'answer': sampled_item.get('teacher_answer', ''),
                            'gold_answer': sampled_item.get('answer', sampled_item.get('gold_answer', '')),
                            'sampled_idx': sampled_idx  # Track which replacement was used
                        }

                logger.info(f"Loaded and randomly sampled {len(teacher_responses)} replacement traces from {replacement_id}")

        # Generate perturbed data (only standard perturbations use perturb_teacher_responses)
        perturbed_data = perturb_teacher_responses(teacher_responses, self.standard_perturbations)
        
        # Run perturbation inference for each perturbation type
        for pert_type, pert_responses in perturbed_data["perturbed"].items():

            logger.info(f"Running {pert_type} perturbation inference ({len(pert_responses)} samples)")
            own_thinking = pert_type not in ["expert_thinking", "replace"]

            if pert_type == "expert_thinking":
                logger.info("Using expert thinking")

                # For expert thinking, group responses by expert model
                expert_model_groups = {}
                for resp in pert_responses:
                    expert_model = resp.get('expert_model_name', 'default')
                    if expert_model not in expert_model_groups:
                        expert_model_groups[expert_model] = []
                    expert_model_groups[expert_model].append(resp)

                # Initialize expert_thinking_perturbation for all responses
                for original_resp in teacher_responses:
                    original_resp['expert_thinking_perturbation'] = {
                        'expert_perplexity': {},
                        'expert_token_logprobs': {},
                        'reward_scores': {},
                        'perturbed_outputs': {},
                        'perturbed_inputs': {}
                    }

                # Process each expert model separately
                for expert_model_name, expert_responses in expert_model_groups.items():
                    logger.info(f"Running expert thinking perturbation for {expert_model_name} ({len(expert_responses)} samples)")

                    # Generate responses using accuracy metric
                    expert_results = accuracy_metric.evaluate(
                        expert_responses,
                        model_manager=model_manager,
                        openai_client=openai_client,
                        own_thinking=False
                    )

                    # Match results back to original responses by index
                    for expert_resp, expert_result in zip(expert_responses, expert_results):
                        # Find the matching original response by index
                        original_index = expert_resp.get('index', -1)

                        # Find teacher response with matching index
                        original_resp = None
                        for teacher_resp in teacher_responses:
                            if teacher_resp.get('index') == original_index:
                                original_resp = teacher_resp
                                break

                        if original_resp is not None:
                            # Store expert-specific results - extract from k_responses
                            k_responses = expert_result.get('k_responses', [])
                            if k_responses:
                                # Use best response for expert thinking
                                best_k = max(k_responses, key=lambda x: x.get('reward_score', 0.0))

                                original_resp['expert_thinking_perturbation']['expert_perplexity'][expert_model_name] = best_k.get('generation_info', {}).get('expert_perplexity', 0.0)
                                original_resp['expert_thinking_perturbation']['expert_token_logprobs'][expert_model_name] = best_k.get('generation_info', {}).get('expert_token_logprobs', [])
                                original_resp['expert_thinking_perturbation']['reward_scores'][expert_model_name] = best_k.get('reward_score', 0.0)
                                original_resp['expert_thinking_perturbation']['perturbed_outputs'][expert_model_name] = best_k.get('teacher_response', '')
                                original_resp['expert_thinking_perturbation']['perturbed_inputs'][expert_model_name] = expert_result.get('question', '')
            elif pert_type == "replace":
                logger.info("Using replacement thinking")

                # For replace, group responses by replacement_id
                replacement_groups = {}
                for resp in pert_responses:
                    replacement_id = resp.get('replacement_id', 'default')
                    if replacement_id not in replacement_groups:
                        replacement_groups[replacement_id] = []
                    replacement_groups[replacement_id].append(resp)

                # Initialize replace_perturbation for all responses
                for original_resp in teacher_responses:
                    original_resp['replace_perturbation'] = {
                        'reward_scores': {},
                        'perturbed_outputs': {},
                        'perturbed_inputs': {},
                        'replacement_answers': {}
                    }

                # Process each replacement separately
                for replacement_id, replacement_responses in replacement_groups.items():
                    logger.info(f"Running replace perturbation for {replacement_id} ({len(replacement_responses)} samples)")

                    # Generate responses using accuracy metric
                    # Note: The answer field in replacement_responses already contains the replacement answer for scoring!
                    replace_results = accuracy_metric.evaluate(
                        replacement_responses,
                        model_manager=model_manager,
                        openai_client=openai_client,
                        own_thinking=True  # Generate own thinking
                    )

                    # Match results back to original responses by index
                    for replace_resp, replace_result in zip(replacement_responses, replace_results):
                        # Find the matching original response by index
                        original_index = replace_resp.get('index', -1)

                        # Find teacher response with matching index
                        original_resp = None
                        for teacher_resp in teacher_responses:
                            if teacher_resp.get('index') == original_index:
                                original_resp = teacher_resp
                                break

                        if original_resp is not None:
                            # Store replacement-specific results - extract from k_responses
                            # Note: reward_score is calculated against the replacement answer!
                            k_responses = replace_result.get('k_responses', [])
                            if k_responses:
                                # Use best response for replacement
                                best_k = max(k_responses, key=lambda x: x.get('reward_score', 0.0))

                                original_resp['replace_perturbation']['reward_scores'][replacement_id] = best_k.get('reward_score', 0.0)
                                original_resp['replace_perturbation']['perturbed_outputs'][replacement_id] = best_k.get('teacher_response', '')
                                original_resp['replace_perturbation']['perturbed_inputs'][replacement_id] = replace_result.get('question', '')
                                original_resp['replace_perturbation']['replacement_answers'][replacement_id] = replace_resp.get('replacement_answer', '')
            else:
                # Generate responses using accuracy metric for non-expert/non-replace perturbations
                pert_results = accuracy_metric.evaluate(
                    pert_responses,
                    model_manager=model_manager,
                    openai_client=openai_client,
                    own_thinking=own_thinking
                )

                # Add results to original responses
                for original_resp, pert_result in zip(teacher_responses, pert_results):
                    # Extract from k_responses - use best response (first one typically has best reward due to sorting)
                    k_responses = pert_result.get('k_responses', [])
                    if k_responses:
                        # Find the response with highest reward
                        best_k_response = max(k_responses, key=lambda x: x.get('reward_score', 0.0))

                        original_resp[f'{pert_type}_perturbation'] = {
                            'perturbed_input': pert_result.get('question', ''),
                            'perturbed_output': best_k_response.get('teacher_response', ''),
                            'reward_score': best_k_response.get('reward_score', 0.0),
                            'token_logprobs': best_k_response.get('generation_info', {}).get('token_logprobs', []),
                            'all_k_rewards': [kr.get('reward_score', 0.0) for kr in k_responses],
                            'best_reward_score': pert_result.get('best_reward_score', 0.0),
                            'mean_reward_score': pert_result.get('mean_reward_score', 0.0)
                        }
                    else:
                        # Fallback for empty k_responses
                        logger.warning(f"Empty k_responses for {pert_type} perturbation")
                        original_resp[f'{pert_type}_perturbation'] = {
                            'perturbed_input': pert_result.get('question', ''),
                            'perturbed_output': '',
                            'reward_score': 0.0,
                            'token_logprobs': []
                        }

        # Handle special perturbations with custom logic
        if "incremental_thinking" in self.special_requested:
            logger.info("Running incremental thinking evaluation")
            self._evaluate_incremental_thinking(teacher_responses, accuracy_metric, model_manager, openai_client)

        return teacher_responses

    def _evaluate_incremental_thinking(self, teacher_responses: List[Dict[str, Any]],
                                      accuracy_metric, model_manager=None, openai_client=None):
        """
        Evaluate answer perplexity at incremental thinking truncation levels.

        For each sample, truncates thinking word-by-word (1 word, 2 words, ..., n words)
        and evaluates answer perplexity at each level.

        This is batched across all samples for efficiency.

        Args:
            teacher_responses: List of teacher response dictionaries
            accuracy_metric: AccuracyMetric instance for generation
            model_manager: Model manager for generation
            openai_client: OpenAI client (if using OpenAI)
        """
        logger.info(f"Starting incremental thinking evaluation for {len(teacher_responses)} samples")

        # Initialize incremental_thinking_evaluation for all responses
        for resp in teacher_responses:
            resp['incremental_thinking_evaluation'] = {
                'perplexities': [],
                'num_words': 0,
                'truncation_levels': []
            }

        # BATCHED APPROACH: Prepare all eval items across all samples
        all_eval_items = []
        sample_metadata = []  # Track which items belong to which sample

        for idx, response in enumerate(teacher_responses):
            # Get thinking and gold answer
            if 'k_responses' in response and response['k_responses']:
                teacher_thinking = response['k_responses'][0].get('teacher_thinking', '')
            else:
                teacher_thinking = response.get('teacher_thinking', '')

            if not teacher_thinking or not teacher_thinking.strip():
                logger.warning(f"Sample {idx}: No thinking found, skipping")
                continue

            # Split thinking into words
            words = teacher_thinking.split()
            num_words = len(words)

            if num_words == 0:
                continue

            # Get the question prompt and gold answer
            full_prompt = response.get('full_prompt', response.get('question', ''))
            gold_answer = response.get('gold_answer', response.get('answer', ''))

            # Create eval items for each truncation level
            sample_start_idx = len(all_eval_items)
            for word_count in range(1, num_words + 1):
                truncated_thinking = ' '.join(words[:word_count])
                prompt_without_answer = f"{full_prompt}{truncated_thinking}</think>\n<answer>"
                full_truncated_prompt = f"{prompt_without_answer}{gold_answer}</answer>"

                eval_item = {
                    'question': full_truncated_prompt,
                    'full_prompt': prompt_without_answer,
                    'answer': gold_answer,
                    'index': response.get('index', idx),
                    'truncation_level': word_count,
                    'metadata': response.get('metadata', {}),
                }
                all_eval_items.append(eval_item)

            # Track metadata for this sample
            sample_metadata.append({
                'sample_idx': idx,
                'num_words': num_words,
                'start_idx': sample_start_idx,
                'end_idx': len(all_eval_items)  # exclusive
            })

        if not all_eval_items:
            logger.warning("No eval items created for incremental thinking")
            return

        # BATCH EVALUATE ALL ITEMS AT ONCE
        logger.info(f"Evaluating {len(all_eval_items)} prompts across {len(sample_metadata)} samples in batch...")
        all_results = accuracy_metric.evaluate(
            all_eval_items,
            model_manager=model_manager,
            openai_client=openai_client,
            own_thinking=False
        )

        # SPLIT RESULTS BACK TO SAMPLES
        for meta in sample_metadata:
            sample_idx = meta['sample_idx']
            start_idx = meta['start_idx']
            end_idx = meta['end_idx']
            num_words = meta['num_words']

            sample_results = all_results[start_idx:end_idx]

            # Extract perplexities
            perplexities = []
            question_perplexity = 0.0
            for result_idx, result in enumerate(sample_results):
                perplexity = result['k_responses'][0]['generation_info'].get('expert_perplexity', 0.0)
                perplexities.append(perplexity)

                if result_idx == 0:
                    question_perplexity = result['k_responses'][0]['generation_info'].get('question_perplexity', 0.0)

            # Store results
            teacher_responses[sample_idx]['incremental_thinking_evaluation'] = {
                'perplexities': perplexities,
                'question_perplexity': question_perplexity,
                'num_words': num_words,
                'truncation_levels': list(range(1, num_words + 1))
            }

            logger.info(f"Sample {sample_idx}: Question PPL = {question_perplexity:.2f}, Answer PPL = {perplexities[:5]}{'...' if len(perplexities) > 5 else ''}")

        logger.info(f"Incremental thinking evaluation completed: {len(all_eval_items)} prompts evaluated")

    def process_teacher_responses(self, teacher_responses_path: str, 
                                teacher_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process teacher responses and calculate perturbation reward metrics.
        This is called AFTER the perturbation inference has been completed.
        
        Args:
            teacher_responses_path: Path where original responses are saved
            teacher_data: Teacher responses with perturbation results
        
        Returns:
            Dictionary with perturbation metrics
        """
        try:            
            # Calculate original reward using average reward
            reward_scores = [item.get('reward_score', 0.0) for item in teacher_data]
            original_reward = sum(reward_scores) / len(reward_scores) if reward_scores else 0.0
            
            # Calculate truncate reward from perturbation results
            truncate_scores = []
            filler_scores = []
            shuffle_scores = []
            remove_thinking_scores = []
            expert_thinking_scores_by_model = {}
            
            for item in teacher_data:
                if 'truncate_perturbation' in item:
                    truncate_scores.append(item['truncate_perturbation'].get('reward_score', 0.0))
                if 'filler_perturbation' in item:
                    filler_scores.append(item['filler_perturbation'].get('reward_score', 0.0))
                if 'shuffle_perturbation' in item:
                    shuffle_scores.append(item['shuffle_perturbation'].get('reward_score', 0.0))
                if 'remove_thinking_perturbation' in item:
                    remove_thinking_scores.append(item['remove_thinking_perturbation'].get('reward_score', 0.0))
                if 'expert_thinking_perturbation' in item:
                    expert_rewards = item['expert_thinking_perturbation'].get('reward_scores', {})
                    for expert_model, reward in expert_rewards.items():
                        if expert_model not in expert_thinking_scores_by_model:
                            expert_thinking_scores_by_model[expert_model] = []
                        expert_thinking_scores_by_model[expert_model].append(reward)
            
            truncate_reward = sum(truncate_scores) / len(truncate_scores) if truncate_scores else 0.0
            filler_reward = sum(filler_scores) / len(filler_scores) if filler_scores else 0.0
            shuffle_reward = sum(shuffle_scores) / len(shuffle_scores) if shuffle_scores else 0.0
            remove_thinking_reward = sum(remove_thinking_scores) / len(remove_thinking_scores) if remove_thinking_scores else 0.0
            
            # Calculate expert thinking rewards for each model
            expert_thinking_rewards = {}
            for expert_model, scores in expert_thinking_scores_by_model.items():
                expert_thinking_rewards[expert_model] = sum(scores) / len(scores) if scores else 0.0

            # Process incremental thinking results
            incremental_thinking_data = self._process_incremental_thinking_results(teacher_data)

            # Prepare results
            results = {
                'original_reward': original_reward,
                'truncate_reward': truncate_reward,
                'filler_reward': filler_reward,
                'shuffle_reward': shuffle_reward,
                'remove_thinking_reward': remove_thinking_reward,
                'expert_thinking_rewards': expert_thinking_rewards,  # Dictionary with expert model names as keys
                'incremental_thinking_perplexities': incremental_thinking_data,  # Dictionary with per-level averages
                'num_samples': len(teacher_data),
                'perturbation_details': {
                    'truncate_samples': len(truncate_scores),
                    'filler_samples': len(filler_scores),
                    'shuffle_samples': len(shuffle_scores),
                    'remove_thinking_samples': len(remove_thinking_scores),
                    'expert_thinking_samples_by_model': {model: len(scores) for model, scores in expert_thinking_scores_by_model.items()},
                    'truncate_drop': original_reward - truncate_reward,
                    'filler_drop': original_reward - filler_reward,
                    'shuffle_drop': original_reward - shuffle_reward,
                    'remove_thinking_drop': original_reward - remove_thinking_reward,
                    'expert_thinking_improvements': {model: reward - original_reward for model, reward in expert_thinking_rewards.items()}
                }
            }
            
            logger.info(f"CoT Perturbation metrics: {len(teacher_data)} samples")
            logger.info(f"  Original reward: {original_reward:.4f}")
            logger.info(f"  Truncate reward: {truncate_reward:.4f} (drop: {original_reward - truncate_reward:.4f})")
            logger.info(f"  Filler reward: {filler_reward:.4f} (drop: {original_reward - filler_reward:.4f})")
            logger.info(f"  Shuffle reward: {shuffle_reward:.4f} (drop: {original_reward - shuffle_reward:.4f})")
            logger.info(f"  Remove thinking reward: {remove_thinking_reward:.4f} (drop: {original_reward - remove_thinking_reward:.4f})")
            
            # Log expert thinking rewards for each model
            for expert_model, expert_reward in expert_thinking_rewards.items():
                logger.info(f"  Expert thinking reward ({expert_model}): {expert_reward:.4f} (improvement: {expert_reward - original_reward:.4f})")

            # Log incremental thinking stats
            if incremental_thinking_data and incremental_thinking_data.get('by_level'):
                logger.info(f"  Incremental thinking: {incremental_thinking_data.get('num_samples', 0)} samples, "
                           f"avg {incremental_thinking_data.get('avg_num_words', 0):.1f} words")
                ppls = incremental_thinking_data['by_level']
                logger.info(f"    Perplexity progression: [{ppls[0]:.3f} -> {ppls[-1]:.3f}] (first to last word)")

            return results
            
        except Exception as e:
            logger.error(f"Error processing CoT perturbation metrics: {e}")
            return {
                'original_reward': 0.0,
                'truncate_reward': 0.0,
                'filler_reward': 0.0,
                'shuffle_reward': 0.0,
                'remove_thinking_reward': 0.0,
                'expert_thinking_rewards': {},
                'incremental_thinking_perplexities': {},
                'num_samples': 0,
                'error': str(e)
            }

    def _process_incremental_thinking_results(self, teacher_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process incremental thinking evaluation results to compute summary statistics.

        Args:
            teacher_data: List of teacher responses with incremental_thinking_evaluation

        Returns:
            Dictionary with:
                - by_level: List of average perplexities at each truncation level
                - num_samples: Number of samples processed
                - avg_num_words: Average number of words in thinking traces
        """
        # Collect all perplexity data by truncation level
        perplexities_by_level = {}
        question_perplexities = []
        total_words = 0
        num_samples = 0

        for item in teacher_data:
            if 'incremental_thinking_evaluation' not in item:
                continue

            eval_data = item['incremental_thinking_evaluation']
            perplexities = eval_data.get('perplexities', [])
            num_words = eval_data.get('num_words', 0)
            question_ppl = eval_data.get('question_perplexity', 0.0)

            if not perplexities or num_words == 0:
                continue

            num_samples += 1
            total_words += num_words
            question_perplexities.append(question_ppl)

            # Aggregate perplexities by level
            for level, ppl in enumerate(perplexities, start=1):
                if level not in perplexities_by_level:
                    perplexities_by_level[level] = []
                perplexities_by_level[level].append(ppl)

        if num_samples == 0:
            return {}

        # Compute average perplexity at each level
        max_level = max(perplexities_by_level.keys()) if perplexities_by_level else 0
        avg_perplexities_by_level = []

        for level in range(1, max_level + 1):
            if level in perplexities_by_level:
                avg_ppl = sum(perplexities_by_level[level]) / len(perplexities_by_level[level])
                avg_perplexities_by_level.append(avg_ppl)
            else:
                # Level not present in all samples, skip
                break

        # Calculate average question perplexity
        avg_question_perplexity = sum(question_perplexities) / len(question_perplexities) if question_perplexities else 0.0

        return {
            'by_level': avg_perplexities_by_level,
            'question_perplexity': avg_question_perplexity,
            'num_samples': num_samples,
            'avg_num_words': total_words / num_samples if num_samples > 0 else 0.0
        }

    def add_checkpoint_result(self, checkpoint_name: str, results: Dict[str, Any]) -> None:
        """
        Add results for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            results: Results dictionary from evaluation
        """
        self.checkpoint_results[checkpoint_name] = {
            **results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Added usefulness results for checkpoint {checkpoint_name}")
    
    def get_checkpoint_result(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """
        Get results for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            
        Returns:
            Results dictionary or None if not found
        """
        return self.checkpoint_results.get(checkpoint_name)
    
    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get results for all checkpoints.
        
        Returns:
            Dictionary mapping checkpoint names to results
        """
        return self.checkpoint_results
    
    def get_sorted_results(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get checkpoint results sorted by checkpoint name/step.
        
        Returns:
            List of (checkpoint_name, results) tuples sorted by step
        """
        def extract_step(checkpoint_name):
            try:
                # Try to extract step number from checkpoint name
                if 'step_' in checkpoint_name:
                    return int(checkpoint_name.split('step_')[1].split('_')[0])
                elif 'epoch_' in checkpoint_name:
                    return int(checkpoint_name.split('epoch_')[1].split('_')[0])
                else:
                    return 0
            except:
                return 0
        
        results = [(name, self.checkpoint_results[name]) 
                  for name in self.checkpoint_results.keys()]
        
        results.sort(key=lambda x: extract_step(x[0]))
        return results
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics across all checkpoints."""
        if not self.checkpoint_results:
            return {}
        
        # Extract rewards
        original_rewards = [r.get('original_reward', 0.0) for r in self.checkpoint_results.values()]
        truncate_rewards = [r.get('truncate_reward', 0.0) for r in self.checkpoint_results.values()]
        filler_rewards = [r.get('filler_reward', 0.0) for r in self.checkpoint_results.values()]
        shuffle_rewards = [r.get('shuffle_reward', 0.0) for r in self.checkpoint_results.values()]
        remove_thinking_rewards = [r.get('remove_thinking_reward', 0.0) for r in self.checkpoint_results.values()]
        
        # Extract expert thinking rewards by model
        expert_thinking_rewards_by_model = {}
        for r in self.checkpoint_results.values():
            expert_rewards = r.get('expert_thinking_rewards', {})
            for model_name, reward in expert_rewards.items():
                if model_name not in expert_thinking_rewards_by_model:
                    expert_thinking_rewards_by_model[model_name] = []
                expert_thinking_rewards_by_model[model_name].append(reward)
        
        # Calculate expert thinking statistics for each model
        expert_thinking_stats = {}
        for model_name, rewards in expert_thinking_rewards_by_model.items():
            if rewards:
                expert_thinking_stats[model_name] = {
                    'mean': float(np.mean(rewards)),
                    'std': float(np.std(rewards)),
                    'min': float(np.min(rewards)),
                    'max': float(np.max(rewards))
                }

        # Extract incremental thinking perplexities
        incremental_thinking_stats = self._summarize_incremental_thinking()

        return {
            'num_checkpoints': len(self.checkpoint_results),
            'original_reward': {
                'mean': float(np.mean(original_rewards)),
                'std': float(np.std(original_rewards)),
                'min': float(np.min(original_rewards)),
                'max': float(np.max(original_rewards))
            },
            'truncate_reward': {
                'mean': float(np.mean(truncate_rewards)),
                'std': float(np.std(truncate_rewards)),
                'min': float(np.min(truncate_rewards)),
                'max': float(np.max(truncate_rewards))
            },
            'filler_reward': {
                'mean': float(np.mean(filler_rewards)),
                'std': float(np.std(filler_rewards)),
                'min': float(np.min(filler_rewards)),
                'max': float(np.max(filler_rewards))
            },
            'shuffle_reward': {
                'mean': float(np.mean(shuffle_rewards)),
                'std': float(np.std(shuffle_rewards)),
                'min': float(np.min(shuffle_rewards)),
                'max': float(np.max(shuffle_rewards))
            },
            'remove_thinking_reward': {
                'mean': float(np.mean(remove_thinking_rewards)),
                'std': float(np.std(remove_thinking_rewards)),
                'min': float(np.min(remove_thinking_rewards)),
                'max': float(np.max(remove_thinking_rewards))
            },
            'expert_thinking_rewards': expert_thinking_stats,
            'incremental_thinking': incremental_thinking_stats
        }

    def _summarize_incremental_thinking(self) -> Dict[str, Any]:
        """Summarize incremental thinking perplexities across checkpoints."""
        if not self.checkpoint_results:
            return {}

        # Collect perplexities by level across all checkpoints
        perplexities_by_level = {}

        for checkpoint_results in self.checkpoint_results.values():
            inc_think_data = checkpoint_results.get('incremental_thinking_perplexities', {})
            by_level = inc_think_data.get('by_level', [])

            for level, ppl in enumerate(by_level, start=1):
                if level not in perplexities_by_level:
                    perplexities_by_level[level] = []
                perplexities_by_level[level].append(ppl)

        if not perplexities_by_level:
            return {}

        # Calculate statistics for each level
        level_stats = {}
        for level in sorted(perplexities_by_level.keys()):
            ppls = perplexities_by_level[level]
            level_stats[level] = {
                'mean': float(np.mean(ppls)),
                'std': float(np.std(ppls)),
                'min': float(np.min(ppls)),
                'max': float(np.max(ppls))
            }

        return {
            'by_level': level_stats,
            'num_levels': len(level_stats)
        }

    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across checkpoints."""
        if len(self.checkpoint_results) < 2:
            return {'error': 'Need at least 2 checkpoints for trend analysis'}
        
        sorted_results = self.get_sorted_results()
        
        # Extract step numbers and rewards
        steps = []
        original_rewards = []
        truncate_rewards = []
        filler_rewards = []
        shuffle_rewards = []
        remove_thinking_rewards = []
        expert_thinking_rewards_by_model = {}
        
        for checkpoint_name, results in sorted_results:
            try:
                if 'step_' in checkpoint_name:
                    step = int(checkpoint_name.split('step_')[1].split('_')[0])
                else:
                    step = 0
            except:
                step = 0
            
            steps.append(step)
            original_rewards.append(results.get('original_reward', 0.0))
            truncate_rewards.append(results.get('truncate_reward', 0.0))
            filler_rewards.append(results.get('filler_reward', 0.0))
            shuffle_rewards.append(results.get('shuffle_reward', 0.0))
            remove_thinking_rewards.append(results.get('remove_thinking_reward', 0.0))
            
            # Handle multiple expert thinking models
            expert_rewards = results.get('expert_thinking_rewards', {})
            for model_name, reward in expert_rewards.items():
                if model_name not in expert_thinking_rewards_by_model:
                    expert_thinking_rewards_by_model[model_name] = []
                expert_thinking_rewards_by_model[model_name].append(reward)
        
        return {
            'steps': steps,
            'original_rewards': original_rewards,
            'truncate_rewards': truncate_rewards,
            'filler_rewards': filler_rewards,
            'shuffle_rewards': shuffle_rewards,
            'remove_thinking_rewards': remove_thinking_rewards,
            'expert_thinking_rewards': expert_thinking_rewards_by_model
        }
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of all results.
        
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {
            'metric_name': 'usefulness',
            'summary_statistics': self.calculate_summary_statistics(),
            'performance_trends': self.analyze_performance_trends(),
            'checkpoint_results': self.checkpoint_results,
            'sorted_results': self.get_sorted_results(),
            'generated_at': datetime.now().isoformat()
        }
        
        return analysis
    
    def save_metrics(self, output_path: str) -> None:
        """
        Save metrics to file.
        
        Args:
            output_path: Path to save metrics
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        analysis = self.get_detailed_analysis()
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Saved usefulness metrics to {output_path}")
    
    def load_metrics(self, input_path: str) -> None:
        """
        Load metrics from file.
        
        Args:
            input_path: Path to load metrics from
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        if 'checkpoint_results' in data:
            self.checkpoint_results = data['checkpoint_results']
            logger.info(f"Loaded usefulness metrics for {len(self.checkpoint_results)} checkpoints")
        else:
            logger.warning("No checkpoint results found in loaded data")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary format.
        
        Returns:
            Dictionary representation of metrics
        """
        return self.get_detailed_analysis()
    
    def print_summary(self) -> None:
        """Print a summary of the CoT perturbation metrics."""
        print("\n=== USEFULNESS EVALUATION SUMMARY ===")
        
        if not self.checkpoint_results:
            print("No checkpoint results available.")
            return
        
        summary = self.calculate_summary_statistics()
        
        print(f"Checkpoints evaluated: {summary['num_checkpoints']}")
        print(f"Original Reward: {summary['original_reward']['mean']:.4f} ± {summary['original_reward']['std']:.4f}")
        print(f"Truncate Reward: {summary['truncate_reward']['mean']:.4f} ± {summary['truncate_reward']['std']:.4f}")
        print(f"Filler Reward: {summary['filler_reward']['mean']:.4f} ± {summary['filler_reward']['std']:.4f}")
        print(f"Shuffle Reward: {summary['shuffle_reward']['mean']:.4f} ± {summary['shuffle_reward']['std']:.4f}")
        print(f"Remove Thinking Reward: {summary['remove_thinking_reward']['mean']:.4f} ± {summary['remove_thinking_reward']['std']:.4f}")
        
        # Show expert thinking rewards for each model
        expert_rewards = summary.get('expert_thinking_rewards', {})
        for model_name, stats in expert_rewards.items():
            print(f"Expert Thinking Reward ({model_name}): {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Show reward drops/improvements
        orig_mean = summary['original_reward']['mean']
        trunc_mean = summary['truncate_reward']['mean']
        filler_mean = summary['filler_reward']['mean']
        shuffle_mean = summary['shuffle_reward']['mean']
        remove_thinking_mean = summary['remove_thinking_reward']['mean']

        print(f"Truncate Reward Drop: {orig_mean - trunc_mean:.4f}")
        print(f"Filler Reward Drop: {orig_mean - filler_mean:.4f}")
        print(f"Shuffle Reward Drop: {orig_mean - shuffle_mean:.4f}")
        print(f"Remove Thinking Reward Drop: {orig_mean - remove_thinking_mean:.4f}")
        
        # Show expert thinking improvements for each model
        for model_name, stats in expert_rewards.items():
            expert_mean = stats['mean']
            print(f"Expert Thinking Improvement ({model_name}): {expert_mean - orig_mean:.4f}")

        # Show incremental thinking perplexity progression
        inc_think_data = summary.get('incremental_thinking', {})
        if inc_think_data and inc_think_data.get('by_level'):
            print(f"\nIncremental Thinking Perplexity Progression:")
            level_stats = inc_think_data['by_level']
            for level in sorted(level_stats.keys())[:10]:  # Show first 10 levels
                stats = level_stats[level]
                print(f"  Level {level}: {stats['mean']:.3f} ± {stats['std']:.3f}")
            if len(level_stats) > 10:
                print(f"  ... ({len(level_stats) - 10} more levels)")

        print("=" * 50)