"""
Accuracy Metric

Handles teacher accuracy evaluation and perplexity calculation.
Receives model dependencies from TeacherSingleModel rather than managing them directly.
"""

import json
import logging
import numpy as np
import os
import re
import string
import glob
import pandas as pd
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

import torch
from vllm import SamplingParams

from .base_metric import BaseMetric
from data.utils import extract_answer, extract_answer_and_think

logger = logging.getLogger(__name__)


class AccuracyMetric(BaseMetric):
    """Handles teacher accuracy evaluation and perplexity calculation.
    
    Combines response generation with checkpoint tracking and analysis
    capabilities for comprehensive accuracy evaluation.
    """
    
    def __init__(self, config: Dict[str, Any], reward_calculator, prompt_manager):
        """Initialize accuracy metric with checkpoint tracking."""
        super().__init__(config, reward_calculator, prompt_manager)

        # Initialize checkpoint tracking from BaseMetrics functionality
        self.checkpoint_results = {}
        self.aggregated_metrics = {}

        # Get k parameter (number of responses per question)
        self.k = self.teacher_config.get('k', 1)
    
    def _init_metric_config(self):
        """Initialize accuracy metric configuration."""
        requested_metrics = self.config.get('evaluation', {}).get('metrics', [])
        self.accuracy_types = ['teacher_accuracy', 'perplexity', 'ngram_w_expert']
        self.requested_accuracy_metrics = [r for r in requested_metrics if r in self.accuracy_types]

        # Initialize expert traces if testset_expert_traces is provided
        self.expert_traces = {}
        expert_traces_dir = self.config.get('evaluation', {}).get('testset_expert_traces', None)
        if expert_traces_dir:
            self._load_expert_traces()
            # Automatically add ngram_w_expert to metrics if not already there
            if 'ngram_w_expert' not in self.requested_accuracy_metrics:
                self.requested_accuracy_metrics.append('ngram_w_expert')
    
    def can_run(self) -> bool:
        """Check if accuracy evaluation should be run."""
        return len(self.requested_accuracy_metrics) > 0
    
    def evaluate(self, teacher_data: List[Dict[str, Any]], model_manager=None, 
                openai_client=None, **kwargs) -> List[Dict[str, Any]]:
        """Generate teacher responses and calculate accuracy metrics."""
        own_thinking = kwargs.get('own_thinking', True)
        
        # Generate responses using provided model dependencies
        if self.use_openai_api:
            if not openai_client:
                raise ValueError("OpenAI client required for API-based evaluation")
            return self._generate_openai_response(teacher_data, openai_client)
        else:
            if not model_manager or not model_manager.current_model:
                raise ValueError("Model manager with loaded model required for local evaluation")
            return self._generate_local_response(teacher_data, model_manager, own_thinking=own_thinking)
    
    async def _generate_openai_response_async(self, teacher_data: List[Dict[str, Any]], openai_client) -> List[Dict[str, Any]]:
        """Generate responses using OpenAI API with batched processing for efficiency."""
        logger.info(f"Generating {self.k} responses via OpenAI for {len(teacher_data)} examples")

        # Prepare prompts
        prompts = [self.prompt_manager.create_openai_prompt(item['question']) for item in teacher_data]

        all_responses = []

        # Process in batches for better memory management and API rate limiting
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]

            logger.info(f"Processing OpenAI batch {i//self.batch_size + 1}/{(len(prompts) + self.batch_size - 1)//self.batch_size}")

            # Generate k responses for each prompt in the batch
            # Repeat each prompt k times for batch processing
            expanded_batch_prompts = []
            for prompt in batch_prompts:
                expanded_batch_prompts.extend([prompt] * self.k)

            # Generate all responses for the batch at once
            batch_responses_flat = await openai_client.generate_individual_async(
                expanded_batch_prompts,
                temperature=self.teacher_config['temperature'],
                max_tokens=self.teacher_config['max_tokens'],
                top_p=self.teacher_config['top_p'],
                n=1  # Generate one response per prompt (already repeated k times)
            )

            # Reshape responses back to k responses per original prompt
            batch_responses = []
            for j in range(len(batch_prompts)):
                k_responses = batch_responses_flat[j * self.k:(j + 1) * self.k]
                batch_responses.append(k_responses)

            all_responses.extend(batch_responses)

        # Process responses
        return self._process_responses(teacher_data, all_responses, prompts, openai_client=openai_client)

    def _generate_openai_response(self, teacher_data: List[Dict[str, Any]], openai_client) -> List[Dict[str, Any]]:
        """Generate responses using OpenAI API (sync wrapper)."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self._generate_openai_response_async(teacher_data, openai_client))
            return result
        finally:
            loop.close()
    
    def _generate_local_response(self, teacher_data: List[Dict[str, Any]], model_manager, own_thinking: bool = False) -> List[Dict[str, Any]]:
        """Generate responses using local vLLM model."""
        logger.info(f"Generating responses for {len(teacher_data)} examples")
        
        # Check if this is CoT perturbation
        is_perturbation = any('perturbation_type' in item for item in teacher_data)
        
        # Prepare prompts
        tokenizer = model_manager.current_model.get_tokenizer() if not is_perturbation else None
        prompts = [
            self.prompt_manager.create_teacher_prompt(
                item['question'], 
                tokenizer, 
                is_perturbation=is_perturbation
            ) 
            for item in teacher_data
        ]
        
        prompts_without_thinking = None
        if not own_thinking:
            prompts_without_thinking = [item.get('full_prompt', '') for item in teacher_data]
        
        # Set up sampling parameters
        sampling_params = self._create_sampling_params(model_manager)
        
        # Generate in batches
        all_responses = self._generate_in_batches(prompts, sampling_params, model_manager, own_thinking, prompts_without_thinking)
        
        # Process responses
        return self._process_responses(teacher_data, all_responses, prompts, model_manager=model_manager, is_perturbation=is_perturbation)
    
    def _create_sampling_params(self, model_manager) -> SamplingParams:
        """Create sampling parameters for vLLM generation."""
        # Check if logprobs are needed
        metrics = self.config.get('evaluation', {}).get('metrics', [])
        enable_logprobs = 'perplexity' in metrics
        enable_prompt_logprobs = ('expert_thinking' in metrics or 'incremental_thinking' in metrics)

        save_token_logprobs = self.teacher_config.get('save_token_logprobs', True)
        
        stop_tokens = model_manager.get_stop_tokens()
        logger.info(f"enable_prompt_logprobs: {enable_prompt_logprobs}")
        logger.info(f"Generating k={self.k} responses per question")

        return SamplingParams(
            temperature=self.teacher_config['temperature'],
            top_p=self.teacher_config['top_p'],
            top_k=self.teacher_config['top_k'],
            max_tokens=self.teacher_config['max_tokens'],
            stop=stop_tokens,
            logprobs=20, #if (enable_logprobs or save_token_logprobs) else None,
            prompt_logprobs=20 if enable_prompt_logprobs else None,
            n=self.k  # Generate k responses per prompt
        )
    
    def _generate_in_batches(self, prompts: List[str], sampling_params: SamplingParams, model_manager,
                           own_thinking: bool = True, prompts_without_thinking: List[str] = None) -> List[str]:
        """Generate responses in batches. Returns k responses per prompt."""
        all_responses = []

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            if not own_thinking:
                batch_prompt_without_thinking = prompts_without_thinking[i:i + self.batch_size]
                tokenizer = model_manager.current_model.get_tokenizer()

                # CRITICAL FIX: We need to calculate the length based on what vLLM ACTUALLY sees
                # The prompts sent to vLLM (batch_prompts) have the full chat template applied
                # We need to find where the answer starts in THOSE prompts, not the raw prompts
                len_prompts_without_thinking = []
                for full_prompt_with_answer in batch_prompts:
                    # Find the "<answer>" tag position in the actual vLLM prompt
                    answer_tag_pos = full_prompt_with_answer.rfind('<answer>')
                    if answer_tag_pos != -1:
                        # Tokenize everything UP TO AND INCLUDING "<answer>"
                        prompt_up_to_answer = full_prompt_with_answer[:answer_tag_pos + len('<answer>')]
                        len_without_thinking = len(tokenizer.encode(prompt_up_to_answer))
                        len_prompts_without_thinking.append(len_without_thinking)
                    else:
                        # Fallback to old method
                        logger.warning("Could not find <answer> tag in prompt, using fallback")
                        len_prompts_without_thinking.append(len(tokenizer.encode(batch_prompt_without_thinking[len(len_prompts_without_thinking)])))



            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(prompts) + self.batch_size - 1)//self.batch_size}")

            outputs = model_manager.current_model.generate(batch_prompts, sampling_params, use_tqdm=False)

            batch_responses = []
            for idx, output in enumerate(outputs):
                # Collect all k responses for this prompt
                k_responses = []
                for k_idx in range(len(output.outputs)):
                    response_text = output.outputs[k_idx].text

                    # DEBUG: Log first generation for remove_thinking
                    if idx == 0 and k_idx == 0 and not response_text.strip():
                        logger.warning(f"[GENERATION DEBUG] Empty response_text from vLLM")
                        logger.warning(f"[GENERATION DEBUG] Finish reason: {output.outputs[k_idx].finish_reason}")
                        logger.warning(f"[GENERATION DEBUG] Prompt (first 200 chars): {batch_prompts[idx][:200]}")
                        logger.warning(f"[GENERATION DEBUG] own_thinking: {own_thinking}")

                    # Calculate perplexity if logprobs available
                    len_prompt = len_prompts_without_thinking[idx] if not own_thinking else None
                    generation_info = self._extract_generation_info_for_k(
                        output, k_idx, own_thinking=own_thinking, len_prompt_without_thinking=len_prompt
                    )

                    k_responses.append({
                        'text': response_text,
                        'generation_info': generation_info
                    })

                batch_responses.append(k_responses)

            all_responses.extend(batch_responses)

        return all_responses
    
    def _extract_generation_info_for_k(self, output, k_idx: int, own_thinking, len_prompt_without_thinking = None) -> Dict[str, Any]:
        """Extract generation info including perplexity from vLLM output for k-th response."""
        info = {
            'finish_reason': output.outputs[k_idx].finish_reason,
            'num_tokens': len(output.outputs[k_idx].token_ids) if hasattr(output.outputs[k_idx], 'token_ids') else 0,
        }

        if own_thinking:
            logprobs = output.outputs[k_idx].logprobs
            tokens = output.outputs[k_idx].token_ids
            perplexity = self._calculate_perplexity(logprobs, tokens, is_expert=not own_thinking, len_prompt_without_thinking=len_prompt_without_thinking)
            info['perplexity'] = perplexity

        else:
            logprobs = output.prompt_logprobs
            tokens = output.prompt_token_ids
            perplexity = self._calculate_perplexity(logprobs, tokens, is_expert=not own_thinking, len_prompt_without_thinking=len_prompt_without_thinking)
            info['expert_perplexity'] = perplexity

            # Calculate question perplexity (before <think> tag)
            question_perplexity = self._calculate_question_perplexity(logprobs, tokens)
            info['question_perplexity'] = question_perplexity


        # Save token logprobs if requested (matching old implementation format)
        if self.teacher_config.get('save_token_logprobs', False):
            if own_thinking:
                info['token_logprobs'] = self._extract_logprobs(
                    logprobs,
                    tokens,
                    is_expert=not own_thinking)

        # Always save expert token logprobs for expert thinking case
        if not own_thinking:
            info['expert_token_logprobs'] = self._extract_logprobs(
                logprobs,
                tokens,
                is_expert=not own_thinking,
                len_prompt_without_thinking = len_prompt_without_thinking)


        return info

    def _extract_generation_info(self, output, own_thinking, len_prompt_without_thinking = None) -> Dict[str, Any]:
        """Extract generation info including perplexity from vLLM output (for backward compatibility with k=1)."""
        return self._extract_generation_info_for_k(output, 0, own_thinking, len_prompt_without_thinking)

    def _load_expert_traces(self):
        """Load expert traces from testset_expert_traces parquet file."""
        expert_traces_path = self.config.get('evaluation', {}).get('testset_expert_traces', None)

        if not expert_traces_path:
            logger.warning("ngram_w_expert metric requested but testset_expert_traces not specified in config")
            return

        if not os.path.exists(expert_traces_path):
            logger.warning(f"Expert traces file not found: {expert_traces_path}")
            return

        logger.info(f"Loading expert traces from {expert_traces_path}")

        # Check if it's a parquet file
        if expert_traces_path.endswith('.parquet'):
            # Load parquet file
            df = pd.read_parquet(expert_traces_path)

            # Check for required columns
            if 'question' not in df.columns:
                logger.error(f"'question' column not found in parquet file. Available columns: {df.columns.tolist()}")
                return

            if 'teacher_thinking_without_answer' not in df.columns:
                logger.error(f"'teacher_thinking_without_answer' column not found in parquet file. Available columns: {df.columns.tolist()}")
                return

            # Build expert traces dictionary from parquet
            for _, row in df.iterrows():
                question = row['question']
                expert_thinking = row['teacher_thinking_without_answer']

                # Skip if either is NaN or empty
                if pd.isna(question) or pd.isna(expert_thinking):
                    continue

                self.expert_traces[question] = expert_thinking

            logger.info(f"Loaded {len(self.expert_traces)} expert traces from parquet file")

        else:
            logger.error(f"testset_expert_traces must be a .parquet file, got: {expert_traces_path}")
            return

    def _normalize_text(self, text: str) -> str:
        """Normalize text by lowercasing and removing punctuation."""
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from a list of tokens."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def _calculate_ngram_overlap(self, teacher_thinking: str, expert_thinking: str, n: int = 1) -> Dict[str, float]:
        """
        Calculate n-gram overlap between teacher and expert thinking.

        Returns precision, recall, and F1 score for the given n-gram size.
        """
        # Normalize texts
        teacher_normalized = self._normalize_text(teacher_thinking)
        expert_normalized = self._normalize_text(expert_thinking)

        # Tokenize (split on whitespace)
        teacher_tokens = teacher_normalized.split()
        expert_tokens = expert_normalized.split()

        if not teacher_tokens or not expert_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # Get n-grams
        teacher_ngrams = self._get_ngrams(teacher_tokens, n)
        expert_ngrams = self._get_ngrams(expert_tokens, n)

        if not teacher_ngrams or not expert_ngrams:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # Count n-grams (using Counter for multiset intersection)
        teacher_counter = Counter(teacher_ngrams)
        expert_counter = Counter(expert_ngrams)

        # Calculate intersection (minimum count for each n-gram)
        overlap = sum((teacher_counter & expert_counter).values())

        # Calculate precision and recall
        precision = overlap / len(teacher_ngrams) if teacher_ngrams else 0.0
        recall = overlap / len(expert_ngrams) if expert_ngrams else 0.0

        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _calculate_all_ngram_overlaps(self, teacher_thinking: str, expert_thinking: str) -> Dict[str, Any]:
        """Calculate n-gram overlaps for n=3,4."""
        results = {}

        for n in [3, 4]:
            overlap_scores = self._calculate_ngram_overlap(teacher_thinking, expert_thinking, n)
            results[f'{n}gram'] = overlap_scores

        # Calculate average F1 across all n-grams
        avg_f1 = np.mean([results[f'{n}gram']['f1'] for n in [3, 4]])
        results['avg_f1'] = avg_f1

        return results

    def _log_ngram_summary(self, teacher_responses: List[Dict[str, Any]]):
        """Log summary statistics for n-gram overlap across all responses."""
        # Collect all n-gram overlap results
        all_ngram_results = []

        for response in teacher_responses:
            for k_response in response.get('k_responses', []):
                if 'ngram_overlap' in k_response:
                    all_ngram_results.append(k_response['ngram_overlap'])

        if not all_ngram_results:
            logger.warning("No n-gram overlap results found")
            return

        # Calculate average metrics across all responses
        avg_metrics = {
            '3gram_f1': np.mean([r['3gram']['f1'] for r in all_ngram_results]),
            '4gram_f1': np.mean([r['4gram']['f1'] for r in all_ngram_results]),
            'avg_f1': np.mean([r['avg_f1'] for r in all_ngram_results])
        }

        logger.info(f"\n{'='*60}")
        logger.info("N-GRAM OVERLAP WITH EXPERT TRACES")
        logger.info(f"{'='*60}")
        logger.info(f"Responses with expert matches: {len(all_ngram_results)}/{len(teacher_responses) * self.k}")
        logger.info(f"3-gram F1: {avg_metrics['3gram_f1']:.3f}")
        logger.info(f"4-gram F1: {avg_metrics['4gram_f1']:.3f}")
        logger.info(f"Average F1: {avg_metrics['avg_f1']:.3f}")
        logger.info(f"{'='*60}\n")
    
    def _calculate_perplexity(self, logprobs, token_ids, is_expert, len_prompt_without_thinking = None) -> float:
        """Calculate perplexity from logprobs."""
        neg_log_probs = []
        debug_tokens = []
        debug_logprobs_list = []

        for token_idx, token_logprob in enumerate(logprobs):
            if not token_logprob or token_idx >= len(token_ids):
                continue

            selected_token_id = token_ids[token_idx]

            if selected_token_id in token_logprob:
                logprob_obj = token_logprob[selected_token_id]
                selected_token = getattr(logprob_obj, 'decoded_token', str(selected_token_id))

                if is_expert:
                    if token_idx < len_prompt_without_thinking:
                        continue
                if '</' in selected_token:
                    break
                neg_log_probs.append(-logprob_obj.logprob)
                debug_tokens.append(selected_token)
                debug_logprobs_list.append(logprob_obj.logprob)

        if neg_log_probs:
            avg_neg_log_prob = sum(neg_log_probs) / len(neg_log_probs)
            perplexity = torch.exp(torch.tensor(avg_neg_log_prob)).item()

            # DEBUG: Log details for first few samples
            if is_expert and len(debug_tokens) > 0 and np.random.random() < 0.02:  # 2% sampling
                logger.warning(f"[PERPLEXITY DEBUG] Answer tokens ({len(debug_tokens)}): {debug_tokens}")
                logger.warning(f"[PERPLEXITY DEBUG] Logprobs: {[f'{lp:.6f}' for lp in debug_logprobs_list[:10]]}")
                logger.warning(f"[PERPLEXITY DEBUG] Neg logprobs: {[f'{-lp:.6f}' for lp in debug_logprobs_list[:10]]}")
                logger.warning(f"[PERPLEXITY DEBUG] Avg neg logprob: {avg_neg_log_prob:.6f}, Perplexity: {perplexity:.6f}")

            return perplexity

        return 0.0

    def _calculate_question_perplexity(self, logprobs, token_ids) -> float:
        """Calculate perplexity of the question (before <think> tag).

        This measures how "difficult" or "surprising" the question is to the model.
        """
        neg_log_probs = []

        for token_idx, token_logprob in enumerate(logprobs):
            if not token_logprob or token_idx >= len(token_ids):
                continue

            selected_token_id = token_ids[token_idx]

            if selected_token_id in token_logprob:
                logprob_obj = token_logprob[selected_token_id]
                selected_token = getattr(logprob_obj, 'decoded_token', str(selected_token_id))

                # Stop when we reach <think> tag
                if '<think>' in selected_token:
                    break

                neg_log_probs.append(-logprob_obj.logprob)

        if neg_log_probs:
            avg_neg_log_prob = sum(neg_log_probs) / len(neg_log_probs)
            return torch.exp(torch.tensor(avg_neg_log_prob)).item()

        return 0.0

    def _extract_logprobs(self, logprobs, token_ids, is_expert: bool = False, len_prompt_without_thinking = None) -> List[Dict[str, Any]]:
        """Extract token or prompt logprobs in the legacy-compatible format."""
        token_logprob_details: List[Dict[str, Any]] = []
        
        for token_idx, token_logprob in enumerate(logprobs):
            if not token_logprob:
                continue

            if token_ids and token_idx < len(token_ids):
                selected_token_id = token_ids[token_idx]

                if selected_token_id in token_logprob:
                    logprob_obj = token_logprob[selected_token_id]
                    selected_token = getattr(logprob_obj, 'decoded_token', str(selected_token_id))
                    selected_logprob = logprob_obj.logprob
                    selected_rank = getattr(logprob_obj, 'rank', None)

                    neg_log_prob = -selected_logprob

                    if is_expert:
                        if token_idx < len_prompt_without_thinking:
                                continue
                    if '</' in selected_token:
                        break
                    
                    token_detail = {
                        'token_idx': token_idx,
                        'selected_token': selected_token,
                        'neg_log_prob': neg_log_prob,
                        'rank': selected_rank,
                    }
                    token_logprob_details.append(token_detail)

        return token_logprob_details
    
    def _process_responses(self, teacher_data: List[Dict[str, Any]], all_responses: List[List[Dict[str, Any]]],
                          prompts: List[str], model_manager=None, openai_client=None, is_perturbation: bool = False) -> List[Dict[str, Any]]:
        """Process generated responses into final format. Now handles k responses per question."""
        teacher_responses = []
        total_best_score = 0.0
        total_mean_score = 0.0

        # Track the highest scoring response across all questions and k
        highest_score = -1.0
        highest_score_instance = None

        for i, (item, k_responses_list, prompt) in enumerate(zip(teacher_data, all_responses, prompts)):
            # Process all k responses for this question
            k_processed_responses = []
            k_reward_scores = []

            for k_idx, response_dict in enumerate(k_responses_list):
                if openai_client:  # OpenAI API response
                    response_text = response_dict.get('text', '')
                    generation_info = {
                        'finish_reason': response_dict.get('finish_reason', 'stop'),
                        'num_tokens': response_dict.get('num_tokens', 0),
                    }
                else:  # Local model response
                    response_text = response_dict.get('text', '')
                    generation_info = response_dict.get('generation_info', {})

                # Extract thinking and answer
                thinking, answer = self._extract_thinking_and_answer(response_text, item, is_perturbation)
                answer_in_thinking = answer in thinking

                # Calculate reward
                reward_score = self._calculate_reward(answer, item, i)
                k_reward_scores.append(reward_score)

                # Track highest scoring response
                if reward_score > highest_score:
                    highest_score = reward_score
                    highest_score_instance = {
                        'question_index': item['index'],
                        'k_idx': k_idx,
                        'question': item.get('question', prompt),
                        'gold_answer': item['answer'],
                        'teacher_answer': answer,
                        'teacher_thinking': thinking,
                        'reward_score': reward_score
                    }

                # Calculate n-gram overlap with expert if requested
                ngram_overlap_results = None
                if 'ngram_w_expert' in self.requested_accuracy_metrics and self.expert_traces:
                    # Try to find matching expert trace by question
                    question = item.get('question', prompt)
                    expert_thinking = self.expert_traces.get(question, None)

                    if expert_thinking:
                        ngram_overlap_results = self._calculate_all_ngram_overlaps(thinking, expert_thinking)
                    else:
                        logger.debug(f"No expert trace found for question index {item['index']}")

                # Create response entry for this k
                single_response = {
                    'k_idx': k_idx,
                    'teacher_response': response_text,
                    'teacher_thinking': thinking,
                    'teacher_answer': answer,
                    'reward_score': reward_score,
                    "answer_in_thinking": answer_in_thinking,
                    'generation_info': generation_info
                }

                # Add n-gram overlap results if available
                if ngram_overlap_results:
                    single_response['ngram_overlap'] = ngram_overlap_results

                # Add perturbation-specific fields
                if is_perturbation:
                    single_response.update({
                        'perturbed_output': response_text,
                        'perturbed_reward': reward_score,
                    })
                    if item.get('perturbation_type', 'unknown') == "expert_thinking":
                        single_response.update({
                            "expert_perplexity": generation_info.get('expert_perplexity', 0.0),
                            "expert_token_logprobs": generation_info.get('expert_token_logprobs', [])
                        })

                k_processed_responses.append(single_response)

            # Calculate metrics for k responses
            max_reward = max(k_reward_scores) if k_reward_scores else 0.0
            mean_reward = sum(k_reward_scores) / len(k_reward_scores) if k_reward_scores else 0.0
            total_best_score += max_reward
            total_mean_score += mean_reward

            # Create main response entry with all k responses
            teacher_response = {
                'index': item['index'],
                'question': item.get('question', prompt),
                'gold_answer': item['answer'],
                'k': self.k,
                'k_responses': k_processed_responses,  # List of all k responses
                'best_reward_score': max_reward,  # Best reward among k responses (Best@k)
                'mean_reward_score': mean_reward,  # Mean reward across k responses (Mean@k)
                'checkpoint_name': self._get_checkpoint_name(model_manager, openai_client),
                'metadata': item.get('metadata', {}),
                'data_source': item.get('data_source', ''),
                'seed': item.get('seed', 42),
                'full_prompt': prompt,
                'chat_template_applied': True,  # Always true for our pipeline
            }

            # Add perturbation-specific fields at the top level
            if is_perturbation:
                teacher_response.update({
                    'perturbation_type': item.get('perturbation_type', 'unknown'),
                    'original_thinking': item.get('original_thinking', ''),
                    'original_question': item.get('original_question', ''),
                    'original_response': item.get('original_response', ''),
                    'original_reward': item.get('original_reward', 0.0),
                    'original_answer': item.get('original_answer', ''),
                    'perturbed_input': item.get('question', prompt),
                    'is_perturbation': True,
                })

            teacher_responses.append(teacher_response)

        # Calculate summary stats
        best_at_k = total_best_score / len(teacher_responses) if teacher_responses else 0.0
        mean_at_k = total_mean_score / len(teacher_responses) if teacher_responses else 0.0

        logger.info(f"Generated {len(teacher_responses)} questions with k={self.k} responses each")
        logger.info(f"Best@{self.k}: {best_at_k:.3f}, Mean@{self.k}: {mean_at_k:.3f}")

        # Calculate and log n-gram overlap summary if available
        if 'ngram_w_expert' in self.requested_accuracy_metrics:
            self._log_ngram_summary(teacher_responses)

        # Log highest scoring instance
        if highest_score_instance:
            logger.info(f"\n{'='*60}")
            logger.info(f"HIGHEST SCORING RESPONSE (Score: {highest_score:.3f})")
            logger.info(f"{'='*60}")
            logger.info(f"Question Index: {highest_score_instance['question_index']}, k_idx: {highest_score_instance['k_idx']}")
            logger.info(f"Question: {highest_score_instance['question'][:200]}...")
            logger.info(f"Gold Answer: {highest_score_instance['gold_answer']}")
            logger.info(f"Teacher Answer: {highest_score_instance['teacher_answer']}")
            logger.info(f"Teacher Thinking: {highest_score_instance['teacher_thinking'][:300]}...")
            logger.info(f"{'='*60}\n")

        for response in teacher_responses:
            response.update({
                'checkpoint_best_at_k': best_at_k,
                'checkpoint_mean_at_k': mean_at_k,
                'checkpoint_total_best_score': total_best_score,
                'checkpoint_total_mean_score': total_mean_score,
                'checkpoint_total_count': len(teacher_responses)
            })

        return teacher_responses
    
    def _extract_thinking_and_answer(self, response_text: str, item: Dict[str, Any], is_perturbation: bool) -> tuple:
        """Extract thinking trace and answer from response."""
        if is_perturbation:
            # For perturbation, use original thinking and extract answer from response
            thinking = item.get('original_thinking', item.get('teacher_thinking', ''))

            # DEBUG: Log perturbation extraction
            if not response_text or not response_text.strip():
                logger.warning(f"[PERTURBATION DEBUG] Empty response_text for perturbation type: {item.get('perturbation_type', 'unknown')}")
                logger.warning(f"[PERTURBATION DEBUG] Item keys: {list(item.keys())}")

            response_text = "<answer>" + response_text
            answer = extract_answer(response_text, "answer") or response_text

            # DEBUG: Log extraction result
            if item.get('perturbation_type') == 'remove_thinking' and len(answer) < 5:
                logger.warning(f"[REMOVE_THINKING DEBUG] Short answer extracted: '{answer}'")
                logger.warning(f"[REMOVE_THINKING DEBUG] Response text: '{response_text[:200]}'")
        else:
            # Normal extraction
            preappend_token = self.teacher_config.get('preappend_token', None)
            if preappend_token:
                response_text = preappend_token + response_text
            
            extracted = extract_answer_and_think(response_text)
            if extracted:
                thinking, answer = extracted
            else:
                answer = extract_answer(response_text, "answer") or response_text
                thinking = extract_answer(response_text, "think") or response_text
        
        return thinking, answer
    
    def get_checkpoint_accuracy(self, checkpoint_name: str) -> Optional[float]:
        """
        Get accuracy for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            
        Returns:
            Accuracy value or None if not found
        """
        result = self.get_checkpoint_result(checkpoint_name)
        return result['accuracy'] if result else None
    
    def get_all_accuracies(self) -> Dict[str, float]:
        """
        Get accuracies for all checkpoints.
        
        Returns:
            Dictionary mapping checkpoint names to accuracies
        """
        return {name: results['accuracy'] for name, results in self.checkpoint_results.items()}
    
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
        
        logger.info(f"Added accuracy results for checkpoint {checkpoint_name}")
    
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
        """
        Calculate summary statistics across all checkpoints.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.checkpoint_results:
            return {}
        
        accuracies = [results['accuracy'] for results in self.checkpoint_results.values()]
        
        summary = {
            'num_checkpoints': len(self.checkpoint_results),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'median_accuracy': np.median(accuracies),
            'accuracy_range': np.max(accuracies) - np.min(accuracies)
        }
        
        # Find best and worst checkpoints
        best_checkpoint = max(self.checkpoint_results.items(), key=lambda x: x[1]['accuracy'])
        worst_checkpoint = min(self.checkpoint_results.items(), key=lambda x: x[1]['accuracy'])
        
        summary['best_checkpoint'] = {
            'name': best_checkpoint[0],
            'accuracy': best_checkpoint[1]['accuracy']
        }
        
        summary['worst_checkpoint'] = {
            'name': worst_checkpoint[0],
            'accuracy': worst_checkpoint[1]['accuracy']
        }
        
        return summary
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze performance trends across checkpoints.
        
        Returns:
            Dictionary with trend analysis
        """
        if len(self.checkpoint_results) < 2:
            return {'trend': 'insufficient_data'}
        
        sorted_results = self.get_sorted_results()
        accuracies = [results['accuracy'] for _, results in sorted_results]
        
        # Simple trend analysis
        if len(accuracies) >= 3:
            # Calculate linear trend
            x = np.arange(len(accuracies))
            coeffs = np.polyfit(x, accuracies, 1)
            trend_slope = coeffs[0]
            
            if trend_slope > 0.01:
                trend = 'improving'
            elif trend_slope < -0.01:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            # Simple comparison for 2 checkpoints
            if accuracies[-1] > accuracies[0]:
                trend = 'improving'
            elif accuracies[-1] < accuracies[0]:
                trend = 'declining'
            else:
                trend = 'stable'
        
        # Calculate improvement metrics
        first_accuracy = accuracies[0]
        last_accuracy = accuracies[-1]
        absolute_improvement = last_accuracy - first_accuracy
        relative_improvement = (absolute_improvement / first_accuracy) * 100 if first_accuracy > 0 else 0
        
        analysis = {
            'trend': trend,
            'first_accuracy': first_accuracy,
            'last_accuracy': last_accuracy,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement,
            'num_checkpoints_analyzed': len(accuracies)
        }
        
        if len(accuracies) >= 3:
            analysis['trend_slope'] = trend_slope
        
        return analysis
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of all results.
        
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {
            'metric_name': 'teacher_accuracy',
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
        
        logger.info(f"Saved accuracy metrics to {output_path}")
    
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
            logger.info(f"Loaded accuracy metrics for {len(self.checkpoint_results)} checkpoints")
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
        """
        Print a summary of the metrics.
        """
        if not self.checkpoint_results:
            print("No checkpoint results available")
            return
        
        summary = self.calculate_summary_statistics()
        trends = self.analyze_performance_trends()
        
        print("\n" + "="*60)
        print("TEACHER REWARD EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Number of checkpoints evaluated: {summary['num_checkpoints']}")
        print(f"Mean reward: {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")
        print(f"Reward range: {summary['min_accuracy']:.3f} - {summary['max_accuracy']:.3f}")
        print(f"Best checkpoint: {summary['best_checkpoint']['name']} ({summary['best_checkpoint']['accuracy']:.3f})")
        print(f"Worst checkpoint: {summary['worst_checkpoint']['name']} ({summary['worst_checkpoint']['accuracy']:.3f})")
        
        print("="*60)


def create_teacher_accuracy_metrics() -> AccuracyMetric:
    """
    Factory function to create an AccuracyMetric instance.
    
    Returns:
        AccuracyMetric instance
    """
    # This will need to be updated to pass proper config, reward_calculator, and prompt_manager
    # when called from actual code
    return AccuracyMetric({}, None, None)