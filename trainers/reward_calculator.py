from typing import List, Dict, Any, Union
from data.tasks import get_task_handler
from data.utils import match_format, extract_answer, extract_gold_answer_gsm8k, extract_gold_answer_math_hard, match_answer_format, extract_answer_and_think, extract_answer_and_think_auto
import reasoning_gym
import asyncio
import re
import json
import os
import math
import torch
import numpy as np
from evaluate_efficient.utils.openai_client import OpenAIClient
from evaluate_efficient.utils.prompt_manager import PromptManager
from evaluate_efficient.metrics.usefulness_metric import UsefulnessMetric
from evaluate_efficient.utils.answer_removal import process_batch_remove_answers
import time
from verl import DataProto


class RewardCalculator:
    """Calculate rewards based on task-specific scoring functions."""
    
    def __init__(self, task, task_type: Union[str, List[str]], reward_partial: bool = False, replacement_data_path: str = None, model_manager=None, model_name: str = None):
        # Handle both single task type and list of task types
        if isinstance(task_type, str):
            self.task_types = [task_type]
        else:
            self.task_types = task_type

        self.task = task
        self.reward_partial = reward_partial
        self.model_manager = model_manager  # For logprob-based rewards like cot_importance
        self.model_name = model_name  # For auto-detecting extraction format (e.g., Qwen4B)

        if task == "reasoning_gym":
            # Create a mapping from task type to task handler
            self.task_handlers = {}
            for task_ in self.task_types:
                self.task_handlers[task_] = reasoning_gym.create_dataset(task_)
        elif task == "gsm8k":
            # GSM8K doesn't need task handlers - simple comparison
            self.task_handlers = None

        # Initialize for informativeness evaluation
        self.prompt_manager = PromptManager(
            teacher_config={},
            developer_prompt=""
        )
        self.openai_client = OpenAIClient(model_name="gpt-4o-mini")

        # Initialize UsefulnessMetric for cot_importance calculation
        # Provide minimal config required by BaseMetric
        self.usefulness_metric = UsefulnessMetric(
            config={
                'evaluation': {
                    'cot_importance_strategy': 'percentage',
                    'teacher_model': {},  # Required by BaseMetric but not used by _evaluate_cot_importance
                    'batch_size': 8,  # Required by BaseMetric but not used by _evaluate_cot_importance
                    'metrics': ['cot_importance']  # Required by UsefulnessMetric._init_metric_config
                }
            },
            reward_calculator=None,  # Not used by _evaluate_cot_importance
            prompt_manager=self.prompt_manager
        )

        # Initialize replacement thinking traces for replace mode
        self.replacement_data_path = replacement_data_path
        self.replacement_thinking_map = {}
        if replacement_data_path:
            self._load_replacement_thinking_traces()

    def _js_bernoulli(self, p: float, q: float, eps: float = 1e-12) -> float:
        """
        Compute Jensen-Shannon divergence for Bernoulli distributions.

        Args:
            p: Probability for first distribution
            q: Probability for second distribution
            eps: Epsilon to avoid log(0)

        Returns:
            JS divergence value (symmetric, bounded [0, log(2)])
        """
        # Clip probabilities to avoid numerical issues
        p = min(max(p, eps), 1.0 - eps)
        q = min(max(q, eps), 1.0 - eps)

        # Compute mixture distribution
        m = 0.5 * (p + q)

        # KL(p || m)
        kl_p_m = p * math.log(p / m) + (1 - p) * math.log((1 - p) / (1 - m))

        # KL(q || m)
        kl_q_m = q * math.log(q / m) + (1 - q) * math.log((1 - q) / (1 - m))

        # JS divergence
        return 0.5 * kl_p_m + 0.5 * kl_q_m

    def calculate_reward(self, prediction: str, entry: Dict[str, Any], preappend_token: str = "", reward_type: str = "rule_based") -> float:
        """
        Calculate reward for a prediction using different scoring methods.

        Args:
            prediction: Model's prediction
            entry: Original data entry containing ground truth and context
            preappend_token: Token to prepend for format matching
            reward_type: Type of reward calculation ("rule_based", "informativeness", "format")

        Returns:
            Reward value (0.0 to 1.0)
        """

        if reward_type == "format":
            return self._calculate_format_reward(prediction, entry, preappend_token)
        else:
            return self._calculate_rule_based_reward(prediction, entry, preappend_token)
    
    def _calculate_rule_based_reward(self, prediction: str, entry: Dict[str, Any], preappend_token: str = "") -> float:
        """Rule-based reward calculation without format bonus."""
        if self.task == "gsm8k":
            # Simple comparison for GSM8K
            ground_truth = entry.get('answer', '')

            # Extract numerical answer from prediction if needed
            prediction_clean = extract_answer(prediction) if prediction else ""
            ground_truth_clean = extract_gold_answer_gsm8k(ground_truth) if ground_truth else ground_truth

            # Compare answers (case-insensitive, strip whitespace)
            try:
                if prediction_clean.strip().lower() == ground_truth_clean.strip().lower():
                    reward = 1.0
                else:
                    reward = 0.0
            except:
                print(f"prediction_clean={prediction_clean}, ground_truth_clean={ground_truth_clean}")
                reward = 0.0

            return reward

        elif self.task == "math_hard":
            ground_truth = entry.get('solution', entry.get('answer', ''))
            prediction_clean = (extract_answer(prediction) or prediction).strip() if prediction else ""
            ground_truth_clean = extract_gold_answer_math_hard(ground_truth) if ground_truth else ground_truth
            print(f"[math_hard reward] pred_extracted='{prediction_clean}' | gold_raw='{str(ground_truth)[:80]}' | gold_extracted='{ground_truth_clean}'")
            try:
                return 1.0 if prediction_clean.strip().lower() == ground_truth_clean.strip().lower() else 0.0
            except:
                return 0.0

        elif self.task == "reasoning_gym":
            task_handler = self.task_handlers[entry['metadata']['source_dataset']]
            prediction_clean = extract_answer(preappend_token + prediction)
            if self.reward_partial:
                return task_handler.score_answer_partial(prediction_clean, entry)
            else:
                return task_handler.score_answer(prediction_clean, entry)

        else:
            raise ValueError(f"Unknown task type: {self.task}")

    def _calculate_format_reward(self, prediction: str, entry: Dict[str, Any], preappend_token: str = "") -> float:
        """Calculate format matching reward."""
        if self.task == "gsm8k":
            # Format matching for GSM8K
            if match_format(prediction, preappend_token=preappend_token):
                return 0.2
            else:
                return 0.0

        elif self.task == "math_hard":
            return 0.2 if match_format(prediction, preappend_token=preappend_token) else 0.0

        elif self.task == "reasoning_gym":
            # Format matching for reasoning gym
            if "answer" in preappend_token:
                return match_answer_format(prediction, preappend_token=preappend_token)
            else:
                return match_format(prediction, preappend_token=preappend_token)

        else:
            raise ValueError(f"Unknown task type: {self.task}")

    def calculate_score(self, prediction: str, entry: Dict[str, Any], preappend_token: str = "") -> float:
        if self.task == "gsm8k":
            # Simple comparison for GSM8K
            ground_truth = entry.get('answer', '')

            prediction_clean = extract_answer(prediction) if prediction else ""
            ground_truth_clean = extract_gold_answer_gsm8k(ground_truth) if ground_truth else ground_truth

            try:
                if prediction_clean.strip().lower() == ground_truth_clean.strip().lower():
                    reward = 1.0
                else:
                    reward = 0.0
            except:
                print(f"prediction_clean={prediction_clean}, ground_truth_clean={ground_truth_clean}")
                reward = 0.0        
            return reward 
        
        elif self.task == "math_hard":
            ground_truth = entry.get('solution', entry.get('answer', ''))
            prediction_clean = (extract_answer(prediction) or prediction).strip() if prediction else ""
            ground_truth_clean = extract_gold_answer_math_hard(ground_truth) if ground_truth else ground_truth
            try:
                return 1.0 if prediction_clean.strip().lower() == ground_truth_clean.strip().lower() else 0.0
            except:
                return 0.0

        elif self.task == "reasoning_gym":
            task_handler = self.task_handlers[entry['metadata']['source_dataset']]
            if self.reward_partial:
                return task_handler.score_answer_partial(prediction, entry)
            else:
                return task_handler.score_answer(prediction, entry)

        else:
            raise ValueError(f"Unknown task type: {self.task}")
        

    def _batch_remove_answer_and_question_from_thinking(
        self,
        thinking_list: List[str],
        teacher_answers: List[str],
        questions: List[str],
    ) -> List[str]:
        """
        Batch process to remove both answers and questions from thinking traces using GPT-4o-mini.

        Args:
            thinking_list: List of thinking trace texts
            teacher_answers: List of answers to remove
            questions: List of questions to remove

        Returns:
            List of cleaned thinking traces with both answers and questions removed
        """
        if not thinking_list:
            return []

        try:
            # Create prompts for batch processing using prompt_manager
            removal_prompts = []
            for thinking, answer, question in zip(thinking_list, teacher_answers, questions):
                if not thinking:
                    removal_prompts.append("")
                    continue

                # Use prompt_manager to create removal prompt
                prompt = self.prompt_manager.create_answer_and_question_removal_prompt(
                    thinking, answer, question
                )
                removal_prompts.append(prompt)

            # Batch process all prompts asynchronously
            responses = asyncio.run(self.openai_client.generate_individual_async(removal_prompts))

            # Extract cleaned thinking traces
            cleaned_thinking_list = []
            for idx, (response, original_thinking) in enumerate(zip(responses, thinking_list)):
                if response and response.get('text'):
                    cleaned = response.get('text', '').strip()
                    cleaned_thinking_list.append(cleaned if cleaned else original_thinking)
                else:
                    cleaned_thinking_list.append(original_thinking)

            return cleaned_thinking_list

        except Exception as e:
            print(f"Error in batch answer and question removal: {e}")
            import traceback
            traceback.print_exc()
            # Return original thinking traces on error
            return thinking_list
    

    def calculate_batch_cot_verifier_rewards(self, predictions: List[str], entries: List[Dict[str, Any]], preappend_token: str = "") -> List[float]:
        """
        Calculate CoT verifier accuracy rewards by comparing model performance with vs without question.

        Reward Logic:
        - 1.0 if model produces the SAME answer both with and without the question (consistency)
        - 0.0 otherwise

        This measures if the reasoning trace alone is sufficient to produce consistent answers,
        demonstrating it's truly informative independent of the question.
        """
        if not predictions or not entries:
            return []

        try:
            # Step 1: Extract all thinking traces, answers, and questions
            thinking_list = []
            teacher_answers = []
            questions = []
            tasks = []
        

            for idx, (prediction, entry) in enumerate(zip(predictions, entries)):
                # Extract thinking and answer from prediction
                result = extract_answer_and_think_auto(preappend_token + prediction, model_name=self.model_name)
                if result:
                    thinking, teacher_answer = result
                else:
                    thinking = preappend_token + prediction
                    teacher_answer = ""

                
                # Get question from entry
                question = entry.get('input_str')
                #question = entry.get('metadata', {}).get('input_str', entry.get('question', ''))
                task = entry.get('metadata', {}).get('source_dataset', entry.get('task', 'reasoning_gym'))

                thinking_list.append(thinking)
                teacher_answers.append(teacher_answer)
                questions.append(question)
                tasks.append(task)

            # Step 2: Batch remove both answers and questions using GPT-4o-mini
            time1 = time.time()
            print(f"\n[COT_VERIFIER] Starting batch removal for {len(thinking_list)} items...")
            cleaned_thinking_list = self._batch_remove_answer_and_question_from_thinking(
                thinking_list, teacher_answers, questions
            )
            print(f"[COT_VERIFIER] Batch removal completed!", time.time() - time1)

            # DEBUG: Print removal results for first few items
            for idx in range(min(2, len(thinking_list))):
                print(f"\n[COT_VERIFIER DEBUG {idx}] ANSWER AND QUESTION REMOVAL")
                print(f"-" * 80)
                print(f"Question: '{questions[idx]}'")
                print(f"Answer to remove: '{teacher_answers[idx]}'")
                print(f"Thinking BEFORE removal: '{thinking_list[idx]}'")
                print(f"Thinking AFTER removal: '{cleaned_thinking_list[idx]}'")

            # Step 3: Create verifier prompts for all cleaned thinking traces
            cot_verifier_prompts = []
            prompt_metadata = []
            

            for idx, (thinking_clean, question, task) in enumerate(zip(cleaned_thinking_list, questions, tasks)):
                # Create TWO prompts:
                # 1. WITH question: answer | question, thinking
                prompt_with_question = self.prompt_manager.create_verifier_prompt_with_question(
                    question,
                    thinking_clean,
                    task=task
                )

                # 2. WITHOUT question: answer | thinking
                prompt_without_question = self.prompt_manager.create_verifier_prompt_without_question(
                    thinking_clean,
                    task=task
                )

                # DEBUG: Print prompts for first few items
                if idx < 0:
                    print(f"\n[COT_VERIFIER DEBUG {idx}] PROMPTS")
                    print(f"-" * 80)
                    print(f"\nPrompt WITH question:\n'{prompt_with_question[:-300]}'")
                    print(f"\nPrompt WITHOUT question:\n'{prompt_without_question[:-300]}'")

                cot_verifier_prompts.append(prompt_with_question)
                prompt_metadata.append({'entry_idx': idx, 'variant': 'with_question'})

                cot_verifier_prompts.append(prompt_without_question)
                prompt_metadata.append({'entry_idx': idx, 'variant': 'without_question'})

            # Batch evaluate all prompts asynchronously using OpenAI (gpt-4o-mini)
            time2 = time.time()
            print(f"START evaluating {len(cot_verifier_prompts)} prompts")
            responses = asyncio.run(self.openai_client.generate_individual_async(cot_verifier_prompts))
            print(f"FINISHED evaluating {len(cot_verifier_prompts)} prompts", time.time()- time2)

            # Process responses and compute rewards
            # Group by entry_idx
            entry_results = {}
            for response, metadata in zip(responses, prompt_metadata):
                entry_idx = metadata['entry_idx']
                variant = metadata['variant']

                # Extract answer from response
                response_text = response.get('text', '') if response else ''
                extracted_answer = extract_answer(response_text)

                # Normalize answer for comparison
                extracted_normalized = extracted_answer.replace('\n', '').replace(' ', '').strip() if extracted_answer else ''

                # DEBUG: Print GPT-4o-mini responses for first few items
                if entry_idx < 0:
                    print(f"\n[COT_VERIFIER DEBUG {entry_idx}] GPT-4o-mini RESPONSE ({variant})")
                    print(f"-" * 80)
                    print(f"Raw response: '{response_text}'")
                    print(f"Extracted answer: '{extracted_answer}'")
                    print(f"Normalized answer: '{extracted_normalized}'")

                # Store result
                if entry_idx not in entry_results:
                    entry_results[entry_idx] = {}
                entry_results[entry_idx][variant] = extracted_normalized


            # Calculate final rewards
            rewards = []
            for idx in range(len(predictions)):
                if idx in entry_results and 'with_question' in entry_results[idx] and 'without_question' in entry_results[idx]:
                    answer_with_q = entry_results[idx]['with_question']
                    answer_without_q = entry_results[idx]['without_question']
                    original_answer = teacher_answers[idx]
                    original_answer = original_answer.replace('\n', '').replace(' ', '').strip() if extracted_answer else ''

                    # Reward = 1.0 if answers match (consistency)
                    # This means the reasoning trace alone is sufficient
                    # Also check that answer is not "noanswerfound" (case-insensitive)
                    if (answer_with_q == answer_without_q and answer_with_q == original_answer and answer_without_q == original_answer and
                        answer_with_q != '' and
                        answer_with_q.lower() != 'noanswerfound' and
                        answer_without_q.lower() != 'noanswerfound'):
                        reward = 1.0
                    else:
                        reward = 0.0

                    # DEBUG: Print final reward calculation for first few items
                    if idx < 2:
                        print(f"\n[COT_VERIFIER DEBUG {idx}] FINAL REWARD CALCULATION")
                        print(f"=" * 80)
                        print(f"Answer WITH question:    '{answer_with_q}'")
                        print(f"Answer WITHOUT question: '{answer_without_q}'")
                        print(f"Original_answer: '{original_answer}'")
                        print(f"Answers match: {answer_with_q == answer_without_q}")
                        print(f"Both non-empty: {answer_with_q != ''}")
                        print(f"Not 'noanswerfound': {answer_with_q.lower() != 'noanswerfound' and answer_without_q.lower() != 'noanswerfound'}")
                        print(f"FINAL REWARD: {reward}")
                        print(f"=" * 80)
                else:
                    reward = 0.0
                    # DEBUG: Print missing data case
                    if idx < 0:
                        print(f"\n[COT_VERIFIER DEBUG {idx}] MISSING DATA - REWARD = 0.0")
                        print(f"=" * 80)

                rewards.append(reward)

            return rewards

        except Exception as e:
            print(f"Error in batch cot_verifier reward calculation: {e}")
            import traceback
            traceback.print_exc()
            return [0.0] * len(predictions)


    def calculate_batch_quality_rewards(self, predictions: List[str], entries: List[Dict[str, Any]], preappend_token: str = "", examples: List[Dict[str, Any]] = None) -> List[float]:
        """Calculate quality rewards for a batch of predictions using async OpenAI evaluation."""
        if not predictions or not entries:
            return []

        try:
            # Prepare all prompts for batch processing
            quality_prompts = []
            teacher_answers = []

            for prediction, entry in zip(predictions, entries):
                # Extract thinking and answer from prediction (with preappend token like rule-based)
                result = extract_answer_and_think_auto(preappend_token + prediction, model_name=self.model_name)
                if result:
                    thinking, teacher_answer = result  # extract_answer_and_think returns (answer, think)
                else:
                    thinking = preappend_token + prediction
                    teacher_answer = ""

                # Remove answer from thinking with improved logic
                #thinking_without_answer = self._remove_answer_from_thinking(thinking, teacher_answer)

                # Create prompt using prompt manager
                examples  = json.load(open(os.path.join(os.path.dirname(__file__), 'count_bits.json')))

                prompt = self.prompt_manager.quality_check_prompt(
                    thinking,
                    question=entry.get('question'),
                    examples=examples,
                    task=entry.get('metadata', {}).get('source_dataset', entry.get('task', 'reasoning_gym'))
                )
                quality_prompts.append(prompt)
                teacher_answers.append(teacher_answer)


            # Batch evaluate all prompts asynchronously
            responses = asyncio.run(self.openai_client.generate_individual_async(quality_prompts))

            # Calculate rewards
            rewards = []
            for idx, response in enumerate(responses):
                response_text = response.get('text', '') if response else ''
                try:
                    extracted_reward = float(response_text)
                except:
                    print(f"Error in batch quality reward calculation: {response_text}")
                    extracted_reward = 0.0
                rewards.append(extracted_reward)

            return rewards

        except Exception as e:
            print(f"Error in batch quality reward calculation: {e}")
            return [0.0] * len(predictions)
        
    def calculate_batch_rewards(self, predictions: List[str], ground_truths: List[str]) -> List[float]:
        """
        Calculate rewards for a batch of predictions using task-specific scoring.
        
        Args:
            predictions: List of model predictions (strings)
            ground_truths: List of ground truth entries (dicts for reasoning_gym, strings for gsm8k)
                
        Returns:
            List of reward values (floats between 0.0 and 1.0), one for each prediction
        """
        rewards = []
       
        for i, pred in enumerate(predictions):         
            reward = self.calculate_reward(pred, ground_truths[i])
            rewards.append(reward)
        return rewards

    def get_replacement_entry_for_index(self, original_entry: Dict[str, Any], replacement_tracking: Dict[int, int] = None) -> Dict[str, Any]:
        """
        Get a perturbed entry with replaced thinking trace for a given index.

        Args:
            original_entry: The original data entry
            replacement_tracking: Dictionary mapping training_index -> replacement_index

        Returns:
            A new entry with the thinking trace replaced, or original entry if no replacement found
        """
        training_index = original_entry.get('index')
        if training_index is None:
            return original_entry

        # Check if we have tracking info for which replacement was used
        if replacement_tracking and training_index in replacement_tracking:
            replacement_index = replacement_tracking[training_index]
            if replacement_index in self.replacement_thinking_map:
                # Create a copy of the original entry
                perturbed_entry = original_entry.copy()
                replacement_data = self.replacement_thinking_map[replacement_index]

                # Replace the answer with the replacement answer (this becomes the new ground truth)
                if 'answer' in perturbed_entry:
                    perturbed_entry['answer'] = replacement_data['gold_answer']

                # Add replacement thinking if needed (for reference)
                perturbed_entry['replacement_thinking'] = replacement_data['thinking']

                return perturbed_entry

        return original_entry


    def calculate_batch_cot_importance_rewards_trainer(
        self,
        predictions: List[str],
        entries: List[Dict[str, Any]],
        preappend_token: str,
        actor_rollout_wg,
        tokenizer
    ) -> List[float]:
        """
        Calculate CoT importance rewards during training using actor_rollout_wg.compute_log_prob().

        Uses hardcoded truncation levels: [0, 30, 60] percentages.
        Computes JS divergence between full thinking and truncated thinking.

        Args:
            predictions: List of model predictions with thinking traces
            entries: List of ground truth entries
            preappend_token: Token prepended to predictions (e.g., "<think>")
            actor_rollout_wg: Actor rollout worker group for computing logprobs
            tokenizer: Tokenizer for tokenization

        Returns:
            List of rewards (average of JS divergences at 3 truncation levels)
        """
        try:
            print(f"\n[COT_IMPORTANCE_TRAINER] Starting calculation for {len(predictions)} predictions...")

            if not predictions or not entries:
                return []

            # Hardcoded truncation percentages
            truncation_percentages = [0, 20, 40, 60, 80]

            # Step 1: Extract thinking and answer from predictions
            print("[COT_IMPORTANCE_TRAINER] Step 1: Extracting thinking and answers...")
            thinking_list = []
            answer_list = []
            question_list = []

            for idx, (prediction, entry) in enumerate(zip(predictions, entries)):
                # Extract thinking and answer
                result = extract_answer_and_think_auto(preappend_token + prediction, model_name=self.model_name)
                if result:
                    thinking, answer = result
                    thinking = preappend_token + thinking
                else:
                    thinking = preappend_token + prediction
                    answer = extract_answer(preappend_token + prediction)

                if idx == 0:
                    print(thinking)
 
                # Get question from entry
                question = tokenizer.apply_chat_template(entry.get('raw_prompt', ''), tokenize=False, add_generation_prompt=True)

                thinking_list.append(thinking)
                answer_list.append(answer)
                question_list.append(question)

            # Step 2: Create sequences for all truncation levels
            print(f"[COT_IMPORTANCE_TRAINER] Step 2: Creating sequences for truncation levels {truncation_percentages}...")
            all_sequences = []  # Will contain (sequence, sample_idx, truncation_pct)

            for idx, (question, thinking, answer) in enumerate(zip(question_list, thinking_list, answer_list)):
                words = thinking.split()
                num_words = len(words)

                # Add full thinking (100%)
                full_sequence = f"{question}{thinking}</think>\n<answer>{answer}</answer>"
                all_sequences.append((full_sequence, idx, 100))

                # Add truncated versions
                for pct in truncation_percentages:
                    target_words = int(num_words * pct / 100.0)
                    truncated_thinking = ' '.join(words[:target_words])
                    truncated_sequence = f"{question}{truncated_thinking}</think>\n<answer>{answer}</answer>"
                    all_sequences.append((truncated_sequence, idx, pct))

            print(f"[COT_IMPORTANCE_TRAINER] Created {len(all_sequences)} sequences ({len(predictions)} samples × {len(truncation_percentages) + 1} levels)")

            # Step 3: Tokenize all sequences and split at <answer> tag
            print("[COT_IMPORTANCE_TRAINER] Step 3: Tokenizing sequences...")
            prompts_list = []
            responses_list = []

            for seq_idx, (full_sequence, sample_idx, truncation_pct) in enumerate(all_sequences):
                # Find the LAST <answer> tag position in text (in case it appears in system prompt)
                answer_tag = "<answer>"
                answer_start_pos = full_sequence.rfind(answer_tag)

                if answer_start_pos == -1:
                    print(f"[COT_IMPORTANCE_TRAINER WARNING] No <answer> tag found in sequence {seq_idx}")
                    continue

                # Split text at <answer> tag
                prompt_text = full_sequence[:answer_start_pos + len(answer_tag)]

                # IMPORTANT: Tokenize the FULL sequence first, then tokenize prompt
                # This ensures token boundaries are consistent and prevents misalignment
                full_ids = tokenizer.encode(full_sequence)
                prompt_ids = tokenizer.encode(prompt_text)

                # Response tokens = everything after prompt tokens in the full sequence
                response_ids = full_ids[len(prompt_ids):]

                prompts_list.append(torch.tensor(prompt_ids))
                responses_list.append(torch.tensor(response_ids))

                # Debug: Print first sequence tokenization details
                if seq_idx == 0:
                    print("\n[COT_IMPORTANCE_TRAINER DEBUG] First sequence tokenization:")
                    print(f"  Full sequence length: {len(full_ids)} tokens")
                    print(f"  Prompt length (up to and including <answer>): {len(prompt_ids)} tokens")
                    print(f"  Response length (answer content): {len(response_ids)} tokens")
                    print(f"  Sum check: {len(prompt_ids)} + {len(response_ids)} = {len(prompt_ids) + len(response_ids)} (should equal {len(full_ids)})")
                    print(f"  prompt: token {tokenizer.decode(prompt_ids)}")
                    print(f"  Response tokens decoded: {tokenizer.decode(response_ids)}")
                    print(f"  Response token IDs: {response_ids}")
                    print()

            if not prompts_list:
                print("[COT_IMPORTANCE_TRAINER ERROR] No valid sequences tokenized")
                return [0.0] * len(predictions)

            # Step 4: Create padded batches
            print("[COT_IMPORTANCE_TRAINER] Step 4: Creating padded DataProto batch...")

            # Concatenate prompt + response first, THEN pad
            # Padding strategy:
            #   - input_ids: [left_pad | prompt | response | response_right_pad]
            #     * left_pad = max_prompt_len - prompt_len (aligns prompts)
            #     * response_right_pad = max_response_len - response_len (matches responses_tensor)
            #   - responses_tensor: [response | response_right_pad]
            #     * response_right_pad = max_response_len - response_len
            #   - attention_mask: [0s (left_pad) | 1s (prompt+response) | 0s (response_right_pad)]
            # This ensures response padding matches between input_ids and responses_tensor
            input_ids_list = []
            prompt_length_list = []
            response_length_list = []

            for prompt, response in zip(prompts_list, responses_list):
                # Concatenate prompt and response
                full_seq = torch.cat([prompt, response])
                input_ids_list.append(full_seq)
                prompt_length_list.append(len(prompt))
                response_length_list.append(len(response))

            # Find max length for padding
            max_seq_len = max(seq.shape[0] for seq in input_ids_list)

            # Pad all sequences to max length
            padded_input_ids = []
            attention_masks = []
            position_ids_list = []
            padded_responses = []

            # Calculate max prompt and response lengths separately
            max_prompt_len = max(prompt_length_list)
            max_response_len = max(response_length_list)

            # Total target length: all sequences padded to same length
            # Structure: [left_pad | prompt | response | response_right_pad]
            # where response_right_pad matches responses_tensor padding
            total_target_len = max_prompt_len + max_response_len

            for seq, prompt_len, response_len in zip(input_ids_list, prompt_length_list, response_length_list):
                # Left pad to align prompts at the same ending position
                left_pad_len = max_prompt_len - prompt_len

                # Right pad to align responses (same as responses_tensor padding)
                response_right_pad_len = max_response_len - response_len

                # Verify total length
                actual_total = left_pad_len + prompt_len + response_len + response_right_pad_len
                assert actual_total == total_target_len, \
                    f"Padding mismatch: {left_pad_len} + {prompt_len} + {response_len} + {response_right_pad_len} = {actual_total} != {total_target_len}"

                # Pad input_ids: [left_pad | prompt | response | response_right_pad]
                # This ensures response padding matches responses_tensor
                padded_seq = torch.nn.functional.pad(
                    seq, (left_pad_len, response_right_pad_len), value=tokenizer.pad_token_id
                )
                padded_input_ids.append(padded_seq)

                # Create attention mask: [0s (left) | 1s (prompt+response) | 0s (right)]
                seq_len = prompt_len + response_len
                attention_mask = torch.cat([
                    torch.zeros(left_pad_len, dtype=torch.long),
                    torch.ones(seq_len, dtype=torch.long),
                    torch.zeros(response_right_pad_len, dtype=torch.long)
                ])
                attention_masks.append(attention_mask)

                # Create position IDs (0 for left pad, 0 to seq_len-1 for real tokens, 0 for right pad)
                position_ids = torch.cat([
                    torch.zeros(left_pad_len, dtype=torch.long),
                    torch.arange(seq_len, dtype=torch.long),
                    torch.zeros(response_right_pad_len, dtype=torch.long)
                ])
                position_ids_list.append(position_ids)

                # Create responses tensor (RIGHT PAD - padding on the right, real tokens on the left)
                # Extract response portion from the concatenated sequence
                response_only = seq[prompt_len:prompt_len + response_len]
                response_padded = torch.nn.functional.pad(
                    response_only, (0, max_response_len - response_len), value=tokenizer.pad_token_id
                )
                padded_responses.append(response_padded)

            # Stack into tensors
            input_ids_tensor = torch.stack(padded_input_ids)
            attention_mask_tensor = torch.stack(attention_masks)
            position_ids_tensor = torch.stack(position_ids_list)
            responses_tensor = torch.stack(padded_responses)

            print(f"[COT_IMPORTANCE_TRAINER] Batch shapes: input_ids={input_ids_tensor.shape}, responses={responses_tensor.shape}")
            print(f"[COT_IMPORTANCE_TRAINER] Prompt lengths: min={min(prompt_length_list)}, max={max(prompt_length_list)}")
            print(f"[COT_IMPORTANCE_TRAINER] Response lengths: min={min(response_length_list)}, max={max(response_length_list)}")

            # Create DataProto batch (no prompts needed - log_prob only returns response logprobs)
            batch_dict = {
                "input_ids": input_ids_tensor,
                "responses": responses_tensor,
                "attention_mask": attention_mask_tensor,
                "position_ids": position_ids_tensor,
            }      

            # Verification: Check if input_ids sequences are unique

            combined_batch = DataProto.from_single_dict(batch_dict)

            # Step 5: Compute logprobs using actor_rollout_wg
            print("[COT_IMPORTANCE_TRAINER] Step 5: Computing logprobs...")
            time_start = time.time()

            log_prob_output = actor_rollout_wg.compute_log_prob(combined_batch)
            log_probs = log_prob_output.batch["old_log_probs"]  # [num_sequences, response_length]
            #print(log_probs[0].tolist())

            print(f"[COT_IMPORTANCE_TRAINER] Logprob computation took {time.time() - time_start:.2f}s")
            print(f"[COT_IMPORTANCE_TRAINER] log_probs shape: {log_probs.shape}")

            # Debug: Print first response details with token-by-token logprobs
            print("\n[COT_IMPORTANCE_TRAINER DEBUG] First response verification:")
            first_prompt_len = prompt_length_list[0]
            first_response_len = response_length_list[0]
            first_response_ids = responses_list[0].tolist()
            first_response_text = tokenizer.decode(first_response_ids)
            first_response_logprobs = log_probs[0, :first_response_len].tolist()

            print(f"  Prompt length: {first_prompt_len} tokens")
            print(f"  Response text: {first_response_text}")
            print(f"  Response length (num tokens): {first_response_len}")
            print(f"  Log probs tensor shape: {log_probs.shape}")
            print(f"  Token-by-token breakdown:")
            for i, (token_id, logprob) in enumerate(zip(first_response_ids, first_response_logprobs)):
                token_text = tokenizer.decode([token_id])
                print(f"    Token {i}: '{token_text}' (ID={token_id}) -> logprob={logprob:.4f}")

            # Step 6: Extract sequence log probabilities
            print("\n[COT_IMPORTANCE_TRAINER] Step 6: Extracting sequence log probabilities...")

            # Group by original sample
            sample_logprobs = {}  # {sample_idx: {truncation_pct: sequence_logprob}}

            seq_idx = 0
            for full_sequence, sample_idx, truncation_pct in all_sequences:
                # Sum log probs for this sequence (only non-padding tokens)
                sequence_log_probs = log_probs[seq_idx]  # [max_response_length]

                # Use tracked response length (not padding)
                response_length = response_length_list[seq_idx]
                sequence_logprob = sequence_log_probs[:response_length].sum().item()

                # Store
                if sample_idx not in sample_logprobs:
                    sample_logprobs[sample_idx] = {}
                sample_logprobs[sample_idx][truncation_pct] = sequence_logprob

                seq_idx += 1

            # Step 7: Compute JS divergences and rewards
            print("[COT_IMPORTANCE_TRAINER] Step 7: Computing JS divergences...")
            rewards = []

            for idx in range(len(predictions)):
                if idx not in sample_logprobs:
                    print(f"[COT_IMPORTANCE_TRAINER WARNING] Sample {idx} not found in results")
                    rewards.append(0.0)
                    continue

                logprobs_dict = sample_logprobs[idx]

                # Get probabilities (convert from log space) - full sequence at 100%
                p_full = math.exp(logprobs_dict.get(100, -1000))

                # Get probabilities for each truncation percentage
                js_divergences = []
                truncated_probs = {}
                for pct in truncation_percentages:
                    p_trunc = math.exp(logprobs_dict.get(pct, -1000))
                    truncated_probs[pct] = p_trunc
                    js_div = self._js_bernoulli(p_full, p_trunc)
                    js_divergences.append(js_div)

                # Average JS divergences as reward
                reward = np.mean(js_divergences) if js_divergences else 0.0
                rewards.append(reward)

                if idx < 2:
                    print(f"\n[COT_IMPORTANCE_TRAINER DEBUG {idx}] REWARD CALCULATION")
                    print(f"=" * 80)
                    # Print log probs for all truncation levels (100% + truncation percentages)
                    logprob_strs = []
                    for pct in [100] + truncation_percentages:
                        if pct in logprobs_dict:
                            logprob_strs.append(f"{pct}%={logprobs_dict[pct]:.4f}")
                        else:
                            logprob_strs.append(f"{pct}%=N/A")
                    print(f"Sequence log probs: {', '.join(logprob_strs)}")

                    # Print probabilities
                    prob_strs = [f"p_full={p_full:.6e}"]
                    for pct in truncation_percentages:
                        prob_strs.append(f"p_{pct}={truncated_probs[pct]:.6e}")
                    print(f"Probabilities: {', '.join(prob_strs)}")

                    # Print JS divergences
                    js_strs = [f"js_{pct}={js_divergences[i]:.6f}" for i, pct in enumerate(truncation_percentages)]
                    print(f"JS divergences: {', '.join(js_strs)}")
                    print(f"Final reward (average): {reward:.6f}")
                    print(f"=" * 80)

            print(f"\n[COT_IMPORTANCE_TRAINER] Computed {len(rewards)} rewards")
            print(f"[COT_IMPORTANCE_TRAINER] Reward statistics: min={min(rewards):.4f}, max={max(rewards):.4f}, mean={np.mean(rewards):.4f}")

            return rewards

        except Exception as e:
            print(f"[ERROR] cot_importance_trainer computation failed: {e}")
            import traceback
            traceback.print_exc()
            return [0.0] * len(predictions)