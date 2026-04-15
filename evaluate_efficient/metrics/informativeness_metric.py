"""
Informativeness Metric

Handles informativeness evaluation of teacher reasoning traces by testing
how well they enable a model to solve new questions from the same task.
"""

import logging
import numpy as np
from typing import Dict, List, Any

from .base_metric import BaseMetric
from ..utils.openai_client import OpenAIClient
from ..utils.answer_removal import remove_answers_batch
from data.utils import extract_answer

logger = logging.getLogger(__name__)


class InformativenessMetric(BaseMetric):
    """Handles informativeness evaluation of teacher reasoning traces."""
    
    def _init_metric_config(self):
        """Initialize informativeness metric configuration."""
        requested_metrics = self.config.get('evaluation', {}).get('metrics', [])
        self.informativeness_types = ['whole_explanation_question_generalize_to_new_question', 'answer_generalize_to_new_question', 'teacher_student_accuracy_generalize', 'teacher_student_accuracy', 'answer_removed_explanation_only', 'answer_removed_explanation_generalize_to_new_question']
        self.requested_informativeness_metrics = [r for r in requested_metrics if r in self.informativeness_types]
    
    def can_run(self) -> bool:
        """Check if informativeness evaluation should be run."""
        return len(self.requested_informativeness_metrics) > 0
    
    def evaluate(self, teacher_responses: List[Dict[str, Any]], model_manager=None, 
                openai_client=None, **kwargs) -> List[Dict[str, Any]]:
        """Run informativeness evaluation on teacher responses."""
        student_data = kwargs.get('student_data', None)
        
        #if not student_data:
        #    raise ValueError("Student data is required for informativeness evaluation")
        
        return self._run_informativeness_evaluation(teacher_responses, student_data)
    
    def _run_informativeness_evaluation(self, teacher_responses: List[Dict[str, Any]],
                                      student_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run informativeness evaluation on teacher responses."""
        import asyncio

        # Batch remove answers BEFORE async loop (Option 1: clean separation)
        metrics_needing_removal = {'teacher_student_accuracy', 'teacher_student_accuracy_generalize',
                                   'answer_removed_explanation_only', 'answer_removed_explanation_generalize_to_new_question'}
        needs_answer_removal = any(m in self.requested_informativeness_metrics for m in metrics_needing_removal)

        thinking_without_answer_map = {}
        if needs_answer_removal:
            logger.info("Batch removing answers from teacher responses using LLM (GPT-4o-mini)...")
            # Prepare items for batch processing
            items_for_removal = []
            for teacher_idx, response in enumerate(teacher_responses):
                items_for_removal.append({
                    'teacher_thinking': response['teacher_thinking'],
                    'teacher_answer': response.get('teacher_answer', ''),
                    'teacher_response': response.get('teacher_response', 'placeholder'),
                    'teacher_idx': teacher_idx  # Track original index
                })

            # Batch remove answers using LLM (runs async internally)
            cleaned_items = remove_answers_batch(items_for_removal, use_llm=True)

            # Build map of teacher_idx -> cleaned thinking
            for item in cleaned_items:
                teacher_idx = item.get('teacher_idx', 0)
                thinking_without_answer_map[teacher_idx] = item.get('teacher_thinking_without_answer', '')

            logger.info(f"Batch removed answers from {len(cleaned_items)} teacher responses")

        # Now run async verification with pre-cleaned thinking
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self._run_informativeness_evaluation_async(teacher_responses, student_data, thinking_without_answer_map)
            )
            return result
        finally:
            loop.close()
    
    async def _run_informativeness_evaluation_async(self, teacher_responses: List[Dict[str, Any]],
                                                   student_data: List[Dict[str, Any]],
                                                   thinking_without_answer_map: Dict[int, str] = None) -> List[Dict[str, Any]]:
        """Async implementation of informativeness evaluation with batching.

        Args:
            teacher_responses: List of teacher response dictionaries
            student_data: List of student data dictionaries
            thinking_without_answer_map: Pre-cleaned thinking traces (teacher_idx -> cleaned_thinking)
        """
        logger.info("Running informativeness evaluation...")

        if thinking_without_answer_map is None:
            thinking_without_answer_map = {}
        
        
        # First, calculate student baseline accuracy and get detailed responses
        student_baseline_accuracy, student_baseline_detailed_results = await self._calculate_student_baseline_accuracy(student_data)
        logger.info(f"Student baseline accuracy: {student_baseline_accuracy:.4f}")
            
        informativeness_prompts = []
        prompt_metadata = []  # Track which teacher/student pair each prompt corresponds to
        
        for teacher_idx, response in enumerate(teacher_responses):
            # Handle teacher_student_accuracy separately (no student data loop needed)
            if 'teacher_student_accuracy' in self.requested_informativeness_metrics:
                # Use pre-cleaned thinking from batch removal
                thinking_without_answer = thinking_without_answer_map.get(
                    teacher_idx,
                    response['teacher_thinking']  # Fallback to original if not found
                )

                prompt = self.prompt_manager.create_answer_removed_explanation_question_to_own_question_prompt(
                    response['metadata']['input_str'],
                    thinking_without_answer,
                    task = response['metadata']['source_dataset']
                )

                informativeness_prompts.append(prompt)
                prompt_metadata.append({
                    'teacher_idx': teacher_idx,
                    'student_idx': 0,  # No real student data, just use 0 as placeholder
                    'student_answer': response['gold_answer'],  # Teacher's own answer
                    'informativeness_type': 'teacher_student_accuracy'
                })
            
            # Handle answer_removed_explanation_only separately (no student data loop needed, no question provided)
            if 'answer_removed_explanation_only' in self.requested_informativeness_metrics:
                # Use pre-cleaned thinking from batch removal
                thinking_without_answer = thinking_without_answer_map.get(
                    teacher_idx,
                    response['teacher_thinking']  # Fallback to original if not found
                )

                prompt = self.prompt_manager.create_answer_removed_explanation_only_prompt(
                    thinking_without_answer,
                    task = response['metadata']['source_dataset']
                )
                
                informativeness_prompts.append(prompt)
                prompt_metadata.append({
                    'teacher_idx': teacher_idx,
                    'student_idx': 0,  # No real student data, just use 0 as placeholder
                    'student_answer': response['gold_answer'],  # Teacher's own answer
                    'informativeness_type': 'answer_removed_explanation_only'
                })
            
            # Handle metrics that need student data
            student_based_metrics = [m for m in self.requested_informativeness_metrics
                                   if m not in ['teacher_student_accuracy', 'answer_removed_explanation_only']]

            for student_idx, student_item in enumerate(student_data):
                for inf_type in student_based_metrics:
                    if inf_type == 'whole_explanation_question_generalize_to_new_question':
                        prompt = self.prompt_manager.create_whole_explanation_question_generalize_to_new_question_prompt(
                            response['metadata']['input_str'],
                            response['teacher_thinking'],
                            student_item['metadata']['input_str'],
                            task = response['metadata']['source_dataset']
                        )
                    elif inf_type == 'answer_generalize_to_new_question':
                        prompt = self.prompt_manager.create_answer_generalize_to_new_question_prompt(
                            response['metadata']['input_str'],
                            response['gold_answer'],
                            student_item['metadata']['input_str'],
                            task = response['metadata']['source_dataset']
                        )
                    elif inf_type == 'teacher_student_accuracy_generalize':
                        # Use pre-cleaned thinking from batch removal
                        thinking_without_answer = thinking_without_answer_map.get(
                            teacher_idx,
                            response['teacher_thinking']  # Fallback to original if not found
                        )

                        prompt = self.prompt_manager.create_answer_removed_explanation_question_generalize_to_new_question_prompt(
                            response['metadata']['input_str'],
                            thinking_without_answer,
                            student_item['metadata']['input_str'],
                            task = response['metadata']['source_dataset']
                        )
                    elif inf_type == 'answer_removed_explanation_generalize_to_new_question':
                        # Use pre-cleaned thinking from batch removal
                        thinking_without_answer = thinking_without_answer_map.get(
                            teacher_idx,
                            response['teacher_thinking']  # Fallback to original if not found
                        )

                        prompt = self.prompt_manager.create_answer_removed_explanation_generalize_to_new_question_prompt(
                            thinking_without_answer,
                            student_item['metadata']['input_str'],
                            task = response['metadata']['source_dataset']
                        )
                    else:
                        continue
                    
                    informativeness_prompts.append(prompt)
                    prompt_metadata.append({
                        'teacher_idx': teacher_idx,
                        'student_idx': student_idx,
                        'student_answer': student_item['answer'],  # Student data answer
                        'informativeness_type': inf_type
                    })
        
        # Initialize client for informativeness evaluation using student model
        student_model_path = self.config['evaluation']['student_model']['model_path']
        logger.info(f"DEBUG: Using student model path for informativeness: {student_model_path}")
        logger.info(f"DEBUG: Number of informativeness prompts: {len(informativeness_prompts)}")
        
        client = OpenAIClient(student_model_path)
        # Store client for cost tracking
        self.student_openai_client = client
        
        # Generate responses using batched processing for efficiency
        informativeness_responses = []
        
        # Process in batches for better memory management and API rate limiting
        for i in range(0, len(informativeness_prompts), self.batch_size):
            batch_prompts = informativeness_prompts[i:i + self.batch_size]
            
            logger.info(f"Processing informativeness batch {i//self.batch_size + 1}/{(len(informativeness_prompts) + self.batch_size - 1)//self.batch_size}")
            
            # Generate responses for this batch using individual async calls
            try:
                batch_responses = await client.generate_individual_async(
                    batch_prompts,
                    temperature=self.teacher_config.get('temperature', 0.7),
                    max_tokens=self.teacher_config.get('max_tokens', 512),
                    model=self.teacher_config.get('model', 'gpt-4')
                )
                logger.info(f"DEBUG: Received {len(batch_responses)} responses for batch {i//self.batch_size + 1}")
                informativeness_responses.extend(batch_responses)
                
            except Exception as e:
                logger.error(f"DEBUG: Error in batch {i//self.batch_size + 1}: {e}")
                logger.error(f"DEBUG: Error type: {type(e)}")
                # For debugging, add empty responses for failed batch
                informativeness_responses.extend([{'text': ''} for _ in batch_prompts])
        
        logger.info(f"DEBUG: Total informativeness responses: {len(informativeness_responses)}")
        
        # Debug first few responses
        for i, response in enumerate(informativeness_responses[:3]):
            logger.info(f"DEBUG: Response {i}: {response}")
        
        # Calculate informativeness scores using reward calculator
        # Group responses by teacher and informativeness type
        teacher_scores = {}
        detailed_results = []  # Store detailed results for saving
        
        for i, (inf_response, metadata) in enumerate(zip(informativeness_responses, prompt_metadata)):
            teacher_idx = metadata['teacher_idx']
            student_idx = metadata['student_idx']
            student_answer = metadata['student_answer']
            inf_type = metadata['informativeness_type']
            
            # Try to extract from <final answer> tag first, then fall back to <answer> tag
            answer = (extract_answer(inf_response.get('text', ''), 'final answer') or 
                     extract_answer(inf_response.get('text', ''), 'answer') or 
                     inf_response.get('text', ''))
            
            # Create entry for scoring against student's correct answer
            entry = {
                'answer': student_answer,
                'metadata': teacher_responses[teacher_idx].get('metadata', {}),
                'data_source': teacher_responses[teacher_idx].get('data_source', ''),
                'index': student_idx
            }
            score = self.reward_calculator.calculate_score(answer, entry)
            
            # Store detailed result including full input prompt
            detailed_result = {
                'teacher_idx': teacher_idx,
                'student_idx': student_idx,
                'informativeness_type': inf_type,
                #'teacher_input': teacher_responses[teacher_idx]['metadata']['input_str'],
                'teacher_thinking': teacher_responses[teacher_idx]['teacher_thinking'],
                #'student_input': student_data[student_idx]['metadata']['input_str'],
                'student_answer': student_answer,
                'full_input_prompt': informativeness_prompts[i],
                'informativeness_response': inf_response.get('text', ''),
                'extracted_answer': answer,
                'score': score,
                'teacher_original_score': teacher_responses[teacher_idx]['reward_score'],
                "informativeness_score": student_answer == teacher_responses[teacher_idx]['teacher_answer']
            }
            detailed_results.append(detailed_result)
            
            # Accumulate scores for each teacher by type
            if teacher_idx not in teacher_scores:
                teacher_scores[teacher_idx] = {}
            if inf_type not in teacher_scores[teacher_idx]:
                teacher_scores[teacher_idx][inf_type] = []
            teacher_scores[teacher_idx][inf_type].append(score)
        
        # Calculate average informativeness score for each teacher response by type
        for teacher_idx, type_scores in teacher_scores.items():
            for inf_type, scores in type_scores.items():
                avg_score = sum(scores) / len(scores) if scores else 0.0
                teacher_responses[teacher_idx][f'{inf_type}_score'] = avg_score
            
            # Store detailed results in teacher response for later saving
            teacher_responses[teacher_idx]['informativeness_detailed_results'] = [
                result for result in detailed_results if result['teacher_idx'] == teacher_idx
            ]
            
            # Store student baseline accuracy for comparison
            teacher_responses[teacher_idx]['student_accuracy_baseline'] = student_baseline_accuracy
            
            # Store student baseline detailed results for later saving (only in first teacher response to avoid duplication)
            if teacher_idx == 0:
                teacher_responses[teacher_idx]['student_baseline_detailed_results'] = student_baseline_detailed_results
            
        return teacher_responses
    
    async def _calculate_student_baseline_accuracy(self, student_data: List[Dict[str, Any]]) -> tuple[float, List[Dict[str, Any]]]:
        """Calculate baseline accuracy of student model on the student dataset and return detailed responses."""
        logger.info("Calculating student baseline accuracy...")
        
        # Create prompts for student model using the same developer prompt structure as teacher
        student_prompts = []
        for student_item in student_data:
            # Use the same prompt structure as teacher - just the question with developer prompt
            question = student_item['question']
            prompt = self.prompt_manager.create_openai_prompt(question)
            student_prompts.append(prompt)
        
        # Initialize student model client
        student_model_path = self.config['evaluation']['student_model']['model_path']
        student_client = OpenAIClient(student_model_path)
        
        # Generate responses using batched processing
        student_responses = []
        for i in range(0, len(student_prompts), self.batch_size):
            batch_prompts = student_prompts[i:i + self.batch_size]
            
            logger.info(f"Processing student baseline batch {i//self.batch_size + 1}/{(len(student_prompts) + self.batch_size - 1)//self.batch_size}")
            
            try:
                batch_responses = await student_client.generate_individual_async(
                    batch_prompts,
                    temperature=self.teacher_config.get('temperature', 0.7),
                    max_tokens=self.teacher_config.get('max_tokens', 512),
                    model=self.teacher_config.get('model', 'gpt-4')
                )
                student_responses.extend(batch_responses)
                
            except Exception as e:
                logger.error(f"Error in student baseline batch {i//self.batch_size + 1}: {e}")
                # Add empty responses for failed batch
                student_responses.extend([{'text': ''} for _ in batch_prompts])
        
        # Calculate accuracy and collect detailed results
        correct_count = 0
        total_count = len(student_data)
        detailed_baseline_results = []
        
        for i, (student_response, student_item) in enumerate(zip(student_responses, student_data)):
            # Extract answer from student response
            answer = (extract_answer(student_response.get('text', ''), 'final answer') or 
                     extract_answer(student_response.get('text', ''), 'answer') or 
                     student_response.get('text', ''))
            
            # Calculate score using the same scoring method as informativeness
            entry = {
                'answer': student_item['answer'],
                'metadata': student_item.get('metadata', {}),
                'data_source': student_item.get('data_source', ''),
                'index': i
            }
            score = self.reward_calculator.calculate_score(answer, entry)
            
            # Store detailed result
            detailed_result = {
                'index': i,
                'question': student_item['question'],
                'correct_answer': student_item['answer'],
                'full_input_prompt': student_prompts[i],
                'student_baseline_response': student_response.get('text', ''),
                'extracted_answer': answer,
                'score': score,
                'metadata': student_item.get('metadata', {}),
                'data_source': student_item.get('data_source', ''),
                'seed': student_item.get('seed', 0)
            }
            detailed_baseline_results.append(detailed_result)
            
            if score > 0:  # Assuming binary scoring (1 for correct, 0 for incorrect)
                correct_count += 1
        
        baseline_accuracy = correct_count / total_count if total_count > 0 else 0.0
        return baseline_accuracy, detailed_baseline_results
    
    def process_teacher_responses(self, teacher_responses_path: str, 
                                teacher_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process teacher responses and calculate informativeness metrics.
        This is called AFTER the informativeness inference has been completed.
        
        Args:
            teacher_responses_path: Path where original responses are saved
            teacher_data: Teacher responses with informativeness results
        
        Returns:
            Dictionary with informativeness metrics
        """
        try:            
            # Calculate scores for each informativeness type
            results = {'num_samples': len(teacher_data)}
            
            for inf_type in self.requested_informativeness_metrics:
                scores = [item.get(f'{inf_type}_score', 0.0) for item in teacher_data]
                mean_score = sum(scores) / len(scores) if scores else 0.0
                
                results[f'mean_{inf_type}_score'] = mean_score
                results[f'{inf_type}_details'] = {
                    'scores': scores,
                    'min_score': min(scores) if scores else 0.0,
                    'max_score': max(scores) if scores else 0.0,
                    'std_score': float(np.std(scores)) if scores else 0.0
                }
            
            # Calculate teacher accuracy for comparison
            teacher_scores = [item.get('reward_score', 0.0) for item in teacher_data]
            mean_teacher_accuracy = sum(teacher_scores) / len(teacher_scores) if teacher_scores else 0.0
            results['mean_teacher_accuracy'] = mean_teacher_accuracy
            
            # Include student baseline accuracy if available
            student_baseline_scores = [item.get('student_accuracy_baseline', 0.0) for item in teacher_data]
            if student_baseline_scores and any(score > 0 for score in student_baseline_scores):
                # Take the first non-zero baseline (should be same for all)
                student_baseline = next((score for score in student_baseline_scores if score > 0), 0.0)
                results['student_accuracy_baseline'] = student_baseline
            
            logger.info(f"Informativeness metrics: {len(teacher_data)} samples")
            for inf_type in self.requested_informativeness_metrics:
                if f'mean_{inf_type}_score' in results:
                    logger.info(f"  Mean {inf_type} score: {results[f'mean_{inf_type}_score']:.4f}")
            logger.info(f"  Mean teacher accuracy: {mean_teacher_accuracy:.4f}")
            if 'student_accuracy_baseline' in results:
                logger.info(f"  Student baseline accuracy: {results['student_accuracy_baseline']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing informativeness metrics: {e}")
            error_results = {
                'mean_teacher_accuracy': 0.0,
                'student_accuracy_baseline': 0.0,
                'num_samples': 0,
                'error': str(e)
            }
            for inf_type in self.requested_informativeness_metrics:
                error_results[f'mean_{inf_type}_score'] = 0.0
            return error_results