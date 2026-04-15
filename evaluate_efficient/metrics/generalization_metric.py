"""
Generalization Metric

Tests the model's ability to generalize in two ways:
1. other_task: Evaluating on other tasks not used for training
2. other_difficulty: Evaluating on the same task with different difficulty parameters
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_metric import BaseMetric
from ..utils.data_generator import DataGenerator

logger = logging.getLogger(__name__)


class GeneralizationMetric(BaseMetric):
    """Tests model generalization across different reasoning tasks and difficulties."""
    
    # All available tasks for generalization testing
    ALL_TASKS = [
        'spiral_matrix',
        'simple_equations', 
        'futoshiki',
        'family_relationships',
        'mini_sudoku'
    ]
    
    # Default task parameters for generalization testing (used for other_task)
    TASK_CONFIGS = {
        'spiral_matrix': {
            'min_n': 2,
            'max_n': 4
        },
        'simple_equations': {
            'min_terms': 2,
            'max_terms': 4,
            'min_value': 1,
            'max_value': 100,
            'operators_weights': [0.4, 0.4, 0.2]
        },
        'futoshiki': {
            "min_board_size": 4,
            "max_board_size": 9,
            "min_difficulty": 0,
            "max_difficulty": 3,
        },
        'family_relationships': {
            'min_family_size': 4,
            'max_family_size': 8
        },
        'mini_sudoku': {
            'min_empty': 6,
            'max_empty': 10
        }
    }
    
    # Alternative difficulty parameters for other_difficulty testing
    DIFFICULTY_CONFIGS = {
        'spiral_matrix': {
            'easy': {'min_n': 2, 'max_n': 4},
            'medium': {'min_n': 4, 'max_n': 6}, 
            'hard': {'min_n': 6, 'max_n': 8}
        },
        'simple_equations': {
            'easy': {
                'min_terms': 2, 'max_terms': 4, 
                'min_value': 1, 'max_value': 100,
                'operators_weights': [0.4, 0.4, 0.2]
            },
            'medium': {
                'min_terms': 3, 'max_terms': 6,
                'min_value': 10, 'max_value': 500, 
                'operators_weights': [0.35, 0.35, 0.3]
            },
            'hard': {
                'min_terms': 4, 'max_terms': 8,
                'min_value': 50, 'max_value': 1000,
                'operators_weights': [0.3, 0.3, 0.4]
            }
        },
        'futoshiki': {
            'easy': {'min_board_size': 4, 'max_board_size': 6, 'min_difficulty': 0, 'max_difficulty': 2},
            'medium': {'min_board_size': 6, 'max_board_size': 8, 'min_difficulty': 1, 'max_difficulty': 3},
            'hard': {'min_board_size': 8, 'max_board_size': 10, 'min_difficulty': 2, 'max_difficulty': 4}
        },
        'family_relationships': {
            'easy': {'min_family_size': 4, 'max_family_size': 8},
            'medium': {'min_family_size': 8, 'max_family_size': 12},
            'hard': {'min_family_size': 12, 'max_family_size': 16}
        },
        'mini_sudoku': {
            'easy': {'min_empty': 6, 'max_empty': 10},
            'medium': {'min_empty': 4, 'max_empty': 5},
            'hard': {'min_empty': 2, 'max_empty': 3}
        }
    }
    
    def __init__(self, config: Dict[str, Any], reward_calculator, prompt_manager):
        """Initialize generalization metric."""
        super().__init__(config, reward_calculator, prompt_manager)
        
        # Initialize metric configuration
        self._init_metric_config()
        
        # Get the training task to exclude from generalization testing
        self.training_task = config['evaluation']['teacher_dataset']['task_name']
        
        # Determine which generalization types to test
        self.generalization_types = self._get_generalization_types()
        
        # Get tasks to test generalization on (all except training task)
        self.generalization_tasks = [task for task in self.ALL_TASKS if task != self.training_task]
        
        # Number of examples to test per task/difficulty
        self.examples_per_task = 100
        
        # Initialize data generators for other_task testing
        self.task_generators = {}
        if 'other_task' in self.generalization_types:
            for task in self.generalization_tasks:
                task_params = self.TASK_CONFIGS.get(task, {})
                self.task_generators[task] = DataGenerator(task, task_params)
        
        # Initialize data generators for other_difficulty testing
        self.difficulty_generators = {}
        if 'other_difficulty' in self.generalization_types:
            for difficulty in ['easy', 'medium', 'hard']:
                if self.training_task in self.DIFFICULTY_CONFIGS:
                    difficulty_params = self.DIFFICULTY_CONFIGS[self.training_task].get(difficulty, {})
                    if difficulty_params:
                        self.difficulty_generators[difficulty] = DataGenerator(self.training_task, difficulty_params)
        
        logger.info(f"Initialized generalization metric for training task '{self.training_task}'")
        logger.info(f"Generalization types: {self.generalization_types}")
        if 'other_task' in self.generalization_types:
            logger.info(f"Will test other_task generalization on: {self.generalization_tasks}")
        if 'other_difficulty' in self.generalization_types:
            logger.info(f"Will test other_difficulty generalization on: {list(self.difficulty_generators.keys())}")
    
    def _get_generalization_types(self) -> List[str]:
        """Determine which generalization types to test based on config."""
        requested_metrics = self.config.get('evaluation', {}).get('metrics', [])
        
        # Check for specific generalization types
        generalization_types = []
        if 'generalization_other_task' in requested_metrics:
            generalization_types.append('other_task')
        if 'generalization_other_difficulty' in requested_metrics:
            generalization_types.append('other_difficulty')
        
        # Fallback: if just 'generalization' is specified, default to both
        if 'generalization' in requested_metrics and not generalization_types:
            generalization_types = ['other_task', 'other_difficulty']
        
        return generalization_types
    
    def _init_metric_config(self):
        """Initialize generalization metric configuration."""
        self.generalization_types = self._get_generalization_types()
        self.can_run_generalization = len(self.generalization_types) > 0
    
    def can_run(self) -> bool:
        """Check if generalization evaluation should be run."""
        return self.can_run_generalization
    
    def evaluate(self, teacher_responses: List[Dict[str, Any]], model_manager=None, 
                openai_client=None, **kwargs) -> List[Dict[str, Any]]:
        """Run generalization evaluation on teacher responses."""
        if not self.can_run():
            logger.warning("Generalization evaluation requested but not configured")
            return teacher_responses
        
        logger.info("Running generalization evaluation...")
        
        generalization_results = {}
        
        # Test other_task generalization
        if 'other_task' in self.generalization_types:
            logger.info("Running other_task generalization evaluation...")
            other_task_data = self._generate_other_task_data()
            
            for task_name, task_data in other_task_data.items():
                logger.info(f"Testing other_task generalization on {task_name} ({len(task_data)} examples)")
                task_results = self._evaluate_task(task_data, model_manager, openai_client, task_name)
                generalization_results[f"other_task_{task_name}"] = task_results
        
        # Test other_difficulty generalization
        if 'other_difficulty' in self.generalization_types:
            logger.info("Running other_difficulty generalization evaluation...")
            other_difficulty_data = self._generate_other_difficulty_data()
            
            for difficulty, task_data in other_difficulty_data.items():
                logger.info(f"Testing other_difficulty generalization on {self.training_task}_{difficulty} ({len(task_data)} examples)")
                task_results = self._evaluate_task(task_data, model_manager, openai_client, f"{self.training_task}_{difficulty}")
                generalization_results[f"other_difficulty_{difficulty}"] = task_results
        
        # Add generalization results to teacher responses
        for i, response in enumerate(teacher_responses):
            response['generalization_results'] = generalization_results
            # Only add detailed results to the first response to avoid duplication
            if i == 0:
                response['generalization_detailed_results'] = self._create_detailed_results(generalization_results)
        
        return teacher_responses
    
    def _generate_other_task_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate test data for other_task generalization."""
        other_task_data = {}
        
        for task_name in self.generalization_tasks:
            logger.info(f"Generating {self.examples_per_task} examples for other_task: {task_name}")
            
            # Use a fixed seed for reproducible generalization testing
            seed = 12345
            
            try:
                # Generate data using the task's data generator
                task_data = self.task_generators[task_name].generate_teacher_dataset(
                    seed,
                    self.examples_per_task,
                    0
                )
                other_task_data[task_name] = task_data
                logger.info(f"Generated {len(task_data)} examples for other_task: {task_name}")
                
            except Exception as e:
                logger.error(f"Failed to generate data for other_task {task_name}: {e}")
                other_task_data[task_name] = []
        
        return other_task_data
    
    def _generate_other_difficulty_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate test data for other_difficulty generalization."""
        other_difficulty_data = {}
        
        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in self.difficulty_generators:
                logger.info(f"Generating {self.examples_per_task} examples for other_difficulty: {self.training_task}_{difficulty}")
                
                # Use a fixed seed for reproducible generalization testing
                seed = 12345 + hash(difficulty) % 10000  # Vary seed by difficulty
                
                try:
                    # Generate data using the difficulty's data generator
                    task_data = self.difficulty_generators[difficulty].generate_teacher_dataset(
                        seed,
                        self.examples_per_task,
                        0
                    )
                    other_difficulty_data[difficulty] = task_data
                    logger.info(f"Generated {len(task_data)} examples for other_difficulty: {self.training_task}_{difficulty}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate data for other_difficulty {self.training_task}_{difficulty}: {e}")
                    other_difficulty_data[difficulty] = []
        
        return other_difficulty_data
    
    def _evaluate_task(self, task_data: List[Dict[str, Any]], model_manager, openai_client, task_name: str) -> Dict[str, Any]:
        """Evaluate model on a specific task's data."""
        from trainers.reward_calculator import RewardCalculator
        
        # Extract base task name (remove difficulty suffix like '_easy', '_medium', '_hard')
        base_task_name = task_name
        if '_easy' in task_name or '_medium' in task_name or '_hard' in task_name:
            # For other_difficulty tests, use the training task as base
            base_task_name = self.training_task
        
        # Create task-specific reward calculator with base task name
        task_reward_calculator = RewardCalculator(task="reasoning_gym", task_type=base_task_name)
        
        # Use the accuracy metric's evaluation logic but with task-specific reward calculator
        if openai_client:
            # OpenAI API evaluation
            responses = self._evaluate_with_openai(task_data, openai_client)
        else:
            # Local model evaluation
            responses = self._evaluate_with_local_model(task_data, model_manager)
        
        # Calculate accuracy using task-specific reward calculator
        total_score = 0
        correct_count = 0
        detailed_responses = []
        
        for i, (response, data_item) in enumerate(zip(responses, task_data)):
            # Extract answer from response
            from data.utils import extract_answer
            answer = extract_answer(response.get('text', ''), 'answer') or response.get('text', '')
            
            # Calculate score using task-specific reward calculator
            entry = {
                'answer': data_item['answer'],
                'metadata': data_item.get('metadata', {}),
                'data_source': data_item.get('data_source', ''),
                'index': i
            }
            score = task_reward_calculator.calculate_score(answer, entry)
            total_score += score
            if score > 0:
                correct_count += 1
            
            # Store detailed response
            detailed_responses.append({
                'index': i,
                'question': data_item['question'],
                'correct_answer': data_item['answer'],
                'model_response': response.get('text', ''),
                'extracted_answer': answer,
                'score': score,
                'task': task_name
            })
        
        accuracy = total_score / len(task_data) if task_data else 0.0
        
        return {
            'task_name': task_name,
            'accuracy': accuracy,
            'total_score': total_score,
            'total_examples': len(task_data),
            'correct_count': correct_count,
            'detailed_responses': detailed_responses
        }
    
    def _evaluate_with_openai(self, task_data: List[Dict[str, Any]], openai_client) -> List[Dict[str, Any]]:
        """Evaluate using OpenAI API."""
        prompts = []
        for item in task_data:
            prompt = self.prompt_manager.create_openai_prompt(item['question'])
            prompts.append(prompt)
        
        # Generate responses in batches
        responses = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            try:
                batch_responses = openai_client.generate_batch(
                    batch_prompts,
                    temperature=0.7,
                    max_tokens=512
                )
                responses.extend(batch_responses)
            except Exception as e:
                logger.error(f"Error in OpenAI batch {i//self.batch_size + 1}: {e}")
                responses.extend([{'text': ''} for _ in batch_prompts])
        
        return responses
    
    def _evaluate_with_local_model(self, task_data: List[Dict[str, Any]], model_manager) -> List[Dict[str, Any]]:
        """Evaluate using local model."""
        from vllm import SamplingParams
        
        prompts = []
        tokenizer = model_manager.current_model.get_tokenizer()
        for item in task_data:
            prompt = self.prompt_manager.create_teacher_prompt(
                item['question'], 
                tokenizer, 
                is_perturbation=False
            )
            prompts.append(prompt)
        
        # Set up sampling parameters
        stop_tokens = model_manager.get_stop_tokens()
        sampling_params = SamplingParams(
            temperature=self.teacher_config['temperature'],
            top_p=self.teacher_config['top_p'],
            top_k=self.teacher_config['top_k'],
            max_tokens=self.teacher_config['max_tokens'],
            stop=stop_tokens
        )
        
        # Generate responses in batches
        all_responses = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            outputs = model_manager.current_model.generate(batch_prompts, sampling_params, use_tqdm=False)
            
            batch_responses = []
            for output in outputs:
                response_text = output.outputs[0].text
                batch_responses.append({
                    'text': response_text,
                    'generation_info': {
                        'finish_reason': output.outputs[0].finish_reason,
                        'num_tokens': len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else 0,
                    }
                })
            
            all_responses.extend(batch_responses)
        
        return all_responses
    
    def _create_detailed_results(self, generalization_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create detailed results list for saving."""
        detailed_results = []
        for task_name, task_results in generalization_results.items():
            detailed_results.extend(task_results.get('detailed_responses', []))
        return detailed_results
    
    def process_teacher_responses(self, teacher_responses_path: str, 
                                teacher_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process teacher responses and calculate generalization metrics.
        This is called AFTER the generalization inference has been completed.
        """
        try:
            # Extract generalization results from teacher responses
            generalization_results = {}
            if teacher_data and 'generalization_results' in teacher_data[0]:
                generalization_results = teacher_data[0]['generalization_results']
            
            # Calculate summary metrics
            results = {
                'training_task': self.training_task,
                'generalization_types': self.generalization_types,
                'generalization_tests': list(generalization_results.keys()),
                'num_examples_per_task': self.examples_per_task,
                'num_samples': len(teacher_data)
            }
            
            # Separate results by generalization type
            other_task_results = {k: v for k, v in generalization_results.items() if k.startswith('other_task_')}
            other_difficulty_results = {k: v for k, v in generalization_results.items() if k.startswith('other_difficulty_')}
            
            # Calculate other_task metrics
            if other_task_results:
                task_accuracies = {}
                total_accuracy = 0
                task_count = 0
                
                for test_name, test_results in other_task_results.items():
                    accuracy = test_results.get('accuracy', 0.0)
                    task_accuracies[test_name] = accuracy
                    total_accuracy += accuracy
                    task_count += 1
                    
                    # Add detailed metrics for this test
                    results[f'{test_name}_accuracy'] = accuracy
                    results[f'{test_name}_correct_count'] = test_results.get('correct_count', 0)
                    results[f'{test_name}_total_examples'] = test_results.get('total_examples', 0)
                
                avg_other_task_accuracy = total_accuracy / task_count if task_count > 0 else 0.0
                results['average_other_task_accuracy'] = avg_other_task_accuracy
                results['other_task_accuracies'] = task_accuracies
            
            # Calculate other_difficulty metrics
            if other_difficulty_results:
                difficulty_accuracies = {}
                total_accuracy = 0
                difficulty_count = 0
                
                for test_name, test_results in other_difficulty_results.items():
                    accuracy = test_results.get('accuracy', 0.0)
                    difficulty_accuracies[test_name] = accuracy
                    total_accuracy += accuracy
                    difficulty_count += 1
                    
                    # Add detailed metrics for this test
                    results[f'{test_name}_accuracy'] = accuracy
                    results[f'{test_name}_correct_count'] = test_results.get('correct_count', 0)
                    results[f'{test_name}_total_examples'] = test_results.get('total_examples', 0)
                
                avg_other_difficulty_accuracy = total_accuracy / difficulty_count if difficulty_count > 0 else 0.0
                results['average_other_difficulty_accuracy'] = avg_other_difficulty_accuracy
                results['other_difficulty_accuracies'] = difficulty_accuracies
            
            # Calculate overall average generalization accuracy
            all_accuracies = []
            for test_results in generalization_results.values():
                all_accuracies.append(test_results.get('accuracy', 0.0))
            
            avg_generalization_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
            results['average_generalization_accuracy'] = avg_generalization_accuracy
            
            logger.info(f"Generalization metrics: {len(teacher_data)} samples")
            logger.info(f"  Training task: {self.training_task}")
            logger.info(f"  Generalization types: {self.generalization_types}")
            logger.info(f"  Overall average generalization accuracy: {avg_generalization_accuracy:.4f}")
            
            if 'other_task' in self.generalization_types and 'other_task_accuracies' in results:
                logger.info(f"  Other task average accuracy: {results['average_other_task_accuracy']:.4f}")
                for test_name, accuracy in results['other_task_accuracies'].items():
                    logger.info(f"    {test_name} accuracy: {accuracy:.4f}")
            
            if 'other_difficulty' in self.generalization_types and 'other_difficulty_accuracies' in results:
                logger.info(f"  Other difficulty average accuracy: {results['average_other_difficulty_accuracy']:.4f}")
                for test_name, accuracy in results['other_difficulty_accuracies'].items():
                    logger.info(f"    {test_name} accuracy: {accuracy:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing generalization metrics: {e}")
            return {
                'training_task': self.training_task,
                'generalization_types': self.generalization_types,
                'average_generalization_accuracy': 0.0,
                'num_samples': 0,
                'error': str(e)
            }