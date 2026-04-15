"""
Teacher Evaluator

Orchestrates teacher model evaluation with clean separation of concerns.
Extracted from the complex run_teacher_only.py for better organization.
"""

import os
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

import torch
from omegaconf import OmegaConf

from .utils.checkpoint_loader import CheckpointLoader
from .utils.data_generator import DataGenerator
from .teacher_single_model import TeacherSingleModel
from .metrics.accuracy_metric import AccuracyMetric
from .utils.performance_monitor import PerformanceMonitor
from .utils.config_validator import validate_config

from .metrics.usefulness_metric import UsefulnessMetric

logger = logging.getLogger(__name__)


class TeacherEvaluator:
    """Orchestrates teacher model evaluation with support for CoT perturbation."""
    
    def __init__(self, config):
        """Initialize the evaluator."""
        logger.info("Initializing Teacher Evaluator...")
        
        self.config = validate_config(config)
        self.eval_config = self.config.evaluation
        
        # Initialize components
        self._init_components()
        self._init_metrics()
        self._init_data_generator()
        
        # Setup output
        self.output_dir = self.eval_config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Teacher Evaluator initialized successfully")
    
    def _init_components(self):
        """Initialize core components."""
        self.teacher_pipeline = TeacherSingleModel(OmegaConf.to_container(self.config, resolve=True))
        self.performance_monitor = PerformanceMonitor(OmegaConf.to_container(self.config, resolve=True))
        
        # Only initialize checkpoint loader for local models
        if not self.teacher_pipeline.use_openai_api:
            self.checkpoint_loader = CheckpointLoader(self.eval_config.base_model_path)
    
    def _init_metrics(self):
        """Initialize metrics based on requested evaluation types."""
        # Use the consolidated metrics from teacher_pipeline which are properly initialized
        # with config, reward_calculator, and prompt_manager
        self.teacher_accuracy_metrics = self.teacher_pipeline.accuracy_metric
        
        # Initialize CoT perturbation metrics if requested
        self.cot_perturbation_metrics = None
        requested_metrics_for_cot = self.config.get('evaluation', {}).get('metrics', [])
        self.cot_types = ['truncate', 'filler', 'shuffle', 'remove_thinking', 'incremental_thinking', 'expert_thinking', 'truncate_random', 'truncate_second', 'truncate_first', 'replace']
        self.requested_cot_metrics = [r for r in requested_metrics_for_cot if r in self.cot_types]

        logger.info(f"Requested metrics: {requested_metrics_for_cot}")
        logger.info(f"Filtered CoT metrics: {self.requested_cot_metrics}")

        if len(self.requested_cot_metrics) > 0:
            self.cot_perturbation_metrics = self.teacher_pipeline.usefulness_metric
            logger.info("Initialized CoT perturbation metrics")
            
        # Initialize informativeness metrics if requested
        self.informativeness_metrics = None
        requested_metrics_for_informativeness = self.config.get('evaluation', {}).get('metrics', [])
        informativeness_types = ['whole_explanation_question_generalize_to_new_question', 'answer_generalize_to_new_question', 'answer_removed_explanation_question_generalize_to_new_question', 'answer_removed_explanation_question_to_own_question', 'answer_removed_explanation_only', 'answer_removed_explanation_generalize_to_new_question']
        self.requested_informativeness_metrics = [r for r in requested_metrics_for_informativeness if r in informativeness_types]
        self.run_informativeness = len(self.requested_informativeness_metrics) > 0
        
        if self.run_informativeness:
            self.informativeness_metrics = self.teacher_pipeline.informativeness_metric
            logger.info("Initialized informativeness evaluation metrics")
        
        # Initialize generalization metrics if requested
        self.generalization_metrics = None
        requested_metrics_for_generalization = self.config.get('evaluation', {}).get('metrics', [])
        self.run_generalization = 'generalization' in requested_metrics_for_generalization
        
        if self.run_generalization:
            self.generalization_metrics = self.teacher_pipeline.generalization_metric
            logger.info("Initialized generalization evaluation metrics")
    
    def _init_data_generator(self):
        """Initialize data generator."""
        dataset_config = OmegaConf.to_container(self.eval_config.teacher_dataset, resolve=True)
        
        task_params = {k: v for k, v in dataset_config.items() 
                      if k not in ['task_name', 'seed', 'size', 'val_start']}
        
        self.data_generator = DataGenerator(dataset_config['task_name'], task_params)
    
    def _generate_teacher_data(self) -> List[Dict[str, Any]]:
        """Generate teacher dataset."""
        logger.info("Generating teacher dataset...")
        
        from_path = OmegaConf.select(
            self.eval_config, 
            "from_path",
            default=None
        )

        logger.info(f"from_path: {from_path}")

        if from_path:
            teacher_data = []
            with open(from_path, 'r') as f:
                teacher_data_from_path = json.load(f)
            for item in teacher_data_from_path:
                teacher_data.append({
                    'index': item['index'],
                    'question': item['question'],
                    'answer': item['teacher_answer'],
                    'metadata': item['metadata'],
                })
            logger.info(f"Loaded {len(teacher_data)} teacher examples from {from_path}")
            return teacher_data
        

        teacher_data = self.data_generator.generate_teacher_dataset(
            self.eval_config.teacher_dataset.seed,
            self.eval_config.teacher_dataset.size,
            getattr(self.eval_config.teacher_dataset, 'val_start', 0)
        )
        
        logger.info(f"Generated {len(teacher_data)} teacher examples")
        return teacher_data
    
    def _generate_student_data(self) -> List[Dict[str, Any]]:
        """Generate student dataset for informativeness evaluation."""
        logger.info("Generating student dataset...")
        
        # Use student dataset config if available, otherwise fall back to teacher config
        student_config = getattr(self.eval_config, 'student_dataset', self.eval_config.teacher_dataset)
        
        student_data = self.data_generator.generate_student_dataset(
            getattr(student_config, 'seed', self.eval_config.teacher_dataset.seed + 1000),
            getattr(student_config, 'size', self.eval_config.teacher_dataset.size),
            getattr(student_config, 'val_start', self.eval_config.teacher_dataset.val_start)
        )
        
        logger.info(f"Generated {len(student_data)} student examples")
        return student_data
    
    def _get_checkpoint_paths(self) -> List[str]:
        """Get checkpoint paths to evaluate."""
        if self.teacher_pipeline.use_openai_api:
            return ["openai_api"]
        
        checkpoint_dir = self.eval_config.checkpoint_dir
        start_step = getattr(self.eval_config, 'start_step', -1)
        
        checkpoint_paths = self.checkpoint_loader.get_checkpoint_paths(checkpoint_dir, start_step)
        
        # Add base model as step_0 if we have checkpoints
        if checkpoint_paths:
            base_model_path = self.eval_config.base_model_path
            # Insert base model at the beginning as step_0
            checkpoint_paths.insert(0, base_model_path)
            logger.info(f"Added base model as step_0: {base_model_path}")
        
        # Apply max checkpoints limit (accounting for base model)
        max_checkpoints = getattr(self.eval_config, 'max_checkpoints', None)
        if max_checkpoints and len(checkpoint_paths) > max_checkpoints:
            # Keep base model (step_0) and sample from remaining checkpoints
            base_model = checkpoint_paths[0]
            remaining_checkpoints = checkpoint_paths[1:]
            if remaining_checkpoints and max_checkpoints > 1:
                step = len(remaining_checkpoints) // (max_checkpoints - 1)
                sampled_checkpoints = remaining_checkpoints[::step][:max_checkpoints - 1]
                checkpoint_paths = [base_model] + sampled_checkpoints
        
        logger.info(f"Found {len(checkpoint_paths)} checkpoints to evaluate (including base model)")
        return checkpoint_paths
    
    def _should_process_checkpoint(self, checkpoint_path: str, start_step: int) -> bool:
        """Check if checkpoint should be processed based on step filtering."""
        if start_step < 0:
            return True
        
        checkpoint_name = self._extract_checkpoint_name(checkpoint_path)
        checkpoint_step_str = self.teacher_pipeline._extract_step_number(checkpoint_name)
        
        try:
            checkpoint_step = int(checkpoint_step_str)
            if checkpoint_step < start_step:
                logger.warning(f"Skipping checkpoint {checkpoint_path} (step {checkpoint_step} < {start_step})")
                return False
            return True
        except (ValueError, TypeError):
            logger.warning(f"Could not parse step number '{checkpoint_step_str}', skipping")
            return False
    
    def _extract_checkpoint_name(self, checkpoint_path: str) -> str:
        """Extract checkpoint name from path."""
        if checkpoint_path == "openai_api":
            return "openai_api"
        elif os.path.basename(checkpoint_path) == "actor":
            return os.path.basename(os.path.dirname(checkpoint_path))
        else:
            return os.path.basename(checkpoint_path)
        
#### --------------- logging --------------- ####
    def _log_cot_results(self, checkpoint_name: str, cot_results: Dict[str, Any]):
        """Log CoT perturbation results."""
        logger.info(f"CoT results - original: {cot_results.get('original_reward', 0):.3f}, "
                   f"truncate: {cot_results.get('truncate_reward', 0):.3f}, "
                   f"filler: {cot_results.get('filler_reward', 0):.3f}")
    
    def _log_checkpoint_results(self, checkpoint_name: str, metrics: Dict[str, Any]):
        """Log checkpoint results."""
        logger.info(f"Checkpoint {checkpoint_name} - reward: {metrics['average_reward']:.3f} "
                   f"({metrics['total_score']:.1f}/{metrics['total_count']})")
        
        if metrics['total_mean_perplexity'] > 0:
            logger.info(f"Checkpoint {checkpoint_name} - perplexity: {metrics['total_mean_perplexity']:.3f} "
                       f"({metrics['perplexity_count']} responses)")
    
    def _log_performance(self):
        """Log performance stats."""
        self.performance_monitor.log_gpu_stats()
        self.performance_monitor.log_memory_usage()
        
        # Cleanup between checkpoints
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

#### --------------- cot metrics processing --------------- ####
    def _process_cot_metrics(self, checkpoint_name: str, teacher_responses: List[Dict[str, Any]], 
                           teacher_responses_path: str) -> Dict[str, Any]:
        """Process CoT perturbation metrics and add checkpoint results."""
        if not self.cot_perturbation_metrics:
            return {}
        
        # Process CoT metrics
        cot_results = self.cot_perturbation_metrics.process_teacher_responses(
            teacher_responses_path, teacher_responses
        )
        
        # Add checkpoint result and log
        self.cot_perturbation_metrics.add_checkpoint_result(checkpoint_name, cot_results)
        self._log_cot_results(checkpoint_name, cot_results)
        
        return cot_results
    
    def _save_cot_metrics(self):
        """Save CoT perturbation metrics to file."""
        if not self.cot_perturbation_metrics:
            return
        
        metrics_file = os.path.join(self.output_dir, "cot_perturbation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.cot_perturbation_metrics.to_dict(), f, indent=2)
        
        self.cot_perturbation_metrics.print_summary()

#### --------------- informativeness metrics processing --------------- ####
    def _process_informativeness_metrics(self, checkpoint_name: str, teacher_responses: List[Dict[str, Any]], 
                                       teacher_responses_path: str) -> Dict[str, Any]:
        """Process informativeness metrics and add checkpoint results."""
        if not self.informativeness_metrics:
            return {}
        
        # Process informativeness metrics
        informativeness_results = self.informativeness_metrics.process_teacher_responses(
            teacher_responses_path, teacher_responses
        )
        
        # Add checkpoint result and log
        self.informativeness_metrics.add_checkpoint_result(checkpoint_name, informativeness_results)
        self._log_informativeness_results(checkpoint_name, informativeness_results)
        
        return informativeness_results
    
    def _log_informativeness_results(self, checkpoint_name: str, informativeness_results: Dict[str, Any]):
        """Log informativeness evaluation results."""
        log_parts = []
        for inf_type in ['whole_explanation_question_generalize_to_new_question', 'answer_generalize_to_new_question']:
            score_key = f'mean_{inf_type}_score'
            if score_key in informativeness_results:
                log_parts.append(f"{inf_type}: {informativeness_results[score_key]:.3f}")
        
        teacher_acc = informativeness_results.get('mean_teacher_accuracy', 0)
        log_message = f"Informativeness results - {', '.join(log_parts)}, teacher accuracy: {teacher_acc:.3f}"
        logger.info(log_message)
    
    def _save_informativeness_metrics(self):
        """Save informativeness metrics to file."""
        if not self.informativeness_metrics:
            return
        
        metrics_file = os.path.join(self.output_dir, "informativeness_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.informativeness_metrics.to_dict(), f, indent=2)
        
        self.informativeness_metrics.print_summary()

#### --------------- generalization metrics processing --------------- ####
    def _process_generalization_metrics(self, checkpoint_name: str, teacher_responses: List[Dict[str, Any]], 
                                       teacher_responses_path: str) -> Dict[str, Any]:
        """Process generalization metrics and add checkpoint results."""
        if not self.generalization_metrics:
            return {}
        
        # Process generalization metrics
        generalization_results = self.generalization_metrics.process_teacher_responses(
            teacher_responses_path, teacher_responses
        )
        
        # Add checkpoint result and log
        self.generalization_metrics.add_checkpoint_result(checkpoint_name, generalization_results)
        self._log_generalization_results(checkpoint_name, generalization_results)
        
        return generalization_results
    
    def _log_generalization_results(self, checkpoint_name: str, generalization_results: Dict[str, Any]):
        """Log generalization evaluation results."""
        training_task = generalization_results.get('training_task', 'unknown')
        avg_accuracy = generalization_results.get('average_generalization_accuracy', 0)
        task_count = len(generalization_results.get('generalization_tasks', []))
        
        logger.info(f"Generalization results - training task: {training_task}, "
                   f"average accuracy: {avg_accuracy:.3f} across {task_count} tasks")
    
    def _save_generalization_metrics(self):
        """Save generalization metrics to file."""
        if not self.generalization_metrics:
            return
        
        metrics_file = os.path.join(self.output_dir, "generalization_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.generalization_metrics.to_dict(), f, indent=2)
        
        self.generalization_metrics.print_summary()

    def _print_cost_summary(self, stage_name: str = ""):
        """Print OpenAI cost summaries for both teacher and student models."""
        teacher_cost = 0.0
        student_cost = 0.0
        
        if self.teacher_pipeline.use_openai_api and hasattr(self.teacher_pipeline, 'openai_client'):
            self.teacher_pipeline.openai_client.print_cost_summary(f"Teacher Model ({stage_name})")
            teacher_cost_info = self.teacher_pipeline.openai_client.calculate_total_cost()
            teacher_cost = teacher_cost_info['total_cost']
        
        # Print student model cost summary if informativeness was run
        if self.informativeness_metrics and hasattr(self.informativeness_metrics, 'student_openai_client'):
            self.informativeness_metrics.student_openai_client.print_cost_summary(f"Student Model ({stage_name})")
            student_cost_info = self.informativeness_metrics.student_openai_client.calculate_total_cost()
            student_cost = student_cost_info['total_cost']
        
        # Print combined total if both models were used
        if teacher_cost > 0 and student_cost > 0:
            total_cost = teacher_cost + student_cost
            print(f"\n🔸 Combined Total Cost ({stage_name}): ${total_cost:.4f} (Teacher: ${teacher_cost:.4f} + Student: ${student_cost:.4f})")


#### --------------- unified evaluation processing --------------- ####
    def _process_single_checkpoint_unified(self, checkpoint_path: str, teacher_data: List[Dict[str, Any]] = None, 
                                         cot_only_mode: bool = False, informativeness_only_mode: bool = False,
                                         generalization_only_mode: bool = False, student_data: List[Dict[str, Any]] = None) -> bool:
        """Process a single checkpoint for both full evaluation and CoT-only modes."""
        checkpoint_name = self._extract_checkpoint_name(checkpoint_path)
        
        try:
            run_cot = cot_only_mode or (self.cot_perturbation_metrics is not None)
            run_informativeness = informativeness_only_mode or (self.informativeness_metrics is not None)
            run_generalization = generalization_only_mode or (self.generalization_metrics is not None)

            logger.info(f"Checkpoint evaluation flags: run_cot={run_cot}, run_informativeness={run_informativeness}, run_generalization={run_generalization}")
            logger.info(f"cot_only_mode={cot_only_mode}, self.cot_perturbation_metrics is not None={self.cot_perturbation_metrics is not None}")
            
            if run_cot or run_informativeness or generalization_only_mode:
                step_num = self.teacher_pipeline._extract_step_number(checkpoint_name)
                teacher_dir = os.path.join(self.output_dir, "teacher", f"step_{step_num}")
                responses_path = os.path.join(teacher_dir, f"teacher_responses_step_{step_num}.json")
            
            if cot_only_mode:
                if not os.path.exists(responses_path):
                    logger.error(f"Teacher responses not found: {responses_path}")
                    return False
                
                with open(responses_path, 'r') as f:
                    teacher_responses = json.load(f)
                
                teacher_responses = self.teacher_pipeline.evaluate_checkpoint_cot_only(
                    checkpoint_path, teacher_responses
                )
                
                with open(responses_path, 'w') as f:
                    json.dump(teacher_responses, f, indent=2)
                
            elif informativeness_only_mode:
                if not os.path.exists(responses_path):
                    logger.error(f"Teacher responses not found: {responses_path}")
                    return False
                
                with open(responses_path, 'r') as f:
                    teacher_responses = json.load(f)
                
                teacher_responses = self.teacher_pipeline.evaluate_checkpoint_informativeness_only(
                    checkpoint_path, teacher_responses, student_data
                )
                
                with open(responses_path, 'w') as f:
                    json.dump(teacher_responses, f, indent=2)
                
            elif generalization_only_mode:
                if not os.path.exists(responses_path):
                    logger.error(f"Teacher responses not found: {responses_path}")
                    return False
                
                with open(responses_path, 'r') as f:
                    teacher_responses = json.load(f)
                
                teacher_responses = self.teacher_pipeline.evaluate_checkpoint_generalization_only(
                    checkpoint_path, teacher_responses
                )
                
                with open(responses_path, 'w') as f:
                    json.dump(teacher_responses, f, indent=2)
                
            else:
                teacher_responses = self.teacher_pipeline.evaluate_checkpoint(
                    checkpoint_path, teacher_data, cleanup_model=True, run_cot=run_cot, run_informativeness=run_informativeness,
                    run_generalization=run_generalization, student_data=student_data
                )
                
                self._process_checkpoint_metrics(teacher_responses, checkpoint_path)
                self._log_performance()
                
                # Print cost summary after teacher evaluation (shows teacher cost)
                if self.teacher_pipeline.use_openai_api:
                    self._print_cost_summary(f"Checkpoint {checkpoint_name}")
            
            if run_cot:
                if not cot_only_mode:
                    save_intermediate = getattr(self.eval_config, 'save_intermediate_results', True)
                    self.teacher_pipeline.save_cot_results(
                        teacher_responses, self.output_dir, checkpoint_path, save_intermediate
                    )
                
                self._process_cot_metrics(checkpoint_name, teacher_responses, responses_path)
            
            if run_informativeness:
                self._process_informativeness_metrics(checkpoint_name, teacher_responses, responses_path)
                # Print cost summary after informativeness processing
                self._print_cost_summary(checkpoint_name)
            
            if run_generalization:
                self._process_generalization_metrics(checkpoint_name, teacher_responses, responses_path)
            
            # Always save teacher responses in all modes
            self.teacher_pipeline.save_teacher_responses(
                teacher_responses, self.output_dir, checkpoint_name
            )
            
            return True
            
        except Exception as e:
            import traceback
            if generalization_only_mode:
                mode_str = "generalization-only"
            elif cot_only_mode:
                mode_str = "CoT-only"
            elif informativeness_only_mode:
                mode_str = "informativeness-only"
            else:
                mode_str = "full evaluation"
            logger.error(f"Failed to process checkpoint {checkpoint_path} in {mode_str} mode: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return False

    def _run_evaluation_unified(self, cot_only_mode: bool = False, informativeness_only_mode: bool = False, generalization_only_mode: bool = False):
        """Unified evaluation method for full, CoT-only, informativeness-only, and generalization-only modes."""
        if generalization_only_mode:
            mode_name = "GENERALIZATION ONLY"
        elif informativeness_only_mode:
            mode_name = "INFORMATIVENESS ONLY"
        elif cot_only_mode:
            mode_name = "COT PERTURBATION ONLY"
        else:
            mode_name = "FULL EVALUATION"
        logger.info(f"=== {mode_name} MODE ===")
        
        # Validate mode requirements
        if cot_only_mode and not self.cot_perturbation_metrics:
            logger.error("CoT perturbation metrics not initialized")
            return
        
        if informativeness_only_mode and not self.informativeness_metrics:
            logger.error("Informativeness metrics not initialized")
            return
        
        if generalization_only_mode and not self.generalization_metrics:
            logger.error("Generalization metrics not initialized")
            return
        
        # Setup for full evaluation mode
        teacher_data = None
        student_data = None
        
        if not cot_only_mode and not informativeness_only_mode and not generalization_only_mode:
            try:
                self.performance_monitor.start_monitoring()
                teacher_data = self._generate_teacher_data()
                
                # Generate student data if informativeness evaluation is enabled
                if self.informativeness_metrics:
                    student_data = self._generate_student_data()
                    
            except Exception as e:
                logger.error(f"Failed to setup full evaluation: {e}")
                return
        elif informativeness_only_mode:
            try:
                # Generate student data for informativeness-only mode
                student_data = self._generate_student_data()
            except Exception as e:
                logger.error(f"Failed to setup informativeness evaluation: {e}")
                return
        
        # Get checkpoints to process
        checkpoint_paths = self._get_checkpoint_paths()
        if not checkpoint_paths:
            logger.error("No checkpoints found")
            return
        
        # Process all checkpoints
        start_step = getattr(self.eval_config, 'start_step', -1)
        processed_count = 0
        
        for i, checkpoint_path in enumerate(checkpoint_paths):
            logger.info(f"Processing checkpoint {i+1}/{len(checkpoint_paths)}")
            
            if not self._should_process_checkpoint(checkpoint_path, start_step):
                continue
            
            if self._process_single_checkpoint_unified(checkpoint_path, teacher_data, cot_only_mode, informativeness_only_mode, generalization_only_mode, student_data):
                processed_count += 1
        
        # Save final results
        try:
            if cot_only_mode:
                if processed_count > 0:
                    self._save_cot_metrics()
                    logger.info(f"Successfully processed CoT perturbation for {processed_count} checkpoints")
                else:
                    logger.error("No checkpoints were successfully processed")
            elif informativeness_only_mode:
                if processed_count > 0:
                    self._save_informativeness_metrics()
                    logger.info(f"Successfully processed informativeness evaluation for {processed_count} checkpoints")
                else:
                    logger.error("No checkpoints were successfully processed")
            elif generalization_only_mode:
                if processed_count > 0:
                    self._save_generalization_metrics()
                    logger.info(f"Successfully processed generalization evaluation for {processed_count} checkpoints")
                else:
                    logger.error("No checkpoints were successfully processed")
            else:
                self._save_final_results()
                logger.info("Teacher evaluation completed successfully")
                
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
            if not cot_only_mode and not informativeness_only_mode and not generalization_only_mode:
                raise
        finally:
            if not cot_only_mode and not informativeness_only_mode and not generalization_only_mode:
                self._cleanup()




#### --------------- calculate metrics --------------- ####

    def _calculate_basic_metrics(self, teacher_responses: List[Dict[str, Any]], checkpoint_path: str) -> Dict[str, Any]:
        """Calculate basic metrics for a checkpoint."""
        teacher_responses = [response for response in teacher_responses if response.get('reward_score', 0.0) > 0]
        total_score = sum(response.get('reward_score', 0.0) for response in teacher_responses)
        total_count = len(teacher_responses)
        average_reward = total_score / total_count if total_count > 0 else 0.0
        
        # Calculate perplexity metrics
        perplexity_values = []
        for response in teacher_responses:
            gen_info = response.get('generation_info', {})
            perplexity = gen_info.get('perplexity', 0.0)
            if perplexity > 0:
                perplexity_values.append(perplexity)
        
        total_mean_perplexity = sum(perplexity_values) / len(perplexity_values) if perplexity_values else 0.0
        
        return {
            'accuracy': average_reward,
            'average_reward': average_reward,
            'total_score': total_score,
            'total_count': total_count,
            'total_mean_perplexity': total_mean_perplexity,
            'perplexity_count': len(perplexity_values),
            'checkpoint_path': checkpoint_path,
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_checkpoint_metrics(self, teacher_responses: List[Dict[str, Any]], checkpoint_path: str):
        """Calculate and store metrics for a checkpoint."""
        checkpoint_name = self._extract_checkpoint_name(checkpoint_path)
        
        # Calculate basic metrics
        metrics = self._calculate_basic_metrics(teacher_responses, checkpoint_path)
        
        # Store teacher accuracy metrics
        self.teacher_accuracy_metrics.add_checkpoint_result(checkpoint_name, metrics)
        
        # Log results
        self._log_checkpoint_results(checkpoint_name, metrics)
    
    
    def _save_final_results(self):
        """Save summary metrics and reports."""
        # Save teacher accuracy metrics
        metrics_file = os.path.join(self.output_dir, "teacher_accuracy_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.teacher_accuracy_metrics.to_dict(), f, indent=2)
        
        self.teacher_accuracy_metrics.print_summary()
        
        # Save CoT perturbation metrics if enabled
        self._save_cot_metrics()
        
        # Save informativeness metrics if enabled
        self._save_informativeness_metrics()
        
        # Save generalization metrics if enabled
        self._save_generalization_metrics()
        
        # Print OpenAI cost summaries
        self._print_cost_summary("Final")
    
    def _cleanup(self):
        """Clean up resources."""
        self.performance_monitor.save_performance_metrics(self.output_dir)
        self.teacher_pipeline.cleanup()
    
    def _should_run_cot_only(self) -> bool:
        """Check if only CoT perturbation should be run."""
        requested_metrics = self.config.get('evaluation', {}).get('metrics', [])
        return (len(self.requested_cot_metrics) > 0 and 
                len([m for m in requested_metrics if m in ['teacher_accuracy', 'perplexity']]) == 0)
    
    def _should_run_informativeness_only(self) -> bool:
        """Check if only informativeness evaluation should be run."""
        requested_metrics = self.config.get('evaluation', {}).get('metrics', [])
        return (len(self.requested_informativeness_metrics) > 0 and 
                len([m for m in requested_metrics if m in ['teacher_accuracy', 'perplexity'] + self.requested_cot_metrics]) == 0)
    
    def _should_run_generalization_only(self) -> bool:
        """Check if only generalization evaluation should be run."""
        requested_metrics = self.config.get('evaluation', {}).get('metrics', [])
        return ('generalization' in requested_metrics and 
                len([m for m in requested_metrics if m in ['teacher_accuracy', 'perplexity'] + self.requested_cot_metrics + self.requested_informativeness_metrics]) == 0)
    
#### --------------- main --------------- ####
    def run_evaluation(self):
        """Main evaluation entry point."""
        logger.info("Starting Teacher Evaluation")
        
        # Determine evaluation mode and run unified evaluation
        cot_only_mode = self._should_run_cot_only()
        informativeness_only_mode = self._should_run_informativeness_only()
        generalization_only_mode = self._should_run_generalization_only()
        
        # Check for conflicting modes
        only_modes = [cot_only_mode, informativeness_only_mode, generalization_only_mode]
        if sum(only_modes) > 1:
            logger.error("Cannot run multiple *-only modes simultaneously")
            return
            
        self._run_evaluation_unified(cot_only_mode, informativeness_only_mode, generalization_only_mode)