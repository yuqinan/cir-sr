#!/usr/bin/env python3
"""
Teacher-Only Evaluation Script

This script runs teacher model evaluation with support for:
- Full pipeline mode (generate teacher responses + optional CoT perturbation)  
- CoT perturbation only mode (load existing responses and add perturbations)

The script optimizes memory usage by reusing loaded models when both perplexity 
and CoT perturbation metrics are requested.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pprint import pprint

import hydra
from omegaconf import OmegaConf
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.checkpoint_loader import CheckpointLoader
from evaluate.utils.data_generator import DataGenerator
from evaluate.utils.teacher_pipeline import TeacherPipeline
from evaluate.utils.teacher_accuracy_metrics import TeacherAccuracyMetrics
from evaluate.utils.performance_monitor import PerformanceMonitor
from evaluate.utils.config_validator import validate_config
from evaluate.utils.cot_perturbation_utils import perturb_teacher_responses
from evaluate.utils.cot_perturbation_metrics import CoTPerturbationMetrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('teacher_eval.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TeacherOnlyEvaluator:
    """
    Evaluator that runs teacher model to generate thinking traces and optionally 
    applies CoT perturbations for robustness testing.
    """
    
    def __init__(self, config):
        """Initialize the teacher-only evaluator with all necessary components."""
        logger.info("Initializing Teacher-Only Evaluator...")
        
        # Validate and optimize configuration
        self.config = validate_config(config)
        self.eval_config = self.config.evaluation
        
        # Initialize core components
        self._initialize_components()
        self._initialize_metrics()
        self._setup_data_generator()
        self._setup_output_directory()
        
        logger.info("Teacher-Only Evaluator initialized successfully")
    
    def _initialize_components(self):
        """Initialize core pipeline components."""
        self.checkpoint_loader = CheckpointLoader(self.eval_config.base_model_path)
        self.teacher_pipeline = TeacherPipeline(OmegaConf.to_container(self.config, resolve=True))
        self.performance_monitor = PerformanceMonitor(OmegaConf.to_container(self.config, resolve=True))
    
    def _initialize_metrics(self):
        """Initialize metrics based on requested evaluation types."""
        # Always initialize teacher accuracy metrics
        self.teacher_accuracy_metrics = TeacherAccuracyMetrics()
        
        # Initialize CoT perturbation metrics if requested
        self.cot_perturbation_metrics = None
        requested_metrics = self.config.get('evaluation', {}).get('metrics', [])
        
        if 'cot_perturbation' in requested_metrics:
            self.cot_perturbation_metrics = CoTPerturbationMetrics()
            logger.info("Initialized CoT perturbation metrics")
    
    def _setup_data_generator(self):
        """Set up teacher data generator with proper configuration."""
        teacher_dataset_config = OmegaConf.to_container(self.eval_config.teacher_dataset, resolve=True)
        
        # Extract task-specific parameters
        task_params = {k: v for k, v in teacher_dataset_config.items() 
                      if k not in ['task_name', 'seed', 'size', 'val_start']}
        
        self.teacher_data_generator = DataGenerator(
            teacher_dataset_config['task_name'],
            task_params
        )
    
    def _setup_output_directory(self):
        """Create output directory for results."""
        self.output_dir = self.eval_config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_teacher_dataset(self) -> List[Dict[str, Any]]:
        """Generate teacher dataset for evaluation."""
        logger.info("Generating teacher dataset...")
        
        teacher_data = self.teacher_data_generator.generate_teacher_dataset(
            self.eval_config.teacher_dataset.seed,
            self.eval_config.teacher_dataset.size,
            getattr(self.eval_config.teacher_dataset, 'val_start', 0)
        )
        
        logger.info(f"Generated {len(teacher_data)} teacher examples")
        return teacher_data
    
    def get_checkpoint_paths(self) -> List[str]:
        """Get list of checkpoint paths to evaluate."""
        # Handle OpenAI API case
        if self.teacher_pipeline.use_openai_api:
            logger.info("Using OpenAI API - no checkpoint paths needed")
            return ["openai_api"]
        
        # Get checkpoints from directory
        checkpoint_dir = self.eval_config.checkpoint_dir
        start_step = getattr(self.eval_config, 'start_step', -1)
        
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        logger.info(f"Start step filter: {start_step}")
        
        checkpoint_paths = self.checkpoint_loader.get_checkpoint_paths(checkpoint_dir, start_step)
        
        # Apply max checkpoints limit if specified
        max_checkpoints = getattr(self.eval_config, 'max_checkpoints', None)
        if max_checkpoints and len(checkpoint_paths) > max_checkpoints:
            step = len(checkpoint_paths) // max_checkpoints
            checkpoint_paths = checkpoint_paths[::step][:max_checkpoints]
        
        logger.info(f"Found {len(checkpoint_paths)} checkpoints to evaluate")
        return checkpoint_paths
    
    def run_teacher_evaluation(self):
        """
        Main evaluation entry point. Determines execution mode and runs appropriate pipeline.
        """
        logger.info("Starting Teacher-Only Evaluation")
        
        # Determine evaluation mode
        evaluation_mode = self._determine_evaluation_mode()
        
        if evaluation_mode == "cot_perturbation_only":
            self._run_cot_perturbation_only_mode()
        else:
            self._run_full_pipeline_mode()
    
    def _determine_evaluation_mode(self) -> str:
        """Determine which evaluation mode to run based on requested metrics."""
        requested_metrics = self.config.get('evaluation', {}).get('metrics', [])
        
        # Check if only CoT perturbation is requested (no fresh teacher generation)
        cot_only = ('cot_perturbation' in requested_metrics and 
                   len([m for m in requested_metrics if m in ['teacher_accuracy', 'perplexity']]) == 0)
        
        if cot_only:
            logger.info("=== COT PERTURBATION ONLY MODE ===")
            return "cot_perturbation_only"
        else:
            logger.info("=== FULL PIPELINE MODE ===")
            return "full_pipeline"
    
    def _run_full_pipeline_mode(self):
        """Run full evaluation pipeline with teacher generation and optional CoT perturbation."""
        self.performance_monitor.start_monitoring()
        
        try:
            # Generate datasets and get checkpoints
            teacher_data = self.generate_teacher_dataset()
            checkpoint_paths = self.get_checkpoint_paths()
            
            if not checkpoint_paths:
                logger.error("No checkpoints found to evaluate")
                return
            
            # Log initial system stats
            self._log_system_stats()
            
            # Process each checkpoint
            self._process_checkpoints(checkpoint_paths, teacher_data)
            
            # Save final results
            self._save_final_results()
            
            logger.info("Teacher evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Teacher evaluation failed: {str(e)}")
            raise
        finally:
            self._cleanup_evaluation()
    
    def _log_system_stats(self):
        """Log initial GPU and memory statistics."""
        self.performance_monitor.log_gpu_stats()
        self.performance_monitor.log_memory_usage()
    
    def _process_checkpoints(self, checkpoint_paths: List[str], teacher_data: List[Dict[str, Any]]):
        """Process all checkpoints with teacher evaluation and optional CoT perturbation."""
        start_step = getattr(self.eval_config, 'start_step', -1)
        
        for i, checkpoint_path in enumerate(checkpoint_paths):
            logger.info(f"Processing checkpoint {i+1}/{len(checkpoint_paths)}")
            
            # Validate checkpoint step if filtering is enabled
            if not self._should_process_checkpoint(checkpoint_path, start_step):
                continue
            
            try:
                self._process_single_checkpoint(checkpoint_path, teacher_data)
                self._log_checkpoint_performance()
                self._cleanup_between_checkpoints()
                
            except Exception as e:
                logger.error(f"Failed to evaluate checkpoint {checkpoint_path}: {str(e)}")
                continue
    
    def _should_process_checkpoint(self, checkpoint_path: str, start_step: int) -> bool:
        """Check if checkpoint should be processed based on step number filtering."""
        if start_step < 0:
            return True  # No filtering
        
        # Extract checkpoint name for step validation
        checkpoint_name = (os.path.basename(os.path.dirname(checkpoint_path)) 
                          if os.path.basename(checkpoint_path) == "actor" 
                          else os.path.basename(checkpoint_path))
        
        checkpoint_step_str = self.teacher_pipeline._extract_step_number(checkpoint_name)
        logger.info(f"Validating checkpoint '{checkpoint_name}' -> step '{checkpoint_step_str}'")
        
        try:
            checkpoint_step = int(checkpoint_step_str)
            if checkpoint_step < start_step:
                logger.warning(f"Skipping checkpoint {checkpoint_path} (step {checkpoint_step} < start_step {start_step})")
                return False
            else:
                logger.info(f"Processing checkpoint {checkpoint_path} (step {checkpoint_step} >= start_step {start_step})")
                return True
        except (ValueError, TypeError):
            logger.warning(f"Could not parse step number '{checkpoint_step_str}', skipping checkpoint")
            return False
    
    def _process_single_checkpoint(self, checkpoint_path: str, teacher_data: List[Dict[str, Any]]):
        """Process a single checkpoint with teacher generation and optional CoT perturbation."""
        # Determine if we need to keep model loaded for CoT perturbation
        skip_cleanup = self.cot_perturbation_metrics is not None
        
        # Generate teacher responses
        teacher_responses = self.teacher_pipeline.evaluate_checkpoint(
            checkpoint_path, teacher_data, cleanup_model=not skip_cleanup
        )
        
        # Run CoT perturbation if requested (reuses loaded model)
        if self.cot_perturbation_metrics:
            teacher_responses = self._run_cot_perturbation_with_current_model(teacher_responses)
            self._save_cot_perturbation_results(teacher_responses, checkpoint_path)
            self._cleanup_after_cot_perturbation()
        
        # Calculate and store metrics
        self._calculate_and_store_metrics(teacher_responses, checkpoint_path)
    
    def _run_cot_perturbation_with_current_model(self, teacher_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run CoT perturbation using the already loaded model."""
        logger.info("Running CoT perturbation with already loaded model...")
        
        # Generate perturbed prompts
        perturbed_data = perturb_teacher_responses(teacher_responses, ["truncate", "filler"])
        
        # Run inference on perturbed data using current model
        for pert_type, pert_responses in perturbed_data["perturbed"].items():
            logger.info(f"Running inference for {pert_type} perturbation ({len(pert_responses)} samples)")
            
            # Use generate_thinking_traces directly (model already loaded)
            pert_results = self.teacher_pipeline.generate_thinking_traces(pert_responses)
            
            # Add perturbation results to original responses
            for original_resp, pert_result in zip(teacher_responses, pert_results):
                perturbation_key = f'{pert_type}_perturbation'
                original_resp[perturbation_key] = {
                    'perturbed_input': pert_result.get('question', ''),
                    'perturbed_output': pert_result.get('teacher_response', ''),
                    'reward_score': pert_result.get('reward_score', 0.0)
                }
        
        logger.info("CoT perturbation inference completed")
        return teacher_responses
    
    def _save_cot_perturbation_results(self, teacher_responses: List[Dict[str, Any]], checkpoint_path: str):
        """Save CoT perturbation results to files."""
        # Use the same checkpoint name extraction logic as the rest of the code
        checkpoint_name = self._extract_checkpoint_name(checkpoint_path)
        
        if getattr(self.eval_config, 'save_intermediate_results', True):
            # Save updated main teacher responses
            self.teacher_pipeline.save_teacher_responses(
                teacher_responses, self.output_dir, checkpoint_name
            )
            
            # Save CoT perturbation results to perturbations folder
            self.teacher_pipeline.save_cot_perturbation_results(
                teacher_responses, self.output_dir, checkpoint_name
            )
    
    def _cleanup_after_cot_perturbation(self):
        """Clean up model after CoT perturbation since we skipped automatic cleanup."""
        logger.info("Manually cleaning up teacher model after CoT perturbation...")
        self.teacher_pipeline._cleanup_current_model()
        self.teacher_pipeline._cleanup_temp_model_files()
    
    def _calculate_and_store_metrics(self, teacher_responses: List[Dict[str, Any]], checkpoint_path: str):
        """Calculate metrics and store results for a checkpoint."""
        # Calculate basic metrics
        metrics = self._calculate_checkpoint_metrics(teacher_responses)
        
        # Extract checkpoint info
        checkpoint_name = self._extract_checkpoint_name(checkpoint_path)
        step_num = self.teacher_pipeline._extract_step_number(checkpoint_name)
        
        # Store teacher accuracy metrics
        self.teacher_accuracy_metrics.add_checkpoint_result(checkpoint_name, metrics)
        
        # Process CoT perturbation metrics if enabled
        if self.cot_perturbation_metrics:
            cot_results = self._process_cot_perturbation_metrics(teacher_responses, step_num)
            self.cot_perturbation_metrics.add_checkpoint_result(checkpoint_name, cot_results)
            self._log_cot_results(checkpoint_name, cot_results)
        
        # Log checkpoint results
        self._log_checkpoint_results(checkpoint_name, metrics)
    
    def _calculate_checkpoint_metrics(self, teacher_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic metrics for a checkpoint."""
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
            'accuracy': average_reward,  # Store reward as accuracy for compatibility
            'average_reward': average_reward,
            'total_score': total_score,
            'total_count': total_count,
            'total_mean_perplexity': total_mean_perplexity,
            'perplexity_count': len(perplexity_values),
            'checkpoint_path': self.teacher_pipeline.current_checkpoint_path,
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_checkpoint_name(self, checkpoint_path: str) -> str:
        """Extract checkpoint name from path."""
        if checkpoint_path == "openai_api":
            return "openai_api"
        else:
            # For step 0 special case and other checkpoints, return the basename
            # The _extract_step_number is used internally by save methods
            if os.path.basename(checkpoint_path) == "actor":
                return os.path.basename(os.path.dirname(checkpoint_path))
            else:
                return os.path.basename(checkpoint_path)
    
    def _process_cot_perturbation_metrics(self, teacher_responses: List[Dict[str, Any]], step_num: str) -> Dict[str, Any]:
        """Process CoT perturbation metrics for a checkpoint."""
        teacher_dir = os.path.join(self.output_dir, "teacher", f"step_{step_num}")
        teacher_responses_path = os.path.join(teacher_dir, f"teacher_responses_step_{step_num}.json")
        
        return self.cot_perturbation_metrics.process_teacher_responses(
            teacher_responses_path, teacher_responses
        )
    
    def _log_cot_results(self, checkpoint_name: str, cot_results: Dict[str, Any]):
        """Log CoT perturbation results."""
        logger.info(f"CoT perturbation results: "
                   f"original={cot_results.get('original_reward', 0):.3f}, "
                   f"truncate={cot_results.get('truncate_reward', 0):.3f}, "
                   f"filler={cot_results.get('filler_reward', 0):.3f}")
    
    def _log_checkpoint_results(self, checkpoint_name: str, metrics: Dict[str, Any]):
        """Log checkpoint evaluation results."""
        logger.info(f"Checkpoint {checkpoint_name} average reward: "
                   f"{metrics['average_reward']:.3f} ({metrics['total_score']:.1f}/{metrics['total_count']})")
        
        if metrics['total_mean_perplexity'] > 0:
            logger.info(f"Checkpoint {checkpoint_name} average perplexity: "
                       f"{metrics['total_mean_perplexity']:.3f} ({metrics['perplexity_count']} responses)")
    
    def _log_checkpoint_performance(self):
        """Log performance stats after checkpoint processing."""
        self.performance_monitor.log_gpu_stats()
        self.performance_monitor.log_memory_usage()
    
    def _cleanup_between_checkpoints(self):
        """Clean up resources between checkpoints."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
    
    def _save_final_results(self):
        """Save summary metrics and generate reports."""
        self.save_summary_metrics()
        
        if self.cot_perturbation_metrics:
            self.save_cot_perturbation_metrics()
    
    def _cleanup_evaluation(self):
        """Clean up resources after evaluation."""
        self.performance_monitor.save_performance_metrics(self.output_dir)
        self.teacher_pipeline.cleanup()
    
    def save_summary_metrics(self):
        """Save teacher accuracy metrics and generate summary report."""
        # Save detailed metrics
        metrics_file = os.path.join(self.output_dir, "teacher_accuracy_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.teacher_accuracy_metrics.to_dict(), f, indent=2)
        
        # Generate and save summary report
        self._generate_teacher_accuracy_report()
        
        # Print summary to console
        self.teacher_accuracy_metrics.print_summary()
    
    def _generate_teacher_accuracy_report(self):
        """Generate detailed teacher accuracy report."""
        summary_stats = self.teacher_accuracy_metrics.calculate_summary_statistics()
        report_file = os.path.join(self.output_dir, "teacher_accuracy_report.txt")
        
        with open(report_file, 'w') as f:
            self._write_report_header(f)
            self._write_summary_statistics(f, summary_stats)
            self._write_individual_results(f)
        
        logger.info(f"Saved teacher accuracy report to {report_file}")
    
    def _write_report_header(self, f):
        """Write report header."""
        f.write("=" * 60 + "\n")
        f.write("TEACHER REWARD EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def _write_summary_statistics(self, f, summary_stats: Dict[str, Any]):
        """Write summary statistics to report."""
        if not summary_stats:
            return
        
        f.write(f"Number of checkpoints evaluated: {summary_stats.get('num_checkpoints', 0)}\n\n")
        f.write("Summary Statistics:\n")
        f.write(f"  Mean reward: {summary_stats['mean_accuracy']:.3f} ± {summary_stats['std_accuracy']:.3f}\n")
        f.write(f"  Reward range: {summary_stats['min_accuracy']:.3f} - {summary_stats['max_accuracy']:.3f}\n")
        f.write(f"  Best checkpoint: {summary_stats['best_checkpoint']['name']} ({summary_stats['best_checkpoint']['accuracy']:.3f})\n")
        f.write(f"  Worst checkpoint: {summary_stats['worst_checkpoint']['name']} ({summary_stats['worst_checkpoint']['accuracy']:.3f})\n\n")
    
    def _write_individual_results(self, f):
        """Write individual checkpoint results to report."""
        f.write("Individual Checkpoint Results:\n")
        for checkpoint_name, results in self.teacher_accuracy_metrics.get_sorted_results():
            total_score = results.get('total_score', 'N/A')
            total_count = results.get('total_count', 'N/A')
            f.write(f"  {checkpoint_name}: {results['accuracy']:.3f} (total: {total_score}/{total_count})\n")
    
    def save_cot_perturbation_metrics(self):
        """Save CoT perturbation metrics and generate report."""
        if not self.cot_perturbation_metrics:
            return
        
        # Save detailed metrics
        metrics_file = os.path.join(self.output_dir, "cot_perturbation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.cot_perturbation_metrics.to_dict(), f, indent=2)
        
        # Generate and save summary report
        self._generate_cot_perturbation_report()
        
        # Print summary to console
        self.cot_perturbation_metrics.print_summary()
    
    def _generate_cot_perturbation_report(self):
        """Generate detailed CoT perturbation report."""
        summary_stats = self.cot_perturbation_metrics.calculate_summary_statistics()
        report_file = os.path.join(self.output_dir, "cot_perturbation_report.txt")
        
        with open(report_file, 'w') as f:
            self._write_cot_report_header(f)
            self._write_cot_summary_statistics(f, summary_stats)
            self._write_cot_individual_results(f)
        
        logger.info(f"Saved CoT perturbation report to {report_file}")
    
    def _write_cot_report_header(self, f):
        """Write CoT perturbation report header."""
        f.write("=" * 60 + "\n")
        f.write("COT PERTURBATION EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def _write_cot_summary_statistics(self, f, summary_stats: Dict[str, Any]):
        """Write CoT perturbation summary statistics."""
        if not summary_stats:
            return
        
        f.write(f"Number of checkpoints evaluated: {summary_stats.get('num_checkpoints', 0)}\n\n")
        
        orig_stats = summary_stats.get('original_reward', {})
        trunc_stats = summary_stats.get('truncate_reward', {})
        filler_stats = summary_stats.get('filler_reward', {})
        
        f.write("Summary Statistics:\n")
        f.write(f"  Original Mean Reward: {orig_stats.get('mean', 0):.3f} ± {orig_stats.get('std', 0):.3f}\n")
        f.write(f"  Truncate Mean Reward: {trunc_stats.get('mean', 0):.3f} ± {trunc_stats.get('std', 0):.3f}\n")
        f.write(f"  Filler Mean Reward: {filler_stats.get('mean', 0):.3f} ± {filler_stats.get('std', 0):.3f}\n\n")
        
        f.write("  Average Reward Drops:\n")
        f.write(f"    Truncate: {orig_stats.get('mean', 0) - trunc_stats.get('mean', 0):.3f}\n")
        f.write(f"    Filler: {orig_stats.get('mean', 0) - filler_stats.get('mean', 0):.3f}\n\n")
    
    def _write_cot_individual_results(self, f):
        """Write individual CoT perturbation results."""
        f.write("Individual Checkpoint Results:\n")
        for checkpoint_name, results in self.cot_perturbation_metrics.get_sorted_results():
            f.write(f"  {checkpoint_name}:\n")
            f.write(f"    Original: {results.get('original_reward', 0):.3f}\n")
            f.write(f"    Truncate: {results.get('truncate_reward', 0):.3f} "
                   f"(drop: {results.get('original_reward', 0) - results.get('truncate_reward', 0):.3f})\n")
            f.write(f"    Filler: {results.get('filler_reward', 0):.3f} "
                   f"(drop: {results.get('original_reward', 0) - results.get('filler_reward', 0):.3f})\n")
    
    def _run_cot_perturbation_only_mode(self):
        """Run CoT perturbation on existing teacher responses without regenerating them."""
        if not self.cot_perturbation_metrics:
            logger.error("CoT perturbation metrics not initialized")
            return
        
        checkpoint_paths = self.get_checkpoint_paths()
        if not checkpoint_paths:
            logger.error("No checkpoints found to evaluate")
            return
        
        processed_count = self._process_existing_responses(checkpoint_paths)
        
        if processed_count > 0:
            self.save_cot_perturbation_metrics()
            logger.info(f"Successfully processed CoT perturbation for {processed_count} checkpoints")
        else:
            logger.error("No checkpoints were successfully processed for CoT perturbation")
    
    def _process_existing_responses(self, checkpoint_paths: List[str]) -> int:
        """Process existing teacher responses for CoT perturbation."""
        start_step = getattr(self.eval_config, 'start_step', -1)
        processed_count = 0
        
        for i, checkpoint_path in enumerate(checkpoint_paths):
            logger.info(f"Processing checkpoint {i+1}/{len(checkpoint_paths)} for CoT perturbation")
            
            try:
                checkpoint_info = self._extract_checkpoint_info(checkpoint_path)
                
                # Validate checkpoint step if filtering enabled
                if not self._validate_checkpoint_step(checkpoint_info, start_step):
                    continue
                
                # Process if teacher responses exist
                if self._process_checkpoint_cot_perturbation(checkpoint_info):
                    processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process CoT perturbation for checkpoint {checkpoint_path}: {str(e)}")
                continue
        
        return processed_count
    
    def _extract_checkpoint_info(self, checkpoint_path: str) -> Dict[str, str]:
        """Extract checkpoint information for processing."""
        if checkpoint_path == "openai_api":
            return {
                'checkpoint_name': "openai_api",
                'step_num': "openai",
                'checkpoint_path': checkpoint_path
            }
        
        # Handle step number extraction
        if os.path.basename(checkpoint_path) == "actor":
            checkpoint_name = os.path.basename(os.path.dirname(checkpoint_path))
        else:
            checkpoint_name = os.path.basename(checkpoint_path)
        
        step_num = self.teacher_pipeline._extract_step_number(checkpoint_name)
        
        return {
            'checkpoint_name': checkpoint_name,
            'step_num': step_num,
            'checkpoint_path': checkpoint_path
        }
    
    def _validate_checkpoint_step(self, checkpoint_info: Dict[str, str], start_step: int) -> bool:
        """Validate checkpoint step against start_step filter."""
        if start_step < 0:
            return True
        
        try:
            checkpoint_step = int(checkpoint_info['step_num'])
            if checkpoint_step < start_step:
                logger.warning(f"Skipping checkpoint {checkpoint_info['checkpoint_path']} "
                             f"(step {checkpoint_step} < start_step {start_step})")
                return False
            return True
        except (ValueError, TypeError):
            logger.warning(f"Could not parse step number '{checkpoint_info['step_num']}', skipping")
            return False
    
    def _process_checkpoint_cot_perturbation(self, checkpoint_info: Dict[str, str]) -> bool:
        """Process CoT perturbation for a single checkpoint."""
        # Check for existing teacher responses
        teacher_dir = os.path.join(self.output_dir, "teacher", f"step_{checkpoint_info['step_num']}")
        teacher_responses_path = os.path.join(teacher_dir, f"teacher_responses_step_{checkpoint_info['step_num']}.json")
        
        if not os.path.exists(teacher_responses_path):
            logger.error(f"Teacher responses file not found: {teacher_responses_path}")
            logger.error("Cannot run CoT perturbation without existing teacher responses")
            return False
        
        # Load existing teacher responses
        logger.info(f"Loading existing teacher responses from {teacher_responses_path}")
        with open(teacher_responses_path, 'r') as f:
            teacher_responses = json.load(f)
        
        # Run CoT perturbation
        logger.info("Loading checkpoint for CoT perturbation...")
        teacher_responses_with_perturb = self._run_cot_perturbation_standalone(
            checkpoint_info['checkpoint_path'], teacher_responses
        )
        
        # Process and save results
        cot_results = self.cot_perturbation_metrics.process_teacher_responses(
            teacher_responses_path, teacher_responses_with_perturb
        )
        self.cot_perturbation_metrics.add_checkpoint_result(checkpoint_info['checkpoint_name'], cot_results)
        
        # Save updated results back to JSON
        with open(teacher_responses_path, 'w') as f:
            json.dump(teacher_responses_with_perturb, f, indent=2)
        
        logger.info(f"CoT perturbation results for {checkpoint_info['checkpoint_name']}: "
                   f"original={cot_results.get('original_reward', 0):.3f}, "
                   f"truncate={cot_results.get('truncate_reward', 0):.3f}, "
                   f"filler={cot_results.get('filler_reward', 0):.3f}")
        
        return True
    
    def _run_cot_perturbation_standalone(self, checkpoint_path: str, teacher_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run CoT perturbation by loading a new model (for CoT-only mode)."""
        logger.info("Running CoT perturbation inference with new model...")
        
        # Generate perturbed prompts
        perturbed_data = perturb_teacher_responses(teacher_responses, ["truncate", "filler"])
        
        # Initialize temporary teacher pipeline for perturbation inference  
        temp_teacher_pipeline = TeacherPipeline(OmegaConf.to_container(self.config, resolve=True))
        
        try:
            # Run inference on perturbed data
            for pert_type, pert_responses in perturbed_data["perturbed"].items():
                logger.info(f"Running inference for {pert_type} perturbation ({len(pert_responses)} samples)")
                
                # Evaluate perturbed responses
                pert_results = temp_teacher_pipeline.evaluate_checkpoint(checkpoint_path, pert_responses)
                
                # Add perturbation results to original responses
                for original_resp, pert_result in zip(teacher_responses, pert_results):
                    perturbation_key = f'{pert_type}_perturbation'
                    original_resp[perturbation_key] = {
                        'perturbed_input': pert_result.get('question', ''),
                        'perturbed_output': pert_result.get('teacher_response', ''),
                        'reward_score': pert_result.get('reward_score', 0.0)
                    }
            
            logger.info("CoT perturbation inference completed")
            return teacher_responses
            
        except Exception as e:
            logger.error(f"Error during CoT perturbation inference: {e}")
            raise
        finally:
            # Clean up temporary teacher pipeline
            temp_teacher_pipeline.cleanup()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@hydra.main(config_path="configs", config_name="mini_sudoku", version_base=None)
def main(config):
    """Main entry point for teacher-only evaluation."""
    # Print resolved configuration
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    
    # Initialize evaluator and run evaluation
    evaluator = TeacherOnlyEvaluator(config)
    evaluator.run_teacher_evaluation()


if __name__ == "__main__":
    main()