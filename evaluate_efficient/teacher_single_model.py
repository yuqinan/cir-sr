"""
Teacher Single Model Inference

Orchestrates teacher model evaluation using modular metrics system.
Handles model loading/cleanup and coordinates between accuracy, usefulness, and informativeness metrics.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .utils.vllm_model_manager import VLLMModelManager
from .utils.prompt_manager import PromptManager
from .utils.checkpoint_loader import CheckpointLoader
from .utils.openai_client import OpenAIClient
from .metrics import AccuracyMetric, UsefulnessMetric, InformativenessMetric, GeneralizationMetric
from trainers.reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)


class TeacherSingleModel:
    """Orchestrates teacher model evaluation using modular metrics system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the teacher pipeline."""
        self.config = config
        self.teacher_config = config['evaluation']['teacher_model']
        self.batch_size = config['evaluation']['batch_size']
        
        # Initialize OpenAI or local model components
        self.use_openai_api = self.teacher_config.get('use_openai_api', False)
        
        if self.use_openai_api:
            self._init_openai_client()
        else:
            self._init_local_model_components()
        
        # Initialize shared dependencies
        self._init_shared_dependencies()
        
        # Initialize metrics
        self._init_metrics()
        
        logger.info("Teacher single model initialized successfully")
    
    def _init_openai_client(self):
        """Initialize OpenAI API client."""
        model_name = self.teacher_config.get('openai_model_name', 'gpt-4o-mini')
        api_key = self.teacher_config.get('openai_api_key', None)
        self.openai_client = OpenAIClient(model_name, api_key)
        self.model_manager = None
        logger.info(f"Initialized OpenAI client: {model_name}")
    
    def _init_local_model_components(self):
        """Initialize local model components."""
        checkpoint_loader = CheckpointLoader(self.config['evaluation']['base_model_path'])
        vllm_config = self.config['evaluation']['vllm']
        
        self.model_manager = VLLMModelManager(vllm_config, checkpoint_loader)
        self.openai_client = None
    
    def _init_shared_dependencies(self):
        """Initialize shared dependencies for metrics."""
        # Initialize prompt manager
        import data.template
        developer_prompt_key = self.teacher_config.get('developer_prompt', 'DeepSeekZero')
        developer_prompt = data.template.SYSTEM_PROMPTS[developer_prompt_key]
        self.prompt_manager = PromptManager(self.teacher_config, developer_prompt)
        
        # Initialize reward calculator
        self.task_name = self.config['evaluation']['teacher_dataset']['task_name']
        try: 
            reward_partial = self.config['evaluation']['reward_partial']
        except:
            reward_partial = False
        self.reward_calculator = RewardCalculator(task="reasoning_gym", task_type=self.task_name, reward_partial=reward_partial)
    
    def _init_metrics(self):
        """Initialize metrics based on requested evaluation types."""
        # Initialize all metrics with shared dependencies
        self.accuracy_metric = AccuracyMetric(self.config, self.reward_calculator, self.prompt_manager)
        self.usefulness_metric = UsefulnessMetric(self.config, self.reward_calculator, self.prompt_manager)
        self.informativeness_metric = InformativenessMetric(self.config, self.reward_calculator, self.prompt_manager)
        self.generalization_metric = GeneralizationMetric(self.config, self.reward_calculator, self.prompt_manager)
        
        logger.info("Initialized all metrics")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint for local model inference."""
        if self.use_openai_api:
            logger.info("Using OpenAI API - no checkpoint loading needed")
            return
        
        self.model_manager.load_model(checkpoint_path)
    
    def generate_response(self, teacher_data: List[Dict[str, Any]], own_thinking: bool = True) -> List[Dict[str, Any]]:
        """Generate responses for teacher dataset using accuracy metric."""
        return self.accuracy_metric.evaluate(
            teacher_data,
            model_manager=self.model_manager,
            openai_client=self.openai_client,
            own_thinking=own_thinking
        )
    
    def run_cot_perturbation(self, checkpoint_path: str, teacher_responses: List[Dict[str, Any]], need_load: bool = False) -> List[Dict[str, Any]]:
        """Run CoT perturbation on existing teacher responses."""
        if not self.usefulness_metric.can_run():
            logger.warning("CoT perturbation requested but no CoT metrics configured")
            return teacher_responses
        
        # Load checkpoint if needed for CoT-only mode
        if not self.use_openai_api and need_load:
            self.load_checkpoint(checkpoint_path)
        
        return self.usefulness_metric.evaluate(
            teacher_responses,
            model_manager=self.model_manager,
            openai_client=self.openai_client,
            checkpoint_path=checkpoint_path,
            accuracy_metric=self.accuracy_metric
        )
    
    def run_informativeness_evaluation(self, teacher_responses: List[Dict[str, Any]], student_data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run informativeness evaluation on teacher responses."""
        if not self.informativeness_metric.can_run():
            logger.warning("Informativeness evaluation requested but not configured")
            return teacher_responses
        
        return self.informativeness_metric.evaluate(
            teacher_responses,
            model_manager=self.model_manager,
            openai_client=self.openai_client,
            student_data=student_data
        )
    
    def run_generalization_evaluation(self, teacher_responses: List[Dict[str, Any]], checkpoint_path: str = None, need_load: bool = False) -> List[Dict[str, Any]]:
        """Run generalization evaluation on teacher responses."""
        if not self.generalization_metric.can_run():
            logger.warning("Generalization evaluation requested but not configured")
            return teacher_responses
        
        # Load checkpoint if needed for generalization-only mode
        if not self.use_openai_api and need_load and checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        return self.generalization_metric.evaluate(
            teacher_responses,
            model_manager=self.model_manager,
            openai_client=self.openai_client
        )
    
    def evaluate_checkpoint(self, checkpoint_path: str, teacher_data: List[Dict[str, Any]], 
                           cleanup_model: bool = True, run_cot: bool = False, run_informativeness: bool = False,
                           run_generalization: bool = False, student_data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Evaluate a single checkpoint with optional CoT perturbation, informativeness, and generalization evaluation."""
        try:
            # Load checkpoint and generate initial responses
            if not self.use_openai_api:
                self.load_checkpoint(checkpoint_path)
            
            teacher_responses = self.generate_response(teacher_data)
            
            # Run CoT perturbation if requested
            if run_cot and self.usefulness_metric.can_run():
                teacher_responses = self.run_cot_perturbation(checkpoint_path, teacher_responses, need_load=False)
            
            # Run informativeness evaluation after CoT metrics if requested
            if run_informativeness and self.informativeness_metric.can_run():
                teacher_responses = self.run_informativeness_evaluation(teacher_responses, student_data)
            
            # Run generalization evaluation if requested
            if run_generalization and self.generalization_metric.can_run():
                teacher_responses = self.run_generalization_evaluation(teacher_responses, checkpoint_path, need_load=False)
            
            return teacher_responses
            
        finally:
            if not self.use_openai_api and cleanup_model:
                self.cleanup()
    
    def evaluate_checkpoint_cot_only(self, checkpoint_path: str, existing_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run only CoT perturbation on existing responses (for CoT-only mode)."""
        try:
            return self.run_cot_perturbation(checkpoint_path, existing_responses, need_load=True)
        finally:
            # Always cleanup after CoT-only processing
            if not self.use_openai_api:
                self.cleanup()
    
    def evaluate_checkpoint_informativeness_only(self, checkpoint_path: str, existing_responses: List[Dict[str, Any]], 
                                                student_data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run only informativeness evaluation on existing responses (for informativeness-only mode)."""
        try:
            return self.run_informativeness_evaluation(existing_responses, student_data)
        finally:
            # No cleanup needed for informativeness (uses external OpenAI API)
            pass
    
    def evaluate_checkpoint_generalization_only(self, checkpoint_path: str, existing_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run only generalization evaluation on existing responses (for generalization-only mode)."""
        try:
            return self.run_generalization_evaluation(existing_responses, checkpoint_path, need_load=True)
        finally:
            # Always cleanup after generalization-only processing for local models
            if not self.use_openai_api:
                self.cleanup()
    
    def save_teacher_responses(self, teacher_responses: List[Dict[str, Any]], 
                              output_dir: str, checkpoint_name: str) -> str:
        """Save teacher responses to structured directory."""
        step_num = self._extract_step_number(checkpoint_name)
        logger.info(f"Checkpoint name: {checkpoint_name}")
        logger.info(f"Step number: {step_num}")
        teacher_dir = os.path.join(output_dir, "teacher", f"step_{step_num}")
        os.makedirs(teacher_dir, exist_ok=True)
        
        # Separate main responses from perturbations and informativeness data
        main_responses = []
        perturbation_data = {}
        informativeness_data = []
        student_baseline_data = []
        generalization_data = []
        
        for response in teacher_responses:
            # Check if this response has perturbation data
            has_perturbations = any(key.endswith('_perturbation') for key in response.keys())
            # Check if this response has informativeness data
            has_informativeness = 'informativeness_detailed_results' in response
            # Check if this response has student baseline data
            has_student_baseline = 'student_baseline_detailed_results' in response
            # Check if this response has generalization data
            has_generalization = 'generalization_detailed_results' in response
            
            if has_perturbations or has_informativeness or has_student_baseline or has_generalization:
                # Include perturbation data but remove detailed informativeness results from main response
                main_response = response.copy()
                
                # Extract informativeness data for separate file
                if has_informativeness:
                    detailed_results = main_response.pop('informativeness_detailed_results', [])
                    informativeness_data.extend(detailed_results)
                
                # Extract student baseline data for separate file
                if has_student_baseline:
                    baseline_results = main_response.pop('student_baseline_detailed_results', [])
                    student_baseline_data.extend(baseline_results)
                
                # Extract generalization data for separate file
                if has_generalization:
                    gen_results = main_response.pop('generalization_detailed_results', [])
                    generalization_data.extend(gen_results)
                
                main_responses.append(main_response)
                
                # Extract perturbation data for separate files
                for key, value in response.items():
                    if key.endswith('_perturbation'):
                        pert_type = key.replace('_perturbation', '')
                        if pert_type not in perturbation_data:
                            perturbation_data[pert_type] = []
                        
                        if pert_type == "expert_thinking":
                            # Handle multi-expert format - create separate entries for each expert model
                            expert_rewards = value.get('reward_scores', {})
                            expert_inputs = value.get('perturbed_inputs', {})
                            expert_outputs = value.get('perturbed_outputs', {})
                            expert_perplexities = value.get('expert_perplexity', {})
                            expert_logprobs = value.get('expert_token_logprobs', {})
                            
                            for expert_model in expert_rewards.keys():
                                pert_entry = {
                                    'index': response['index'],
                                    'original_question': response['question'],
                                    'original_response': response['teacher_response'],
                                    'original_reward': response['reward_score'],
                                    'original_answer': response['teacher_answer'],
                                    'perturbed_input': expert_inputs.get(expert_model, ''),
                                    'perturbed_output': expert_outputs.get(expert_model, ''),
                                    'perturbed_reward': expert_rewards.get(expert_model, 0.0),
                                    'perturbation_type': pert_type,  # Keep as 'expert_thinking' for grouping
                                    'expert_model': expert_model,
                                    'expert_perplexity': expert_perplexities.get(expert_model, 0.0),
                                    'expert_token_logprobs': expert_logprobs.get(expert_model, [])
                                }
                                perturbation_data[pert_type].append(pert_entry)
                        else:
                            # Handle regular perturbations (non-expert)
                            # Get k_idx if available to fetch the correct k response
                            k_idx = value.get('k_idx', 0)

                            # Get teacher_response, reward, and answer from the appropriate k response
                            if 'k_responses' in response and response['k_responses'] and k_idx < len(response['k_responses']):
                                k_response = response['k_responses'][k_idx]
                                teacher_response = k_response.get('teacher_response', '')
                                original_reward = k_response.get('reward_score', 0)
                                teacher_answer = k_response.get('teacher_answer', '')
                            else:
                                # Fallback to top-level fields for old format
                                teacher_response = response.get('teacher_response', '')
                                original_reward = response.get('reward_score', 0)
                                teacher_answer = response.get('teacher_answer', '')

                            pert_entry = {
                                'index': response['index'],
                                'k_idx': k_idx,
                                'original_question': response['question'],
                                'original_response': teacher_response,
                                'original_reward': original_reward,
                                'original_answer': teacher_answer,
                                'perturbed_input': value['perturbed_input'],
                                'perturbed_output': value['perturbed_output'],
                                'perturbed_reward': value['reward_score'],
                                'perturbation_type': pert_type,
                            }
                            perturbation_data[pert_type].append(pert_entry)
            else:
                # No perturbations, just add the response as-is
                main_responses.append(response)
        
        # Save main teacher responses
        filename = f"teacher_responses_step_{step_num}.json"
        filepath = os.path.join(teacher_dir, filename)
        
        # Save perturbation data if exists
        if perturbation_data:
            perturbations_dir = os.path.join(teacher_dir, "perturbations")
            os.makedirs(perturbations_dir, exist_ok=True)
            
            for pert_type, pert_responses in perturbation_data.items():
                pert_filename = f"perturbation_{pert_type}_step_{step_num}.json"
                pert_filepath = os.path.join(perturbations_dir, pert_filename)
                
                with open(pert_filepath, 'w') as f:
                    json.dump(pert_responses, f, indent=2)
                
                logger.info(f"Saved {len(pert_responses)} {pert_type} perturbation responses to {pert_filepath}")

        # Save informativeness data if exists - separate by type
        if informativeness_data:
            informativeness_dir = os.path.join(teacher_dir, "informativeness")
            os.makedirs(informativeness_dir, exist_ok=True)
            
            # Group by informativeness type
            data_by_type = {}
            for item in informativeness_data:
                inf_type = item.get('informativeness_type', 'explanation_demo')
                if inf_type not in data_by_type:
                    data_by_type[inf_type] = []
                data_by_type[inf_type].append(item)
            
            # Save separate files for each type
            for inf_type, type_data in data_by_type.items():
                inf_filename = f"{inf_type}_step_{step_num}.json"
                inf_filepath = os.path.join(informativeness_dir, inf_filename)
                
                with open(inf_filepath, 'w') as f:
                    json.dump(type_data, f, indent=2)
                
                logger.info(f"Saved {len(type_data)} {inf_type} results to {inf_filepath}")
        
        # Save student baseline data if exists
        if student_baseline_data:
            informativeness_dir = os.path.join(teacher_dir, "informativeness")
            os.makedirs(informativeness_dir, exist_ok=True)
            
            baseline_filename = f"student_response_baseline_step_{step_num}.json"
            baseline_filepath = os.path.join(informativeness_dir, baseline_filename)
            
            with open(baseline_filepath, 'w') as f:
                json.dump(student_baseline_data, f, indent=2)
            
            logger.info(f"Saved {len(student_baseline_data)} student baseline responses to {baseline_filepath}")
        
        # Save generalization data if exists
        if generalization_data:
            generalization_dir = os.path.join(teacher_dir, "generalization")
            os.makedirs(generalization_dir, exist_ok=True)
            
            # Group by task name
            data_by_task = {}
            for item in generalization_data:
                task_name = item.get('task', 'unknown')
                if task_name not in data_by_task:
                    data_by_task[task_name] = []
                data_by_task[task_name].append(item)
            
            # Save separate files for each task
            for task_name, task_data in data_by_task.items():
                gen_filename = f"{task_name}_step_{step_num}.json"
                gen_filepath = os.path.join(generalization_dir, gen_filename)
                
                with open(gen_filepath, 'w') as f:
                    json.dump(task_data, f, indent=2)
                
                logger.info(f"Saved {len(task_data)} {task_name} generalization results to {gen_filepath}")

        with open(filepath, 'w') as f:
            json.dump(main_responses, f, indent=2)
        
        logger.info(f"Saved {len(main_responses)} main responses to {filepath}")
        
        return filepath
    
    def save_cot_results(self, teacher_responses: List[Dict[str, Any]], output_dir: str, 
                        checkpoint_path: str, save_intermediate_results: bool = True) -> str:
        """Save CoT perturbation results."""
        if save_intermediate_results:
            checkpoint_name = self._extract_checkpoint_name_from_path(checkpoint_path)
            return self.save_teacher_responses(teacher_responses, output_dir, checkpoint_name)
        return ""
    
    def _extract_checkpoint_name_from_path(self, checkpoint_path: str) -> str:
        """Extract checkpoint name from path."""
        if checkpoint_path == "openai_api":
            return "openai_api"
        elif os.path.basename(checkpoint_path) == "actor":
            return os.path.basename(os.path.dirname(checkpoint_path))
        else:
            return os.path.basename(checkpoint_path)
    
    def _extract_step_number(self, checkpoint_name: str) -> str:
        """Extract step number from checkpoint name.
        
        Returns "0" if checkpoint_dir is same as base_model_path (base model evaluation)
        or if this is an OpenAI API model.
        """
        import re
        
        # Handle OpenAI API models
        if checkpoint_name == "openai_api":
            return "0"
        
        # Check if this is the base model (checkpoint_dir same as base_model_path)
        base_model_path = self.config.get('evaluation', {}).get('base_model_path', '')
        checkpoint_dir = self.config.get('evaluation', {}).get('checkpoint_dir', '')
        
        # Normalize paths for comparison
        if base_model_path and checkpoint_dir:
            base_model_path = os.path.normpath(base_model_path)
            checkpoint_dir = os.path.normpath(checkpoint_dir)
            
            if base_model_path == checkpoint_dir:
                return "0"
        
        patterns = [r'global_step_(\d+)', r'step_(\d+)', r'(\d+)']
        
        #logger.info(f"Checkpoint name in extract_step_number: {checkpoint_name}")
        for pattern in patterns:
            match = re.search(pattern, checkpoint_name)
            if match:
                #logger.info(f"Match: {match.group(1)}")
                return match.group(1)
        
        return checkpoint_name
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if not self.use_openai_api and hasattr(self, 'model_manager') and self.model_manager:
            self.model_manager.cleanup_current_model()
        
        logger.info("Teacher single model cleanup completed")