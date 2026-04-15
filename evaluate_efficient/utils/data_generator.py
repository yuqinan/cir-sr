import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import reasoning_gym
import logging

logger = logging.getLogger(__name__)

class DataGenerator:
    """
    Utility class for generating datasets with different random seeds for teacher and student models.
    Based on the reasoning gym task system and config patterns from trainers/.
    """
    
    def __init__(self, task_name: str, task_params: Dict[str, Any]):
        """
        Initialize the data generator.
        
        Args:
            task_name: Name of the task (e.g., 'mini_sudoku', 'spiral_matrix')
            task_params: Task-specific parameters from config
        """
        self.task_name = task_name
        self.task_params = task_params
        
    def generate_teacher_dataset(self, teacher_seed: int, size: int, val_start: int = 0) -> List[Dict[str, Any]]:
        """
        Generate dataset for teacher model evaluation.
        
        Args:
            teacher_seed: Random seed for teacher dataset
            size: Number of examples to generate
            val_start: Starting index for validation data (default: 0)
            
        Returns:
            List of dataset items for teacher evaluation
        """
        logger.info(f"Generating teacher dataset for {self.task_name} with seed={teacher_seed}, size={size}, val_start={val_start}")
        
        # Generate dataset using reasoning gym
        data = reasoning_gym.create_dataset(
            self.task_name,
            seed=teacher_seed,
            size=size,
            **self.task_params
        )
        
        # Convert to format expected by evaluation system
        teacher_data = []
        for i, item in enumerate(data):
            # Skip items before val_start index
            if i < val_start:
                continue
                
            teacher_data.append({
                'index': i,
                'question': item['question'],
                'answer': item['answer'],
                'metadata': item.get('metadata', {}),
                'data_source': self.task_name,
                'seed': teacher_seed
            })
        
        logger.info(f"Generated {len(teacher_data)} teacher examples (starting from index {val_start})")
        return teacher_data
    
    def generate_student_dataset(self, student_seed: int, size: int, val_start: int = 0) -> List[Dict[str, Any]]:
        """
        Generate dataset for student model evaluation.
        
        Args:
            student_seed: Random seed for student dataset  
            size: Number of examples to generate
            
        Returns:
            List of dataset items for student evaluation
        """
        logger.info(f"Generating student dataset for {self.task_name} with seed={student_seed}, size={size}")
        
        # Generate dataset using reasoning gym
        data = reasoning_gym.create_dataset(
            self.task_name,
            seed=student_seed,
            size=size,
            **self.task_params
        )
        
        # Convert to format expected by evaluation system
        student_data = []
        for i, item in enumerate(data):
            if i < val_start:
                continue
                
            student_data.append({
                'index': i,
                'question': item['question'],
                'answer': item['answer'],
                'metadata': item.get('metadata', {}),
                'data_source': self.task_name,
                'seed': student_seed
            })
        
        logger.info(f"Generated {len(student_data)} student examples")
        return student_data
    
    def generate_few_shot_examples(self, n_shot_seed: int, n_shot: int) -> List[Dict[str, Any]]:
        """
        Generate few-shot examples for student model.
        
        Args:
            n_shot_seed: Random seed for few-shot examples
            n_shot: Number of few-shot examples to generate
            
        Returns:
            List of few-shot examples
        """
        logger.info(f"Generating {n_shot} few-shot examples for {self.task_name} with seed={n_shot_seed}")
        
        # Generate dataset using reasoning gym
        data = reasoning_gym.create_dataset(
            self.task_name,
            seed=n_shot_seed,
            size=n_shot,
            **self.task_params
        )
        
        # Convert to format expected by evaluation system
        examples = []
        for i, item in enumerate(data):
            examples.append({
                'index': i,
                'question': item['question'],
                'answer': item['answer'],
                'metadata': item.get('metadata', {}),
                'data_source': self.task_name,
                'seed': n_shot_seed
            })
        
        logger.info(f"Generated {len(examples)} few-shot examples")
        return examples
    
    def generate_paired_datasets(self, teacher_seed: int, student_seed: int, size: int, val_start: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate paired datasets for teacher and student with different seeds.
        
        Args:
            teacher_seed: Random seed for teacher dataset
            student_seed: Random seed for student dataset
            size: Number of examples to generate for each dataset
            val_start: Starting index for validation data (default: 0)
            
        Returns:
            Tuple of (teacher_data, student_data)
        """
        logger.info(f"Generating paired datasets for {self.task_name} - teacher_seed={teacher_seed}, student_seed={student_seed}, size={size}, val_start={val_start}")
        
        teacher_data = self.generate_teacher_dataset(teacher_seed, size, val_start)
        student_data = self.generate_student_dataset(student_seed, size)
        
        return teacher_data, student_data
    
    def create_few_shot_prompt(self, few_shot_examples: List[Dict[str, Any]], 
                              question: str, 
                              include_thinking: bool = True) -> str:
        """
        Create a few-shot prompt with examples.
        
        Args:
            few_shot_examples: List of few-shot examples
            question: Question to answer
            include_thinking: Whether to include thinking traces in examples
            
        Returns:
            Formatted few-shot prompt
        """
        prompt = f"Here are some examples of solving {self.task_name} problems:\n\n"
        
        for i, example in enumerate(few_shot_examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Question: {example['question']}\n"
            
            if include_thinking:
                prompt += f"<think>\nLet me solve this step by step.\n</think>\n"
            
            prompt += f"<answer>{example['answer']}</answer>\n\n"
        
        prompt += f"Now solve this new problem:\n"
        prompt += f"Question: {question}\n"
        
        if include_thinking:
            prompt += f"<think>\nLet me solve this step by step.\n</think>\n"
        
        prompt += f"<answer>"
        
        return prompt