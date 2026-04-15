import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Any, Optional, Tuple
import reasoning_gym
import random
import json
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import from data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tasks.spiral_matrix import SpiralMatrixTask
from data.tasks.reasoning_gym_task import ReasoningGymTask
from data.template import COT_TEMPLATE
from reward_calculator import RewardCalculator

def transform_to_new_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    
    transformed_data = []
    for item in data:
        transformed_item = {
            'prompt': item['question'],
            'ground_truth': item['expected_output']
        }
        transformed_data.append(transformed_item)
    return transformed_data

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:

    # Extract different fields from the batch
    prompts = [item['prompt'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]
    
    print(prompts)
    return {
        'prompt': prompts,
        'ground_truth': ground_truths,
    }


class ReasoningGymTaskDataset(Dataset):
    """Dataset class for reasoning tasks like spiral matrix."""
    
    def __init__(self, task_type: str, config: Dict[str, Any], split: str = "train", tokenizer=None):
        """
        Initialize the dataset.
        
        Args:
            task_type: Type of reasoning task (e.g., "spiral_matrix")
            config: Configuration dictionary containing task parameters
            split: Dataset split ("train", "val", "test")
            tokenizer: Tokenizer to apply chat template (optional)
        """
        self.task_type = task_type
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        
        # Initialize task handler
        self.task_handler = ReasoningGymTask(task_type)
        
        # Load data based on task type
        raw_data = self._load_data()
        
        # Transform to new format
        self.data = transform_to_new_format(raw_data)
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data for the specific task using the abstracted function."""
        return self.task_handler.load_processed_data(self.task_type, self.config, self.split, self.tokenizer)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

def create_data_loaders(config: Dict[str, Any], tokenizer=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        config: Training configuration dictionary
        tokenizer: Tokenizer to apply chat template (optional)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    task_type = config.get("task_type", "spiral_matrix")
    
    # Create full dataset
    full_dataset = ReasoningGymTaskDataset(task_type, config, split="full", tokenizer=tokenizer)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * config.get("train_split", 0.8))
    val_size = int(total_size * config.get("val_split", 0.1))
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.get("data_seed", 42))
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=config.get("dataloader_num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("eval_batch_size", 8),
        shuffle=False,
        num_workers=config.get("dataloader_num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("eval_batch_size", 8),
        shuffle=False,
        num_workers=config.get("dataloader_num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader 