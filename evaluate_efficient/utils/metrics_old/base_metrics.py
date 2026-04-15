import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class BaseMetrics:
    """
    Base class for evaluation metrics.
    Provides common functionality for tracking and analyzing checkpoint results.
    """
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.checkpoint_results = {}
        self.aggregated_metrics = {}
        
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
        
        logger.info(f"Added {self.metric_name} results for checkpoint {checkpoint_name}")
    
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
        Should be implemented by subclasses.
        
        Returns:
            Dictionary with summary statistics
        """
        raise NotImplementedError("Subclasses must implement calculate_summary_statistics")
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze performance trends across checkpoints.
        Should be implemented by subclasses.
        
        Returns:
            Dictionary with trend analysis
        """
        raise NotImplementedError("Subclasses must implement analyze_performance_trends")
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of all results.
        
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {
            'metric_name': self.metric_name,
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
        
        logger.info(f"Saved {self.metric_name} metrics to {output_path}")
    
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
            logger.info(f"Loaded {self.metric_name} metrics for {len(self.checkpoint_results)} checkpoints")
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
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement print_summary")