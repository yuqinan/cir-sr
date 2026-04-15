"""
Base Metric Class

Abstract base class for all teacher evaluation metrics.
Provides common interface and shared functionality without managing models directly.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import os
import logging

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """Abstract base class for teacher evaluation metrics.
    
    Metrics receive model dependencies (model_manager, openai_client) from the main
    orchestrator rather than managing them directly. This ensures proper model
    loading/cleanup coordination and resource sharing.
    """
    
    def __init__(self, config: Dict[str, Any], reward_calculator, prompt_manager):
        """Initialize base metric with common dependencies."""
        self.config = config
        self.teacher_config = config['evaluation']['teacher_model']
        self.batch_size = config['evaluation']['batch_size']
        self.reward_calculator = reward_calculator
        self.prompt_manager = prompt_manager
        
        # Model management will be handled by TeacherSingleModel
        self.use_openai_api = self.teacher_config.get('use_openai_api', False)
        
        # Initialize metric-specific configuration
        self._init_metric_config()
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def _init_metric_config(self):
        """Initialize metric-specific configuration."""
        pass
    
    @abstractmethod
    def can_run(self) -> bool:
        """Check if this metric should be run based on configuration."""
        pass
    
    @abstractmethod
    def evaluate(self, teacher_data: List[Dict[str, Any]], model_manager=None, 
                openai_client=None, **kwargs) -> List[Dict[str, Any]]:
        """Run the metric evaluation.
        
        Args:
            teacher_data: Input data or existing teacher responses
            model_manager: VLLMModelManager instance (for local models)
            openai_client: OpenAIClient instance (for API models)
            **kwargs: Additional metric-specific arguments
        """
        pass
    
    def _calculate_reward(self, answer: str, item: Dict[str, Any], index: int) -> float:
        """Calculate reward score for answer (shared utility)."""
        entry = {
            'answer': item['answer'],
            'metadata': item.get('metadata', {}),
            'data_source': item.get('data_source', ''),
            'index': item.get('index', index)
        }
        
        # Ensure metadata has source_dataset
        if 'metadata' not in entry or 'source_dataset' not in entry['metadata']:
            entry['metadata'] = {'source_dataset': entry.get('data_source', 'mini_sudoku')}
        
        return self.reward_calculator.calculate_score(answer, entry)
    
    def _extract_step_number(self, checkpoint_name: str) -> str:
        """Extract step number from checkpoint name (shared utility)."""
        import re
        
        # Handle OpenAI API models
        if checkpoint_name == "openai_api":
            return "0"
        
        patterns = [r'global_step_(\d+)', r'step_(\d+)', r'(\d+)']
        
        for pattern in patterns:
            match = re.search(pattern, checkpoint_name)
            if match:
                return match.group(1)
        
        return checkpoint_name
    
    def _get_checkpoint_name(self, model_manager=None, openai_client=None) -> str:
        """Get current checkpoint name (shared utility)."""
        if self.use_openai_api and openai_client:
            return f"openai_{openai_client.model_name}"
        elif model_manager:
            return model_manager.current_checkpoint_name or "unknown"
        else:
            return "unknown"
    
    # Base checkpoint tracking functionality (from BaseMetrics)
    def add_checkpoint_result(self, checkpoint_name: str, results: Dict[str, Any]) -> None:
        """
        Add results for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            results: Results dictionary from evaluation
        """
        if not hasattr(self, 'checkpoint_results'):
            self.checkpoint_results = {}
        
        self.checkpoint_results[checkpoint_name] = {
            **results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Added {self.__class__.__name__} results for checkpoint {checkpoint_name}")
    
    def get_checkpoint_result(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """
        Get results for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            
        Returns:
            Results dictionary or None if not found
        """
        if not hasattr(self, 'checkpoint_results'):
            return None
        return self.checkpoint_results.get(checkpoint_name)
    
    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get results for all checkpoints.
        
        Returns:
            Dictionary mapping checkpoint names to results
        """
        if not hasattr(self, 'checkpoint_results'):
            return {}
        return self.checkpoint_results
    
    def get_sorted_results(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get checkpoint results sorted by checkpoint name/step.
        
        Returns:
            List of (checkpoint_name, results) tuples sorted by step
        """
        if not hasattr(self, 'checkpoint_results'):
            return []
        
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
        Should be implemented by subclasses for specific metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        return {}
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze performance trends across checkpoints.
        Should be implemented by subclasses for specific metrics.
        
        Returns:
            Dictionary with trend analysis
        """
        return {}
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of all results.
        
        Returns:
            Dictionary with detailed analysis
        """
        if not hasattr(self, 'checkpoint_results'):
            self.checkpoint_results = {}
        
        analysis = {
            'metric_name': self.__class__.__name__.lower().replace('metric', ''),
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
        
        logger.info(f"Saved {self.__class__.__name__} metrics to {output_path}")
    
    def load_metrics(self, input_path: str) -> None:
        """
        Load metrics from file.
        
        Args:
            input_path: Path to load metrics from
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        if 'checkpoint_results' in data:
            if not hasattr(self, 'checkpoint_results'):
                self.checkpoint_results = {}
            self.checkpoint_results = data['checkpoint_results']
            logger.info(f"Loaded {self.__class__.__name__} metrics for {len(self.checkpoint_results)} checkpoints")
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
        Should be implemented by subclasses for specific output format.
        """
        print(f"\n=== {self.__class__.__name__.upper()} SUMMARY ===")
        
        if not hasattr(self, 'checkpoint_results') or not self.checkpoint_results:
            print("No checkpoint results available.")
            return
        
        print(f"Checkpoints evaluated: {len(self.checkpoint_results)}")
        print("=" * 50)