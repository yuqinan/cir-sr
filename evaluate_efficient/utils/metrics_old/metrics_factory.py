from typing import Dict, List, Any, Optional
import logging
from .base_metrics import BaseMetrics
from .student_accuracy_metrics import StudentAccuracyMetrics
from .teacher_accuracy_metrics import TeacherAccuracyMetrics
from .perplexity_metrics import PerplexityMetrics
from .cot_perturbation_metrics import CoTPerturbationMetrics

logger = logging.getLogger(__name__)

class MetricsFactory:
    """
    Factory class for creating and managing different types of metrics.
    """
    
    AVAILABLE_METRICS = {
        "student_accuracy": StudentAccuracyMetrics,
        "teacher_accuracy": TeacherAccuracyMetrics,
        "perplexity": PerplexityMetrics,
        "cot_perturbation": CoTPerturbationMetrics,
    }
    
    @classmethod
    def create_metrics(cls, metric_names: List[str]) -> Dict[str, BaseMetrics]:
        """
        Create metrics instances based on metric names.
        
        Args:
            metric_names: List of metric names to create
            
        Returns:
            Dictionary mapping metric names to metric instances
        """
        metrics = {}
        
        for metric_name in metric_names:
            if metric_name not in cls.AVAILABLE_METRICS:
                logger.warning(f"Unknown metric type: {metric_name}. Available metrics: {list(cls.AVAILABLE_METRICS.keys())}")
                continue
                
            try:
                metric_class = cls.AVAILABLE_METRICS[metric_name]
                metrics[metric_name] = metric_class()
                logger.info(f"Created {metric_name} metric")
            except Exception as e:
                logger.error(f"Failed to create {metric_name} metric: {e}")
                
        return metrics
    
    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """
        Get list of available metric types.
        
        Returns:
            List of available metric names
        """
        return list(cls.AVAILABLE_METRICS.keys())
    
    @classmethod
    def validate_metric_names(cls, metric_names: List[str]) -> List[str]:
        """
        Validate metric names and return only valid ones.
        
        Args:
            metric_names: List of metric names to validate
            
        Returns:
            List of valid metric names
        """
        valid_metrics = []
        
        for metric_name in metric_names:
            if metric_name in cls.AVAILABLE_METRICS:
                valid_metrics.append(metric_name)
            else:
                logger.warning(f"Invalid metric name: {metric_name}. Available: {list(cls.AVAILABLE_METRICS.keys())}")
                
        return valid_metrics

def create_metrics_from_config(config: Dict[str, Any]) -> Dict[str, BaseMetrics]:
    """
    Create metrics instances from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping metric names to metric instances
    """
    eval_config = config.get('evaluation', {})
    
    # Get metrics from config
    if 'metrics' in eval_config:
        metric_names = eval_config['metrics']
    else:
        # Default to student_accuracy
        metric_names = ['student_accuracy']
        logger.warning("No metrics specified, defaulting to student_accuracy")
    
    # Validate metric names
    valid_metrics = MetricsFactory.validate_metric_names(metric_names)
    
    if not valid_metrics:
        logger.error("No valid metrics found, defaulting to student_accuracy")
        valid_metrics = ['student_accuracy']
    
    # Create metrics instances
    metrics = MetricsFactory.create_metrics(valid_metrics)
    
    logger.info(f"Created metrics: {list(metrics.keys())}")
    return metrics