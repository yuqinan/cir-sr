import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from .base_metrics import BaseMetrics

logger = logging.getLogger(__name__)

class TeacherAccuracyMetrics(BaseMetrics):
    """
    Metrics calculator for teacher accuracy evaluation.
    Tracks accuracy across different checkpoints and provides analysis.
    """
    
    def __init__(self):
        super().__init__("teacher_accuracy")
        
    def get_checkpoint_accuracy(self, checkpoint_name: str) -> Optional[float]:
        """
        Get accuracy for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            
        Returns:
            Accuracy value or None if not found
        """
        result = self.get_checkpoint_result(checkpoint_name)
        return result['accuracy'] if result else None
    
    def get_all_accuracies(self) -> Dict[str, float]:
        """
        Get accuracies for all checkpoints.
        
        Returns:
            Dictionary mapping checkpoint names to accuracies
        """
        return {name: results['accuracy'] for name, results in self.checkpoint_results.items()}
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics across all checkpoints.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.checkpoint_results:
            return {}
        
        accuracies = [results['accuracy'] for results in self.checkpoint_results.values()]
        
        summary = {
            'num_checkpoints': len(self.checkpoint_results),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'median_accuracy': np.median(accuracies),
            'accuracy_range': np.max(accuracies) - np.min(accuracies)
        }
        
        # Find best and worst checkpoints
        best_checkpoint = max(self.checkpoint_results.items(), key=lambda x: x[1]['accuracy'])
        worst_checkpoint = min(self.checkpoint_results.items(), key=lambda x: x[1]['accuracy'])
        
        summary['best_checkpoint'] = {
            'name': best_checkpoint[0],
            'accuracy': best_checkpoint[1]['accuracy']
        }
        
        summary['worst_checkpoint'] = {
            'name': worst_checkpoint[0],
            'accuracy': worst_checkpoint[1]['accuracy']
        }
        
        return summary
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze performance trends across checkpoints.
        
        Returns:
            Dictionary with trend analysis
        """
        if len(self.checkpoint_results) < 2:
            return {'trend': 'insufficient_data'}
        
        sorted_results = self.get_sorted_results()
        accuracies = [results['accuracy'] for _, results in sorted_results]
        
        # Simple trend analysis
        if len(accuracies) >= 3:
            # Calculate linear trend
            x = np.arange(len(accuracies))
            coeffs = np.polyfit(x, accuracies, 1)
            trend_slope = coeffs[0]
            
            if trend_slope > 0.01:
                trend = 'improving'
            elif trend_slope < -0.01:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            # Simple comparison for 2 checkpoints
            if accuracies[-1] > accuracies[0]:
                trend = 'improving'
            elif accuracies[-1] < accuracies[0]:
                trend = 'declining'
            else:
                trend = 'stable'
        
        # Calculate improvement metrics
        first_accuracy = accuracies[0]
        last_accuracy = accuracies[-1]
        absolute_improvement = last_accuracy - first_accuracy
        relative_improvement = (absolute_improvement / first_accuracy) * 100 if first_accuracy > 0 else 0
        
        analysis = {
            'trend': trend,
            'first_accuracy': first_accuracy,
            'last_accuracy': last_accuracy,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement,
            'num_checkpoints_analyzed': len(accuracies)
        }
        
        if len(accuracies) >= 3:
            analysis['trend_slope'] = trend_slope
        
        return analysis
    
    def print_summary(self) -> None:
        """
        Print a summary of the metrics.
        """
        if not self.checkpoint_results:
            print("No checkpoint results available")
            return
        
        summary = self.calculate_summary_statistics()
        trends = self.analyze_performance_trends()
        
        print("\n" + "="*60)
        print("TEACHER REWARD EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Number of checkpoints evaluated: {summary['num_checkpoints']}")
        print(f"Mean reward: {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")
        print(f"Reward range: {summary['min_accuracy']:.3f} - {summary['max_accuracy']:.3f}")
        print(f"Best checkpoint: {summary['best_checkpoint']['name']} ({summary['best_checkpoint']['accuracy']:.3f})")
        print(f"Worst checkpoint: {summary['worst_checkpoint']['name']} ({summary['worst_checkpoint']['accuracy']:.3f})")
        
        #print(f"\nPerformance trend: {trends['trend']}")
        #print(f"First → Last reward: {trends['first_accuracy']:.3f} → {trends['last_accuracy']:.3f}")
        #print(f"Absolute improvement: {trends['absolute_improvement']:.3f}")
        #print(f"Relative improvement: {trends['relative_improvement']:.1f}%")
        #
        #print("\nCheckpoint Results:")
        #for checkpoint_name, results in self.get_sorted_results():
        #    print(f"  {checkpoint_name}: {results['accuracy']:.3f}")
        
        print("="*60)

def create_teacher_accuracy_metrics() -> TeacherAccuracyMetrics:
    """
    Factory function to create a TeacherAccuracyMetrics instance.
    
    Returns:
        TeacherAccuracyMetrics instance
    """
    return TeacherAccuracyMetrics()