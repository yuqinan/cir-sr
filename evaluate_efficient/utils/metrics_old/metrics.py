import numpy as np
import json
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import os
import logging
from data.utils import match_format, extract_answer, extract_gold_answer_gsm8k
import reasoning_gym

logger = logging.getLogger(__name__)

class AccuracyCalculator:
    """Calculate accuracy based on task-specific scoring functions, following RewardCalculator pattern."""
    
    def __init__(self, task: str, task_type: Union[str, List[str]]):
        # Handle both single task type and list of task types
        if isinstance(task_type, str):
            self.task_types = [task_type]
        else:
            self.task_types = task_type
        
        self.task = task
        
        if task == "reasoning_gym":
            # Create a mapping from task type to task handler
            self.task_handlers = {}
            for task_ in self.task_types:
                self.task_handlers[task_] = reasoning_gym.create_dataset(task_)
        elif task == "gsm8k":
            # GSM8K doesn't need task handlers - simple comparison
            self.task_handlers = None
    
    def calculate_accuracy(self, prediction: str, entry: Dict[str, Any]) -> float:
        """
        Calculate accuracy for a prediction using task-specific scoring.
        Returns 1.0 for correct, 0.0 for incorrect (no format bonus like reward calculation).
        
        Args:
            prediction: Model's prediction
            entry: Original data entry containing ground truth and context
            
        Returns:
            Accuracy value (0.0 or 1.0)
        """
        if self.task == "gsm8k":
            # Simple comparison for GSM8K
            ground_truth = entry.get('answer', '')
            
            # Extract numerical answer from prediction if needed
            prediction_clean = extract_answer(prediction) if prediction else ""
            ground_truth_clean = extract_gold_answer_gsm8k(ground_truth) if ground_truth else ground_truth
            
            # Compare answers (case-insensitive, strip whitespace)
            try:
                if prediction_clean.strip().lower() == ground_truth_clean.strip().lower():
                    return 1.0
                else:
                    return 0.0
            except:
                logger.warning(f"Error comparing prediction='{prediction_clean}' vs ground_truth='{ground_truth_clean}'")
                return 0.0
        
        elif self.task == "reasoning_gym":
            # Use task-specific scoring function for reasoning gym
            task_handler = self.task_handlers[entry['metadata']['source_dataset']]
            prediction_clean = extract_answer(prediction)
            # Use raw score from task handler (no format bonus for accuracy)
            return task_handler.score_answer(prediction_clean, entry)
        
        else:
            raise ValueError(f"Unknown task type: {self.task}")
    
    def calculate_batch_accuracy(self, predictions: List[str], entries: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate accuracy for a batch of predictions using task-specific scoring.
        
        Args:
            predictions: List of model predictions (strings)
            entries: List of ground truth entries (dicts)
                
        Returns:
            List of accuracy values (0.0 or 1.0), one for each prediction
        """
        accuracies = []
        
        for i, pred in enumerate(predictions):         
            accuracy = self.calculate_accuracy(pred, entries[i])
            accuracies.append(accuracy)
        return accuracies
    
    def evaluate_predictions(self, predictions: List[str], entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions and return comprehensive metrics.
        
        Args:
            predictions: List of model predictions
            entries: List of ground truth entries
            
        Returns:
            Dictionary with accuracy metrics
        """
        if len(predictions) != len(entries):
            raise ValueError(f"Predictions ({len(predictions)}) and entries ({len(entries)}) must have same length")
        
        accuracies = self.calculate_batch_accuracy(predictions, entries)
        correct_count = sum(accuracies)
        total_count = len(accuracies)
        
        # Calculate detailed results for each prediction
        detailed_results = []
        for i, (pred, entry, acc) in enumerate(zip(predictions, entries, accuracies)):
            detailed_results.append({
                'index': i,
                'prediction': pred,
                'ground_truth': entry.get('answer', ''),
                'task_type': entry.get('metadata', {}).get('source_dataset', 'unknown'),
                'correct': acc == 1.0,
                'accuracy': acc
            })
        
        results = {
            'accuracy': correct_count / total_count if total_count > 0 else 0.0,
            'correct': int(correct_count),
            'total': total_count,
            'detailed_results': detailed_results,
            'task': self.task,
            'task_types': self.task_types
        }
        
        return results

class StudentAccuracyMetrics:
    """
    Metrics calculator for student accuracy evaluation.
    Tracks accuracy across different checkpoints and provides analysis.
    """
    
    def __init__(self):
        self.checkpoint_results = {}
        self.aggregated_metrics = {}
        
    def add_checkpoint_result(self, checkpoint_name: str, results: Dict[str, Any]) -> None:
        """
        Add results for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            results: Results dictionary from student evaluation
        """
        self.checkpoint_results[checkpoint_name] = {
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total'],
            'detailed_results': results['detailed_results'],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Added results for checkpoint {checkpoint_name}: {results['accuracy']:.3f}")
    
    def get_checkpoint_accuracy(self, checkpoint_name: str) -> Optional[float]:
        """
        Get accuracy for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            
        Returns:
            Accuracy value or None if not found
        """
        if checkpoint_name in self.checkpoint_results:
            return self.checkpoint_results[checkpoint_name]['accuracy']
        return None
    
    def get_all_accuracies(self) -> Dict[str, float]:
        """
        Get accuracies for all checkpoints.
        
        Returns:
            Dictionary mapping checkpoint names to accuracies
        """
        return {name: results['accuracy'] for name, results in self.checkpoint_results.items()}
    
    def get_sorted_results(self) -> List[Tuple[str, float]]:
        """
        Get checkpoint results sorted by checkpoint name/step.
        
        Returns:
            List of (checkpoint_name, accuracy) tuples sorted by step
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
        
        results = [(name, self.checkpoint_results[name]['accuracy']) 
                  for name in self.checkpoint_results.keys()]
        
        results.sort(key=lambda x: extract_step(x[0]))
        return results
    
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
        accuracies = [acc for _, acc in sorted_results]
        
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
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of all results.
        
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {
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
        
        logger.info(f"Saved metrics to {output_path}")
    
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
            logger.info(f"Loaded metrics for {len(self.checkpoint_results)} checkpoints")
        else:
            logger.warning("No checkpoint results found in loaded data")
    
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
        print("STUDENT ACCURACY EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Number of checkpoints evaluated: {summary['num_checkpoints']}")
        print(f"Mean accuracy: {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")
        print(f"Accuracy range: {summary['min_accuracy']:.3f} - {summary['max_accuracy']:.3f}")
        print(f"Best checkpoint: {summary['best_checkpoint']['name']} ({summary['best_checkpoint']['accuracy']:.3f})")
        print(f"Worst checkpoint: {summary['worst_checkpoint']['name']} ({summary['worst_checkpoint']['accuracy']:.3f})")
        
        print(f"\nPerformance trend: {trends['trend']}")
        print(f"First → Last accuracy: {trends['first_accuracy']:.3f} → {trends['last_accuracy']:.3f}")
        print(f"Absolute improvement: {trends['absolute_improvement']:.3f}")
        print(f"Relative improvement: {trends['relative_improvement']:.1f}%")
        
        print("\nCheckpoint Results:")
        for checkpoint_name, accuracy in self.get_sorted_results():
            print(f"  {checkpoint_name}: {accuracy:.3f}")
        
        print("="*60)

def create_accuracy_calculator(task: str, task_type: Union[str, List[str]]) -> AccuracyCalculator:
    """
    Factory function to create an AccuracyCalculator instance.
    
    Args:
        task: Task name (e.g., "reasoning_gym", "gsm8k")
        task_type: Task type(s) (e.g., "mini_sudoku", ["mini_sudoku", "spiral_matrix"])
    
    Returns:
        AccuracyCalculator instance
    """
    return AccuracyCalculator(task, task_type)

def create_accuracy_metrics() -> StudentAccuracyMetrics:
    """
    Factory function to create a StudentAccuracyMetrics instance.
    
    Returns:
        StudentAccuracyMetrics instance
    """
    return StudentAccuracyMetrics()