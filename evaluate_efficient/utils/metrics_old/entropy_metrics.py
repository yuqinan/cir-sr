import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import logging
from .base_metrics import BaseMetrics

logger = logging.getLogger(__name__)

class EntropyMetrics(BaseMetrics):
    """
    Metrics calculator for teacher model entropy evaluation.
    Tracks entropy across different checkpoints and provides analysis.
    """
    
    def __init__(self):
        super().__init__("entropy")
        
    def calculate_entropy_from_logits(self, logits: torch.Tensor) -> float:
        """
        Calculate entropy from logits.
        
        Args:
            logits: Tensor of shape (seq_len, vocab_size) or (vocab_size,)
            
        Returns:
            Average entropy across sequence
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate entropy: -sum(p * log(p))
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Return average entropy across sequence
        return entropy.mean().item()
    
    def calculate_entropy_from_token_probs(self, token_probs: List[float]) -> float:
        """
        Calculate entropy from token probabilities.
        
        Args:
            token_probs: List of token probabilities
            
        Returns:
            Entropy value
        """
        probs = np.array(token_probs)
        probs = probs / probs.sum()  # Normalize
        
        # Avoid log(0)
        probs = probs[probs > 0]
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs))
        return entropy
    
    def process_teacher_responses(self, teacher_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process teacher responses to calculate entropy metrics.
        
        Args:
            teacher_responses: List of teacher responses with generation info
            
        Returns:
            Dictionary with entropy metrics
        """
        entropies = []
        per_example_results = []
        
        for i, response in enumerate(teacher_responses):
            # Extract entropy from response if available
            if 'generation_entropy' in response:
                entropy = response['generation_entropy']
            elif 'logits' in response:
                # Calculate entropy from logits if available
                logits = response['logits']
                if isinstance(logits, list):
                    logits = torch.tensor(logits)
                entropy = self.calculate_entropy_from_logits(logits)
            elif 'token_probs' in response:
                # Calculate entropy from token probabilities
                entropy = self.calculate_entropy_from_token_probs(response['token_probs'])
            else:
                # If no entropy info available, skip this example
                logger.warning(f"No entropy information available for response {i}")
                continue
            
            entropies.append(entropy)
            per_example_results.append({
                'index': i,
                'question': response.get('question', ''),
                'entropy': entropy,
                'response_length': len(response.get('teacher_response', ''))
            })
        
        if not entropies:
            logger.warning("No entropy values calculated")
            return {
                'mean_entropy': 0.0,
                'std_entropy': 0.0,
                'min_entropy': 0.0,
                'max_entropy': 0.0,
                'num_examples': 0,
                'per_example_results': []
            }
        
        results = {
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies),
            'median_entropy': np.median(entropies),
            'num_examples': len(entropies),
            'per_example_results': per_example_results
        }
        
        return results
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics across all checkpoints.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.checkpoint_results:
            return {}
        
        all_entropies = []
        checkpoint_means = []
        
        for checkpoint_name, results in self.checkpoint_results.items():
            checkpoint_means.append(results['mean_entropy'])
            # Collect all individual entropies
            for example in results['per_example_results']:
                all_entropies.append(example['entropy'])
        
        summary = {
            'num_checkpoints': len(self.checkpoint_results),
            'overall_mean_entropy': np.mean(all_entropies) if all_entropies else 0.0,
            'overall_std_entropy': np.std(all_entropies) if all_entropies else 0.0,
            'overall_min_entropy': np.min(all_entropies) if all_entropies else 0.0,
            'overall_max_entropy': np.max(all_entropies) if all_entropies else 0.0,
            'checkpoint_mean_entropy': np.mean(checkpoint_means) if checkpoint_means else 0.0,
            'checkpoint_std_entropy': np.std(checkpoint_means) if checkpoint_means else 0.0,
            'total_examples': len(all_entropies)
        }
        
        # Find checkpoint with highest and lowest mean entropy
        if checkpoint_means:
            best_checkpoint = min(self.checkpoint_results.items(), key=lambda x: x[1]['mean_entropy'])
            worst_checkpoint = max(self.checkpoint_results.items(), key=lambda x: x[1]['mean_entropy'])
            
            summary['lowest_entropy_checkpoint'] = {
                'name': best_checkpoint[0],
                'mean_entropy': best_checkpoint[1]['mean_entropy']
            }
            
            summary['highest_entropy_checkpoint'] = {
                'name': worst_checkpoint[0],
                'mean_entropy': worst_checkpoint[1]['mean_entropy']
            }
        
        return summary
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze entropy trends across checkpoints.
        
        Returns:
            Dictionary with trend analysis
        """
        if len(self.checkpoint_results) < 2:
            return {'trend': 'insufficient_data'}
        
        sorted_results = self.get_sorted_results()
        entropies = [results['mean_entropy'] for _, results in sorted_results]
        
        # Simple trend analysis
        if len(entropies) >= 3:
            # Calculate linear trend
            x = np.arange(len(entropies))
            coeffs = np.polyfit(x, entropies, 1)
            trend_slope = coeffs[0]
            
            if trend_slope > 0.01:
                trend = 'increasing'  # Higher entropy = more uncertain
            elif trend_slope < -0.01:
                trend = 'decreasing'  # Lower entropy = more confident
            else:
                trend = 'stable'
        else:
            # Simple comparison for 2 checkpoints
            if entropies[-1] > entropies[0]:
                trend = 'increasing'
            elif entropies[-1] < entropies[0]:
                trend = 'decreasing'
            else:
                trend = 'stable'
        
        # Calculate change metrics
        first_entropy = entropies[0]
        last_entropy = entropies[-1]
        absolute_change = last_entropy - first_entropy
        relative_change = (absolute_change / first_entropy) * 100 if first_entropy > 0 else 0
        
        analysis = {
            'trend': trend,
            'first_entropy': first_entropy,
            'last_entropy': last_entropy,
            'absolute_change': absolute_change,
            'relative_change': relative_change,
            'num_checkpoints_analyzed': len(entropies)
        }
        
        if len(entropies) >= 3:
            analysis['trend_slope'] = trend_slope
        
        return analysis
    
    def print_summary(self) -> None:
        """
        Print a summary of the entropy metrics.
        """
        if not self.checkpoint_results:
            print("No checkpoint results available")
            return
        
        summary = self.calculate_summary_statistics()
        trends = self.analyze_performance_trends()
        
        print("\n" + "="*60)
        print("TEACHER MODEL ENTROPY EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Number of checkpoints evaluated: {summary['num_checkpoints']}")
        print(f"Overall mean entropy: {summary['overall_mean_entropy']:.3f} ± {summary['overall_std_entropy']:.3f}")
        print(f"Entropy range: {summary['overall_min_entropy']:.3f} - {summary['overall_max_entropy']:.3f}")
        print(f"Total examples analyzed: {summary['total_examples']}")
        
        if 'lowest_entropy_checkpoint' in summary:
            print(f"Lowest entropy checkpoint: {summary['lowest_entropy_checkpoint']['name']} ({summary['lowest_entropy_checkpoint']['mean_entropy']:.3f})")
            print(f"Highest entropy checkpoint: {summary['highest_entropy_checkpoint']['name']} ({summary['highest_entropy_checkpoint']['mean_entropy']:.3f})")
        
        print(f"\nEntropy trend: {trends['trend']}")
        print(f"First → Last entropy: {trends['first_entropy']:.3f} → {trends['last_entropy']:.3f}")
        print(f"Absolute change: {trends['absolute_change']:.3f}")
        print(f"Relative change: {trends['relative_change']:.1f}%")
        
        print("\nCheckpoint Results:")
        for checkpoint_name, results in self.get_sorted_results():
            print(f"  {checkpoint_name}: {results['mean_entropy']:.3f} ± {results['std_entropy']:.3f}")
        
        print("="*60)

def create_entropy_metrics() -> EntropyMetrics:
    """
    Factory function to create an EntropyMetrics instance.
    
    Returns:
        EntropyMetrics instance
    """
    return EntropyMetrics()