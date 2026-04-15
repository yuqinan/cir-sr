import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import logging
from .base_metrics import BaseMetrics

logger = logging.getLogger(__name__)

class PerplexityMetrics(BaseMetrics):
    """
    Metrics calculator for teacher model entropy evaluation.
    Tracks entropy across different checkpoints and provides analysis.
    """
    
    def __init__(self):
        super().__init__("perplexity")
        
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
        Process teacher responses to extract perplexity metrics.
        
        Args:
            teacher_responses: List of teacher responses with generation info
            
        Returns:
            Dictionary with perplexity metrics
        """
        perplexities = []
        per_example_results = []
        
        for i, response in enumerate(teacher_responses):
            # Extract perplexity from response using current data structure
            perplexity = None
            
            # Try different sources of perplexity data
            if 'generation_perplexity' in response:
                perplexity = response['generation_perplexity']
            elif 'generation_info' in response and 'perplexity' in response['generation_info']:
                perplexity = response['generation_info']['perplexity']
            elif 'generation_entropy' in response:
                # Legacy support for entropy field
                perplexity = response['generation_entropy']
            elif 'logits' in response:
                # Fallback: calculate from logits if available
                logits = response['logits']
                if isinstance(logits, list):
                    logits = torch.tensor(logits)
                perplexity = self.calculate_entropy_from_logits(logits)
            elif 'token_probs' in response:
                # Fallback: calculate from token probabilities
                perplexity = self.calculate_entropy_from_token_probs(response['token_probs'])
            
            # Skip if no perplexity data found
            if perplexity is None or perplexity <= 0:
                logger.debug(f"No valid perplexity information available for response {i}")
                continue
            
            perplexities.append(perplexity)
            per_example_results.append({
                'index': i,
                'question': response.get('question', ''),
                'perplexity': perplexity,
                'response_length': len(response.get('teacher_response', ''))
            })
        
        if not perplexities:
            logger.warning("No perplexity values found in teacher responses")
            return {
                'mean_perplexity': 0.0,
                'std_perplexity': 0.0,
                'min_perplexity': 0.0,
                'max_perplexity': 0.0,
                'num_examples': 0,
                'per_example_results': []
            }
        
        results = {
            'mean_perplexity': np.mean(perplexities),
            'std_perplexity': np.std(perplexities),
            'min_perplexity': np.min(perplexities),
            'max_perplexity': np.max(perplexities),
            'median_perplexity': np.median(perplexities),
            'num_examples': len(perplexities),
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
        
        all_perplexities = []
        checkpoint_means = []
        
        for checkpoint_name, results in self.checkpoint_results.items():
            # Handle both old and new field names
            mean_key = 'mean_perplexity' if 'mean_perplexity' in results else 'mean_entropy'
            checkpoint_means.append(results[mean_key])
            
            # Collect all individual perplexities
            for example in results['per_example_results']:
                perplexity_key = 'perplexity' if 'perplexity' in example else 'entropy'
                all_perplexities.append(example[perplexity_key])
        
        summary = {
            'num_checkpoints': len(self.checkpoint_results),
            'overall_mean_perplexity': np.mean(all_perplexities) if all_perplexities else 0.0,
            'overall_std_perplexity': np.std(all_perplexities) if all_perplexities else 0.0,
            'overall_min_perplexity': np.min(all_perplexities) if all_perplexities else 0.0,
            'overall_max_perplexity': np.max(all_perplexities) if all_perplexities else 0.0,
            'checkpoint_mean_perplexity': np.mean(checkpoint_means) if checkpoint_means else 0.0,
            'checkpoint_std_perplexity': np.std(checkpoint_means) if checkpoint_means else 0.0,
            'total_examples': len(all_perplexities)
        }
        
        # Find checkpoint with highest and lowest mean perplexity
        if checkpoint_means:
            mean_key = 'mean_perplexity' if 'mean_perplexity' in list(self.checkpoint_results.values())[0] else 'mean_entropy'
            best_checkpoint = min(self.checkpoint_results.items(), key=lambda x: x[1][mean_key])
            worst_checkpoint = max(self.checkpoint_results.items(), key=lambda x: x[1][mean_key])
            
            summary['lowest_perplexity_checkpoint'] = {
                'name': best_checkpoint[0],
                'mean_perplexity': best_checkpoint[1][mean_key]
            }
            
            summary['highest_perplexity_checkpoint'] = {
                'name': worst_checkpoint[0],
                'mean_perplexity': worst_checkpoint[1][mean_key]
            }
        
        return summary
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze perplexity trends across checkpoints.
        
        Returns:
            Dictionary with trend analysis
        """
        if len(self.checkpoint_results) < 2:
            return {'trend': 'insufficient_data'}
        
        sorted_results = self.get_sorted_results()
        
        # Handle both old and new field names
        first_result = sorted_results[0][1]
        mean_key = 'mean_perplexity' if 'mean_perplexity' in first_result else 'mean_entropy'
        perplexities = [results[mean_key] for _, results in sorted_results]
        
        # Simple trend analysis
        if len(perplexities) >= 3:
            # Calculate linear trend
            x = np.arange(len(perplexities))
            coeffs = np.polyfit(x, perplexities, 1)
            trend_slope = coeffs[0]
            
            if trend_slope > 0.01:
                trend = 'increasing'  # Higher perplexity = more uncertain
            elif trend_slope < -0.01:
                trend = 'decreasing'  # Lower perplexity = more confident
            else:
                trend = 'stable'
        else:
            # Simple comparison for 2 checkpoints
            if perplexities[-1] > perplexities[0]:
                trend = 'increasing'
            elif perplexities[-1] < perplexities[0]:
                trend = 'decreasing'
            else:
                trend = 'stable'
        
        # Calculate change metrics
        first_perplexity = perplexities[0]
        last_perplexity = perplexities[-1]
        absolute_change = last_perplexity - first_perplexity
        relative_change = (absolute_change / first_perplexity) * 100 if first_perplexity > 0 else 0
        
        analysis = {
            'trend': trend,
            'first_perplexity': first_perplexity,
            'last_perplexity': last_perplexity,
            'absolute_change': absolute_change,
            'relative_change': relative_change,
            'num_checkpoints_analyzed': len(perplexities)
        }
        
        if len(perplexities) >= 3:
            analysis['trend_slope'] = trend_slope
        
        return analysis
    
    def print_summary(self) -> None:
        """
        Print a summary of the perplexity metrics.
        """
        if not self.checkpoint_results:
            print("No checkpoint results available")
            return
        
        summary = self.calculate_summary_statistics()
        trends = self.analyze_performance_trends()
        
        print("\n" + "="*60)
        print("TEACHER MODEL PERPLEXITY EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Number of checkpoints evaluated: {summary['num_checkpoints']}")
        print(f"Overall mean perplexity: {summary['overall_mean_perplexity']:.3f} ± {summary['overall_std_perplexity']:.3f}")
        print(f"Perplexity range: {summary['overall_min_perplexity']:.3f} - {summary['overall_max_perplexity']:.3f}")
        print(f"Total examples analyzed: {summary['total_examples']}")
        
        if 'lowest_perplexity_checkpoint' in summary:
            print(f"Lowest perplexity checkpoint: {summary['lowest_perplexity_checkpoint']['name']} ({summary['lowest_perplexity_checkpoint']['mean_perplexity']:.3f})")
            print(f"Highest perplexity checkpoint: {summary['highest_perplexity_checkpoint']['name']} ({summary['highest_perplexity_checkpoint']['mean_perplexity']:.3f})")
        
        print(f"\nPerplexity trend: {trends['trend']}")
        print(f"First → Last perplexity: {trends['first_perplexity']:.3f} → {trends['last_perplexity']:.3f}")
        print(f"Absolute change: {trends['absolute_change']:.3f}")
        print(f"Relative change: {trends['relative_change']:.1f}%")
        
        print("\nCheckpoint Results:")
        for checkpoint_name, results in self.get_sorted_results():
            # Handle both old and new field names
            mean_key = 'mean_perplexity' if 'mean_perplexity' in results else 'mean_entropy'
            std_key = 'std_perplexity' if 'std_perplexity' in results else 'std_entropy'
            print(f"  {checkpoint_name}: {results[mean_key]:.3f} ± {results[std_key]:.3f}")
        
        print("="*60)

def create_perplexity_metrics() -> PerplexityMetrics:
    """
    Factory function to create a PerplexityMetrics instance.
    
    Returns:
        PerplexityMetrics instance
    """
    return PerplexityMetrics()