"""
Simplified Chain-of-Thought Perturbation Metrics

This module provides simple metrics for CoT perturbation that work as post-processing
on already generated teacher responses. It only tracks accuracy for different perturbations.
"""

import json
import os
from typing import Dict, List, Any, Optional
import logging
from .base_metrics import BaseMetrics

logger = logging.getLogger(__name__)

class CoTPerturbationMetrics(BaseMetrics):
    """
    Simple metrics calculator for chain-of-thought perturbation evaluation.
    Only tracks reward for original, truncated, and filler perturbations.
    """
    
    def __init__(self):
        super().__init__("cot_perturbation")
    
    def process_teacher_responses(self, teacher_responses_path: str, 
                                teacher_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process teacher responses and calculate perturbation reward metrics.
        This is called AFTER the perturbation inference has been completed.
        
        Args:
            teacher_responses_path: Path where original responses are saved
            teacher_data: Teacher responses with perturbation results
        
        Returns:
            Dictionary with perturbation metrics
        """
        try:            
            # Calculate original reward using average reward
            reward_scores = [item.get('reward_score', 0.0) for item in teacher_data]
            original_reward = sum(reward_scores) / len(reward_scores) if reward_scores else 0.0
            
            # Calculate truncate reward from perturbation results
            truncate_scores = []
            filler_scores = []
            expert_thinking_scores = []
            
            for item in teacher_data:
                if 'truncate_perturbation' in item:
                    truncate_scores.append(item['truncate_perturbation'].get('reward_score', 0.0))
                if 'filler_perturbation' in item:
                    filler_scores.append(item['filler_perturbation'].get('reward_score', 0.0))
                if 'expert_thinking_perturbation' in item:
                    expert_thinking_scores.append(item['expert_thinking_perturbation'].get('reward_score', 0.0))
            
            truncate_reward = sum(truncate_scores) / len(truncate_scores) if truncate_scores else 0.0
            filler_reward = sum(filler_scores) / len(filler_scores) if filler_scores else 0.0
            expert_thinking_reward = sum(expert_thinking_scores) / len(expert_thinking_scores) if expert_thinking_scores else 0.0
            
            # Prepare results
            results = {
                'original_reward': original_reward,
                'truncate_reward': truncate_reward,
                'filler_reward': filler_reward,
                'expert_thinking_reward': expert_thinking_reward,
                'num_samples': len(teacher_data),
                'perturbation_details': {
                    'truncate_samples': len(truncate_scores),
                    'filler_samples': len(filler_scores),
                    'expert_thinking_samples': len(expert_thinking_scores),
                    'truncate_drop': original_reward - truncate_reward,
                    'filler_drop': original_reward - filler_reward,
                    'expert_thinking_improvement': expert_thinking_reward - original_reward
                }
            }
            
            logger.info(f"CoT Perturbation metrics: {len(teacher_data)} samples")
            logger.info(f"  Original reward: {original_reward:.4f}")
            logger.info(f"  Truncate reward: {truncate_reward:.4f} (drop: {original_reward - truncate_reward:.4f})")
            logger.info(f"  Filler reward: {filler_reward:.4f} (drop: {original_reward - filler_reward:.4f})")
            logger.info(f"  Expert thinking reward: {expert_thinking_reward:.4f} (improvement: {expert_thinking_reward - original_reward:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing CoT perturbation metrics: {e}")
            return {
                'original_reward': 0.0,
                'truncate_reward': 0.0,
                'filler_reward': 0.0,
                'expert_thinking_reward': 0.0,
                'num_samples': 0,
                'error': str(e)
            }
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics across all checkpoints."""
        if not self.checkpoint_results:
            return {}
        
        # Extract rewards
        original_rewards = [r.get('original_reward', 0.0) for r in self.checkpoint_results.values()]
        truncate_rewards = [r.get('truncate_reward', 0.0) for r in self.checkpoint_results.values()]
        filler_rewards = [r.get('filler_reward', 0.0) for r in self.checkpoint_results.values()]
        expert_thinking_rewards = [r.get('expert_thinking_reward', 0.0) for r in self.checkpoint_results.values()]
        
        import numpy as np
        
        return {
            'num_checkpoints': len(self.checkpoint_results),
            'original_reward': {
                'mean': float(np.mean(original_rewards)),
                'std': float(np.std(original_rewards)),
                'min': float(np.min(original_rewards)),
                'max': float(np.max(original_rewards))
            },
            'truncate_reward': {
                'mean': float(np.mean(truncate_rewards)),
                'std': float(np.std(truncate_rewards)),
                'min': float(np.min(truncate_rewards)),
                'max': float(np.max(truncate_rewards))
            },
            'filler_reward': {
                'mean': float(np.mean(filler_rewards)),
                'std': float(np.std(filler_rewards)),
                'min': float(np.min(filler_rewards)),
                'max': float(np.max(filler_rewards))
            },
            'expert_thinking_reward': {
                'mean': float(np.mean(expert_thinking_rewards)),
                'std': float(np.std(expert_thinking_rewards)),
                'min': float(np.min(expert_thinking_rewards)),
                'max': float(np.max(expert_thinking_rewards))
            }
        }
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across checkpoints."""
        if len(self.checkpoint_results) < 2:
            return {'error': 'Need at least 2 checkpoints for trend analysis'}
        
        sorted_results = self.get_sorted_results()
        
        # Extract step numbers and rewards
        steps = []
        original_rewards = []
        truncate_rewards = []
        filler_rewards = []
        expert_thinking_rewards = []
        
        for checkpoint_name, results in sorted_results:
            try:
                if 'step_' in checkpoint_name:
                    step = int(checkpoint_name.split('step_')[1].split('_')[0])
                else:
                    step = 0
            except:
                step = 0
            
            steps.append(step)
            original_rewards.append(results.get('original_reward', 0.0))
            truncate_rewards.append(results.get('truncate_reward', 0.0))
            filler_rewards.append(results.get('filler_reward', 0.0))
            expert_thinking_rewards.append(results.get('expert_thinking_reward', 0.0))
        
        return {
            'steps': steps,
            'original_rewards': original_rewards,
            'truncate_rewards': truncate_rewards,
            'filler_rewards': filler_rewards,
            'expert_thinking_rewards': expert_thinking_rewards
        }
    
    def print_summary(self) -> None:
        """Print a summary of the CoT perturbation metrics."""
        print(f"\n=== {self.metric_name.upper()} SUMMARY ===")
        
        if not self.checkpoint_results:
            print("No checkpoint results available.")
            return
        
        summary = self.calculate_summary_statistics()
        
        print(f"Checkpoints evaluated: {summary['num_checkpoints']}")
        print(f"Original Reward: {summary['original_reward']['mean']:.4f} ± {summary['original_reward']['std']:.4f}")
        print(f"Truncate Reward: {summary['truncate_reward']['mean']:.4f} ± {summary['truncate_reward']['std']:.4f}")
        print(f"Filler Reward: {summary['filler_reward']['mean']:.4f} ± {summary['filler_reward']['std']:.4f}")
        print(f"Expert Thinking Reward: {summary['expert_thinking_reward']['mean']:.4f} ± {summary['expert_thinking_reward']['std']:.4f}")
        
        # Show reward drops/improvements
        orig_mean = summary['original_reward']['mean']
        trunc_mean = summary['truncate_reward']['mean']
        filler_mean = summary['filler_reward']['mean']
        expert_thinking_mean = summary['expert_thinking_reward']['mean']
        
        print(f"Truncate Reward Drop: {orig_mean - trunc_mean:.4f}")
        print(f"Filler Reward Drop: {orig_mean - filler_mean:.4f}")
        print(f"Expert Thinking Improvement: {expert_thinking_mean - orig_mean:.4f}")
        
        print("=" * 50)