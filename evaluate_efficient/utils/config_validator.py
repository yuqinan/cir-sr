import torch
import logging
from typing import Dict, Any, Tuple
from omegaconf import OmegaConf, DictConfig
import warnings

logger = logging.getLogger(__name__)

class ConfigValidator:
    """
    Validates and optimizes configuration settings based on hardware capabilities.
    """
    
    def __init__(self):
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.gpu_memory_total = self._get_total_gpu_memory()
        self.system_memory = self._get_system_memory()
        
    def _get_total_gpu_memory(self) -> int:
        """Get total GPU memory across all devices."""
        if not torch.cuda.is_available():
            return 0
            
        total_memory = 0
        for i in range(self.gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_memory += props.total_memory
        return total_memory
    
    def _get_system_memory(self) -> int:
        """Get total system memory."""
        try:
            import psutil
            return psutil.virtual_memory().total
        except:
            return 32 * 1024**3  # Default to 32GB
    
    def validate_and_optimize_config(self, config: DictConfig) -> DictConfig:
        """
        Validate and optimize configuration based on hardware capabilities.
        
        Args:
            config: OmegaConf configuration
            
        Returns:
            Optimized configuration
        """
        logger.info(f"Validating configuration with {self.gpu_count} GPUs, {self.gpu_memory_total / (1024**3):.1f}GB GPU memory")
        
        # Make a copy to avoid modifying original
        optimized_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        
        # Validate and optimize different sections
        self._validate_vllm_config(optimized_config)
        self._validate_batch_sizes(optimized_config)
        self._validate_parallel_processing(optimized_config)
        self._validate_memory_settings(optimized_config)
        self._validate_dataset_sizes(optimized_config)
        
        # Log optimizations
        self._log_optimizations(config, optimized_config)
        
        return optimized_config
    
    def _validate_vllm_config(self, config: DictConfig) -> None:
        """Validate and optimize vLLM configuration."""
        vllm_config = config.evaluation.vllm
        
        # Validate tensor_parallel_size
        if vllm_config.tensor_parallel_size > self.gpu_count:
            logger.warning(f"tensor_parallel_size ({vllm_config.tensor_parallel_size}) > available GPUs ({self.gpu_count})")
            vllm_config.tensor_parallel_size = max(1, self.gpu_count)
        
        # Optimize for single GPU
        if self.gpu_count == 1:
            vllm_config.tensor_parallel_size = 1
            vllm_config.gpu_memory_utilization = min(0.8, vllm_config.gpu_memory_utilization)
        
        # Optimize for multi-GPU
        elif self.gpu_count >= 4:
            if vllm_config.tensor_parallel_size == 1:
                vllm_config.tensor_parallel_size = min(4, self.gpu_count)
                logger.info(f"Optimized tensor_parallel_size to {vllm_config.tensor_parallel_size} for multi-GPU")
        
        # Validate memory utilization
        if vllm_config.gpu_memory_utilization > 0.95:
            logger.warning("GPU memory utilization > 95% may cause OOM errors")
            vllm_config.gpu_memory_utilization = 0.9
        
        # Optimize batch processing parameters based on GPU count
        if self.gpu_count >= 4:
            if 'max_num_batched_tokens' not in vllm_config:
                vllm_config.max_num_batched_tokens = 16384
            if 'max_num_seqs' not in vllm_config:
                vllm_config.max_num_seqs = 256
            if 'enable_chunked_prefill' not in vllm_config:
                vllm_config.enable_chunked_prefill = True
        
        # Validate max_model_len
        if vllm_config.max_model_len > 8192:
            logger.warning(f"max_model_len ({vllm_config.max_model_len}) is very large, may cause memory issues")
    
    def _validate_batch_sizes(self, config: DictConfig) -> None:
        """Validate and optimize batch sizes based on GPU capabilities."""
        eval_config = config.evaluation
        
        # Calculate recommended batch size based on GPU memory
        gpu_memory_per_device = self.gpu_memory_total / max(1, self.gpu_count)
        
        # Rough heuristic: 1GB GPU memory ≈ batch size of 4-8 depending on model size
        if gpu_memory_per_device < 8 * 1024**3:  # < 8GB
            recommended_batch_size = 4
        elif gpu_memory_per_device < 16 * 1024**3:  # < 16GB
            recommended_batch_size = 8
        elif gpu_memory_per_device < 32 * 1024**3:  # < 32GB
            recommended_batch_size = 16
        else:  # >= 32GB
            recommended_batch_size = 32
        
        # Adjust for multiple GPUs
        if self.gpu_count > 1:
            recommended_batch_size = min(recommended_batch_size * 2, 64)
        
        # Warn if batch size is too large
        if eval_config.batch_size > recommended_batch_size * 2:
            logger.warning(f"Batch size ({eval_config.batch_size}) may be too large for available GPU memory")
            logger.warning(f"Recommended batch size: {recommended_batch_size}")
            
        # Warn if batch size is too small for multi-GPU
        if self.gpu_count > 2 and eval_config.batch_size < 8:
            logger.warning(f"Batch size ({eval_config.batch_size}) may be too small for {self.gpu_count} GPUs")
            logger.warning(f"Consider increasing to {recommended_batch_size} for better GPU utilization")
        
        # Auto-adjust if requested
        if eval_config.get('auto_optimize_batch_size', False):
            original_batch_size = eval_config.batch_size
            eval_config.batch_size = recommended_batch_size
            logger.info(f"Auto-optimized batch size: {original_batch_size} → {recommended_batch_size}")
    
    def _validate_parallel_processing(self, config: DictConfig) -> None:
        """Validate parallel processing configuration."""
        if 'parallel_processing' not in config.evaluation:
            return
            
        parallel_config = config.evaluation.parallel_processing
        
        # Disable parallel processing if single GPU
        if self.gpu_count <= 1 and parallel_config.enabled:
            logger.info("Disabling parallel processing for single GPU setup")
            parallel_config.enabled = False
        
        # Optimize worker count
        if parallel_config.enabled:
            max_workers = min(4, max(1, self.gpu_count // 2))
            if parallel_config.num_workers > max_workers:
                logger.warning(f"Reducing num_workers from {parallel_config.num_workers} to {max_workers}")
                parallel_config.num_workers = max_workers
    
    def _validate_memory_settings(self, config: DictConfig) -> None:
        """Validate memory-related settings."""
        eval_config = config.evaluation
        
        # Check if dataset sizes are reasonable for available memory
        teacher_size = eval_config.teacher_dataset.size
        student_size = eval_config.student_dataset.size
        
        # Rough estimate: each example needs ~1KB in memory
        estimated_memory_need = (teacher_size + student_size) * 1024
        
        if estimated_memory_need > self.system_memory * 0.5:
            logger.warning(f"Large dataset sizes may cause memory issues")
            logger.warning(f"Consider reducing dataset sizes or enabling data streaming")
        
        # Optimize memory cleanup frequency
        if 'memory_optimization' in eval_config:
            memory_config = eval_config.memory_optimization
            if self.gpu_count >= 4 and memory_config.cleanup_frequency < 4:
                memory_config.cleanup_frequency = 4
                logger.info("Optimized memory cleanup frequency for multi-GPU")
    
    def _validate_dataset_sizes(self, config: DictConfig) -> None:
        """Validate dataset sizes for evaluation."""
        eval_config = config.evaluation
        
        teacher_size = eval_config.teacher_dataset.size
        student_size = eval_config.student_dataset.size
        max_checkpoints = eval_config.get('max_checkpoints', 10)
        
        # Warn about very large datasets
        if teacher_size > 5000 or student_size > 5000:
            logger.warning(f"Large dataset sizes ({teacher_size}, {student_size}) will take significant time")
            logger.warning("Consider using smaller sizes for faster iteration")
        
        # Warn about memory implications for multiple checkpoints
        if max_checkpoints > 5 and (teacher_size > 1000 or student_size > 1000):
            logger.warning(f"Evaluating {max_checkpoints} checkpoints with large datasets may cause memory issues")
            logger.warning("Models are properly cleaned up between checkpoints, but consider:")
            logger.warning("  • Reducing dataset sizes for faster iteration")
            logger.warning("  • Using max_checkpoints to limit evaluation")
        
        # Warn about mismatched sizes
        if abs(teacher_size - student_size) > min(teacher_size, student_size) * 0.5:
            logger.warning(f"Teacher ({teacher_size}) and student ({student_size}) dataset sizes are very different")
            logger.warning("This may affect evaluation quality")
        
        # Validate few-shot settings
        if eval_config.few_shot.enabled:
            n_shot = eval_config.few_shot.n_shot
            if n_shot > 10:
                logger.warning(f"Large few-shot size ({n_shot}) may cause context length issues")
            if n_shot > min(teacher_size, student_size) * 0.1:
                logger.warning(f"Few-shot size ({n_shot}) is large relative to dataset sizes")
    
    def _log_optimizations(self, original_config: DictConfig, optimized_config: DictConfig) -> None:
        """Log any optimizations made to the configuration."""
        changes = []
        
        # Check for changes in key parameters
        original_eval = original_config.evaluation
        optimized_eval = optimized_config.evaluation
        
        if original_eval.batch_size != optimized_eval.batch_size:
            changes.append(f"batch_size: {original_eval.batch_size} → {optimized_eval.batch_size}")
        
        if original_eval.vllm.tensor_parallel_size != optimized_eval.vllm.tensor_parallel_size:
            changes.append(f"tensor_parallel_size: {original_eval.vllm.tensor_parallel_size} → {optimized_eval.vllm.tensor_parallel_size}")
        
        if original_eval.vllm.gpu_memory_utilization != optimized_eval.vllm.gpu_memory_utilization:
            changes.append(f"gpu_memory_utilization: {original_eval.vllm.gpu_memory_utilization} → {optimized_eval.vllm.gpu_memory_utilization}")
        
        if changes:
            logger.info("Configuration optimizations applied:")
            for change in changes:
                logger.info(f"  • {change}")
        else:
            logger.info("No configuration optimizations needed")
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """Get a summary of hardware capabilities."""
        return {
            'gpu_count': self.gpu_count,
            'gpu_memory_total_gb': self.gpu_memory_total / (1024**3),
            'gpu_memory_per_device_gb': self.gpu_memory_total / (1024**3) / max(1, self.gpu_count),
            'system_memory_gb': self.system_memory / (1024**3),
            'cuda_available': torch.cuda.is_available(),
            'recommendations': self._get_recommendations()
        }
    
    def _get_recommendations(self) -> Dict[str, str]:
        """Get hardware-specific recommendations."""
        recommendations = {}
        
        if self.gpu_count == 0:
            recommendations['warning'] = "No GPUs detected. Evaluation will be very slow on CPU."
        elif self.gpu_count == 1:
            recommendations['batch_size'] = "Use batch_size 4-8 for single GPU"
            recommendations['tensor_parallel'] = "Set tensor_parallel_size=1"
        elif self.gpu_count >= 4:
            recommendations['batch_size'] = "Use batch_size 16-32 for multi-GPU"
            recommendations['tensor_parallel'] = f"Set tensor_parallel_size={min(4, self.gpu_count)}"
            recommendations['optimization'] = "Enable parallel processing and chunked prefill"
        
        gpu_memory_per_device = self.gpu_memory_total / max(1, self.gpu_count)
        if gpu_memory_per_device < 8 * 1024**3:
            recommendations['memory'] = "Low GPU memory detected. Use smaller batch sizes and reduce max_model_len"
        elif gpu_memory_per_device > 32 * 1024**3:
            recommendations['memory'] = "High GPU memory available. Can use larger batch sizes and longer sequences"
        
        return recommendations

def validate_config(config: DictConfig) -> DictConfig:
    """
    Convenience function to validate and optimize configuration.
    
    Args:
        config: OmegaConf configuration
        
    Returns:
        Optimized configuration
    """
    validator = ConfigValidator()
    
    # Print hardware summary
    hardware_summary = validator.get_hardware_summary()
    logger.info("Hardware Summary:")
    logger.info(f"  GPUs: {hardware_summary['gpu_count']}")
    logger.info(f"  GPU Memory: {hardware_summary['gpu_memory_total_gb']:.1f}GB total")
    logger.info(f"  System Memory: {hardware_summary['system_memory_gb']:.1f}GB")
    
    if hardware_summary['recommendations']:
        logger.info("Hardware Recommendations:")
        for key, value in hardware_summary['recommendations'].items():
            logger.info(f"  {key}: {value}")
    
    # Validate and optimize configuration
    return validator.validate_and_optimize_config(config)