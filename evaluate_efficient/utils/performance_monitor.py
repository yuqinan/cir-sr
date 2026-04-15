import time
import psutil
import torch
import logging
from typing import Dict, List, Optional
from contextlib import contextmanager
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitor performance metrics during evaluation for multi-GPU optimization.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = {
            'batch_times': [],
            'gpu_stats': [],
            'memory_usage': [],
            'throughput': [],
            'timestamps': []
        }
        self.start_time = None
        self.total_processed = 0
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        logger.info("Performance monitoring started")
        
    def log_batch_metrics(self, batch_size: int, batch_time: float, stage: str = "generation"):
        """
        Log metrics for a batch.
        
        Args:
            batch_size: Size of the batch
            batch_time: Time taken for the batch
            stage: Stage of processing (generation, evaluation, etc.)
        """
        if not self.config.get('performance', {}).get('log_batch_times', False):
            return
            
        throughput = batch_size / batch_time if batch_time > 0 else 0
        
        self.metrics['batch_times'].append({
            'stage': stage,
            'batch_size': batch_size,
            'time': batch_time,
            'throughput': throughput,
            'timestamp': time.time()
        })
        
        self.total_processed += batch_size
        
        logger.info(f"Batch {stage}: {batch_size} items in {batch_time:.2f}s ({throughput:.2f} items/s)")
        
    def log_gpu_stats(self):
        """Log GPU statistics."""
        if not self.config.get('performance', {}).get('log_gpu_stats', False):
            return
            
        if not torch.cuda.is_available():
            return
            
        gpu_stats = []
        for i in range(torch.cuda.device_count()):
            stats = {
                'device': i,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'utilization': self._get_gpu_utilization(i),
                'timestamp': time.time()
            }
            gpu_stats.append(stats)
            
        self.metrics['gpu_stats'].append(gpu_stats)
        
        # Log summary
        total_memory = sum(stat['memory_total'] for stat in gpu_stats)
        total_allocated = sum(stat['memory_allocated'] for stat in gpu_stats)
        avg_utilization = sum(stat['utilization'] for stat in gpu_stats) / len(gpu_stats)
        
        logger.info(f"GPU Stats: {total_allocated / (1024**3):.1f}GB / {total_memory / (1024**3):.1f}GB allocated, {avg_utilization:.1f}% avg utilization")
        
    def log_memory_usage(self):
        """Log system memory usage."""
        if not self.config.get('performance', {}).get('log_memory_usage', False):
            return
            
        memory = psutil.virtual_memory()
        
        memory_stats = {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'timestamp': time.time()
        }
        
        self.metrics['memory_usage'].append(memory_stats)
        
        logger.info(f"Memory Usage: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent:.1f}%)")
        
    def _get_gpu_utilization(self, device_id: int) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return info.gpu
        except:
            return 0.0
            
    def get_throughput_stats(self) -> Dict:
        """Calculate throughput statistics."""
        if not self.start_time:
            return {}
            
        elapsed_time = time.time() - self.start_time
        overall_throughput = self.total_processed / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate batch throughput statistics
        batch_throughputs = [metric['throughput'] for metric in self.metrics['batch_times']]
        
        stats = {
            'overall_throughput': overall_throughput,
            'total_processed': self.total_processed,
            'elapsed_time': elapsed_time,
            'average_batch_throughput': sum(batch_throughputs) / len(batch_throughputs) if batch_throughputs else 0,
            'max_batch_throughput': max(batch_throughputs) if batch_throughputs else 0,
            'min_batch_throughput': min(batch_throughputs) if batch_throughputs else 0
        }
        
        return stats
        
    def get_gpu_efficiency(self) -> Dict:
        """Calculate GPU efficiency metrics."""
        if not self.metrics['gpu_stats']:
            return {}
            
        # Calculate average GPU utilization
        all_utilizations = []
        for gpu_snapshot in self.metrics['gpu_stats']:
            for gpu_stat in gpu_snapshot:
                all_utilizations.append(gpu_stat['utilization'])
        
        # Calculate memory efficiency
        all_memory_usage = []
        for gpu_snapshot in self.metrics['gpu_stats']:
            for gpu_stat in gpu_snapshot:
                usage_percent = (gpu_stat['memory_allocated'] / gpu_stat['memory_total']) * 100
                all_memory_usage.append(usage_percent)
        
        efficiency = {
            'avg_gpu_utilization': sum(all_utilizations) / len(all_utilizations) if all_utilizations else 0,
            'max_gpu_utilization': max(all_utilizations) if all_utilizations else 0,
            'avg_memory_usage': sum(all_memory_usage) / len(all_memory_usage) if all_memory_usage else 0,
            'max_memory_usage': max(all_memory_usage) if all_memory_usage else 0,
            'num_gpus': torch.cuda.device_count()
        }
        
        return efficiency
        
    def save_performance_metrics(self, output_dir: str):
        """Save performance metrics to file."""
        if not self.config.get('performance', {}).get('save_performance_metrics', False):
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Compile comprehensive performance report
        performance_report = {
            'configuration': self.config,
            'throughput_stats': self.get_throughput_stats(),
            'gpu_efficiency': self.get_gpu_efficiency(),
            'detailed_metrics': self.metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"performance_metrics_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(performance_report, f, indent=2)
            
        logger.info(f"Performance metrics saved to {filepath}")
        
        # Print performance summary
        self.print_performance_summary()
        
    def print_performance_summary(self):
        """Print a summary of performance metrics."""
        throughput_stats = self.get_throughput_stats()
        gpu_efficiency = self.get_gpu_efficiency()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"Total processed: {self.total_processed} items")
        print(f"Elapsed time: {throughput_stats.get('elapsed_time', 0):.2f}s")
        print(f"Overall throughput: {throughput_stats.get('overall_throughput', 0):.2f} items/s")
        print(f"Average batch throughput: {throughput_stats.get('average_batch_throughput', 0):.2f} items/s")
        print(f"Max batch throughput: {throughput_stats.get('max_batch_throughput', 0):.2f} items/s")
        
        if gpu_efficiency:
            print(f"\nGPU Efficiency:")
            print(f"  Number of GPUs: {gpu_efficiency['num_gpus']}")
            print(f"  Average GPU utilization: {gpu_efficiency['avg_gpu_utilization']:.1f}%")
            print(f"  Max GPU utilization: {gpu_efficiency['max_gpu_utilization']:.1f}%")
            print(f"  Average memory usage: {gpu_efficiency['avg_memory_usage']:.1f}%")
            print(f"  Max memory usage: {gpu_efficiency['max_memory_usage']:.1f}%")
        
        print("="*60)

@contextmanager
def batch_timer(monitor: PerformanceMonitor, batch_size: int, stage: str = "generation"):
    """Context manager for timing batches."""
    start_time = time.time()
    try:
        yield
    finally:
        batch_time = time.time() - start_time
        monitor.log_batch_metrics(batch_size, batch_time, stage)

def create_performance_monitor(config: Dict) -> PerformanceMonitor:
    """Factory function to create a performance monitor."""
    return PerformanceMonitor(config)