"""
vLLM Model Manager

Handles loading, unloading, and configuration detection for vLLM models.
Separated from teacher_pipeline.py for better organization.
"""

import os
import json
import logging
import torch
import hashlib
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class VLLMModelManager:
    """Manages vLLM model lifecycle and configuration detection."""
    
    def __init__(self, vllm_config: Dict[str, Any], checkpoint_loader):
        """
        Initialize the model manager.
        
        Args:
            vllm_config: vLLM configuration parameters
            checkpoint_loader: CheckpointLoader instance for FSDP conversion
        """
        self.vllm_config = vllm_config
        self.checkpoint_loader = checkpoint_loader
        
        # Current model state
        self.current_model: Optional[LLM] = None
        self.current_model_path: Optional[str] = None
        self.current_checkpoint_path: Optional[str] = None
        self.current_checkpoint_name: Optional[str] = None
        self.model_family: Optional[str] = None
    
    def load_model(self, checkpoint_path: str) -> None:
        """
        Load a model from checkpoint path.
        
        Args:
            checkpoint_path: Path to checkpoint directory or HuggingFace model
        """
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Clean up previous model first
        self.cleanup_current_model()
        
        try:
            # Get HuggingFace model path
            hf_model_path = self._prepare_model_path(checkpoint_path)
            
            # Load model into vLLM
            self._load_vllm_model(hf_model_path)
            
            # Store model info
            self.current_model_path = hf_model_path
            self.current_checkpoint_path = checkpoint_path
            self.current_checkpoint_name = os.path.basename(checkpoint_path)
            self.model_family = self._detect_model_family(hf_model_path)
            
            logger.info(f"Successfully loaded model: {self.current_checkpoint_name}")
            logger.info(f"Detected model family: {self.model_family}")
            
        except Exception as e:
            logger.error(f"Failed to load model {checkpoint_path}: {str(e)}")
            self.cleanup_current_model()
            raise
    
    def _prepare_model_path(self, checkpoint_path: str) -> str:
        """Prepare model path (convert FSDP if needed)."""
        # Convert to absolute path for parallel execution safety
        checkpoint_path = os.path.abspath(checkpoint_path)
        
        if self._is_huggingface_model(checkpoint_path):
            logger.info(f"Using HuggingFace model directly: {checkpoint_path}")
            return checkpoint_path
        
        logger.info("Converting FSDP checkpoint to HuggingFace format in-place")
        
        # Use in-place conversion - this will save to global_step_X directory and remove actor folder
        converted_path = self.checkpoint_loader._convert_fsdp_to_safetensors_inplace(checkpoint_path)
        
        logger.info(f"Converted FSDP checkpoint in-place to: {converted_path}")
        return converted_path
    
    def _load_vllm_model(self, model_path: str) -> None:
        """Load model into vLLM."""
        logger.info(f"Loading model into vLLM: {model_path}")
        
        vllm_kwargs = {
            'model': model_path,
            'trust_remote_code': True,
            'tensor_parallel_size': self.vllm_config['tensor_parallel_size'],
            'gpu_memory_utilization': self.vllm_config['gpu_memory_utilization'],
            'max_model_len': self.vllm_config['max_model_len'],
            'dtype': self.vllm_config['dtype'],
            'enforce_eager': self.vllm_config['enforce_eager'],
            'disable_log_stats': self.vllm_config['disable_log_stats']
        }
        
        # Add optional parameters
        optional_params = ['enable_chunked_prefill', 'max_num_batched_tokens', 'max_num_seqs']
        for param in optional_params:
            if param in self.vllm_config:
                vllm_kwargs[param] = self.vllm_config[param]
        
        self.current_model = LLM(**vllm_kwargs)
        
        # Force cleanup after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_stop_tokens(self) -> List[str]:
        """Get appropriate stop tokens for current model."""
        if not self.current_model_path or not os.path.exists(self.current_model_path):
            logger.warning("No model path available for stop token detection")
            return ['<|im_end|>', '</s>', '<|endoftext|>']
        
        # Extract from config files first
        config_tokens = self._extract_stop_tokens_from_config(self.current_model_path)
        if config_tokens:
            logger.info(f"Using stop tokens from config: {config_tokens}")
            return config_tokens
        
        # Fallback to empirical detection
        logger.info("Config-based detection failed, using empirical detection")
        return self._detect_stop_tokens_empirically()
    
    def _extract_stop_tokens_from_config(self, model_path: str) -> List[str]:
        """Extract stop tokens from model config files."""
        stop_tokens = []
        
        # Check tokenizer_config.json
        tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            try:
                with open(tokenizer_config_path, 'r') as f:
                    config = json.load(f)
                
                stop_fields = ['eos_token', 'pad_token', 'sep_token', 'stop_tokens']
                for field in stop_fields:
                    if field in config:
                        value = config[field]
                        if isinstance(value, str):
                            stop_tokens.append(value)
                        elif isinstance(value, list):
                            stop_tokens.extend([str(v) for v in value if v])
                        elif isinstance(value, dict) and 'content' in value:
                            stop_tokens.append(str(value['content']))
                
            except Exception as e:
                logger.warning(f"Error reading tokenizer config: {e}")
        
        # Remove duplicates
        return list(set([token.strip() for token in stop_tokens if token and token.strip()]))
    
    def _detect_stop_tokens_empirically(self) -> List[str]:
        """Detect stop tokens by test generation."""
        if not self.current_model:
            return ['<|im_end|>', '</s>']
        
        try:
            test_params = SamplingParams(temperature=0.0, max_tokens=50, stop=None)
            outputs = self.current_model.generate(["Hello"], test_params, use_tqdm=False)
            generated_text = outputs[0].outputs[0].text
            
            common_stops = ['<|im_end|>', '</s>', '<|endoftext|>', '<end_of_turn>']
            detected = [stop for stop in common_stops if stop in generated_text]
            
            return detected if detected else ['<|im_end|>', '</s>', '<|endoftext|>']
            
        except Exception as e:
            logger.warning(f"Error in empirical detection: {e}")
            return ['<|im_end|>', '</s>', '<|endoftext|>']
    
    def _detect_model_family(self, model_path: str) -> str:
        """Detect model family from config.json."""
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_path):
            logger.warning(f"No config.json found at {config_path}")
            return "unknown"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_type = config.get('model_type', '').lower()
            architectures = config.get('architectures', [])
            model_name = config.get('_name_or_path', '').lower()
            
            identifiers = [model_type] + [arch.lower() for arch in architectures] + [model_name]
            identifiers_str = ' '.join(identifiers)
            
            # Family detection
            if any(kw in identifiers_str for kw in ['qwen', 'qwen2']):
                return 'qwen'
            elif any(kw in identifiers_str for kw in ['llama', 'llamaforcausallm']):
                return 'llama'
            elif any(kw in identifiers_str for kw in ['deepseek']):
                return 'deepseek'
            elif any(kw in identifiers_str for kw in ['gemma']):
                return 'gemma'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Error reading config: {e}")
            return 'unknown'
    
    def _is_huggingface_model(self, path: str) -> bool:
        """Check if path is a HuggingFace model directory."""
        if not os.path.exists(path):
            return False
        
        hf_files = ["config.json", "pytorch_model.bin", "model.safetensors", "tokenizer.json"]
        return any(os.path.exists(os.path.join(path, f)) for f in hf_files)
    
    def _extract_run_type_and_task(self, checkpoint_path: str) -> tuple:
        """Extract run type and task from checkpoint path."""
        parts = checkpoint_path.split('/')
        
        try:
            checkpoints_idx = parts.index('checkpoints')
            if checkpoints_idx + 2 < len(parts):
                run_type = parts[checkpoints_idx + 1]
                task_name = parts[checkpoints_idx + 2]
                return run_type, task_name
        except ValueError:
            pass
        
        # Fallback detection
        run_type = "unknown"
        if "/direct/" in checkpoint_path:
            run_type = "direct"
        elif "/grpo/" in checkpoint_path:
            run_type = "grpo"
        
        return run_type, "unknown_task"
    
    def cleanup_current_model(self) -> None:
        """Clean up current model from memory."""
        if self.current_model is not None:
            logger.info(f"Cleaning up model: {self.current_checkpoint_name}")
            
            del self.current_model
            self.current_model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            import gc
            gc.collect()
            
            logger.info("Model cleanup completed")
    
    def _cleanup_temp_model_files(self) -> None:
        """Clean up temporary model files."""
        if not self.current_checkpoint_path:
            return
        
        import shutil
        import glob
        
        path_hash = hashlib.md5(self.current_checkpoint_path.encode()).hexdigest()[:8]
        run_type, task_name = self._extract_run_type_and_task(self.current_checkpoint_path)
        temp_pattern = f"/nlp/scr/qinanyu/rl-explanations/temp/teacher_model_{run_type}_{task_name}_{path_hash}_*"
        
        cleaned_count = 0
        for temp_dir in glob.glob(temp_pattern):
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Could not clean up {temp_dir}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary directories")