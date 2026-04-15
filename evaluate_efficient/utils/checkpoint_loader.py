import os
import torch
import glob
import shutil
from typing import Dict, List
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetensors.torch import save_file
import logging

logger = logging.getLogger(__name__)

class CheckpointLoader:
    """
    Utility class for loading FSDP checkpoints and converting them to HuggingFace format.
    Based on the simple load_fsdp_to_hf.py pattern.
    """
    
    def __init__(self, base_model_path: str):
        """
        Initialize the checkpoint loader.
        
        Args:
            base_model_path: Path to the base HuggingFace model used for training
        """
        self.base_model_path = base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
    def _auto_detect_world_size(self, checkpoint_path: str) -> int:
        """
        Auto-detect world size from checkpoint files.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Detected world size
        """
        # Look for pattern model_world_size_X_rank_Y.pt
        import re
        
        logger.info(f"Attempting to auto-detect world_size from: {checkpoint_path}")
        
        # Check if the path exists
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
            return 2
        
        # List what's actually in the directory
        try:
            items = os.listdir(checkpoint_path)
            logger.info(f"Directory contents ({len(items)} items): {items[:10]}")  # Show first 10 items
        except Exception as e:
            logger.warning(f"Cannot list directory {checkpoint_path}: {e}")
            return 2
        
        # First try to find model files specifically
        model_files = glob.glob(os.path.join(checkpoint_path, "model_world_size_*.pt"))
        logger.info(f"Found {len(model_files)} model_world_size_*.pt files in main directory")
        
        # Also get all .pt files for debugging
        all_pt_files = glob.glob(os.path.join(checkpoint_path, "*.pt"))
        logger.info(f"Found {len(all_pt_files)} total .pt files in main directory")
        
        checkpoint_files = model_files if model_files else all_pt_files
        
        # If no model files found, try the actor subdirectory
        if not model_files:
            actor_path = os.path.join(checkpoint_path, "actor")
            logger.info(f"Checking actor subdirectory: {actor_path}")
            if os.path.exists(actor_path):
                try:
                    actor_items = os.listdir(actor_path)
                    logger.info(f"Actor directory contents ({len(actor_items)} items): {actor_items[:10]}")
                except Exception as e:
                    logger.warning(f"Cannot list actor directory: {e}")
                
                actor_model_files = glob.glob(os.path.join(actor_path, "model_world_size_*.pt"))
                actor_all_files = glob.glob(os.path.join(actor_path, "*.pt"))
                logger.info(f"Found {len(actor_model_files)} model files and {len(actor_all_files)} total .pt files in actor subdirectory")
                
                checkpoint_files = actor_model_files if actor_model_files else actor_all_files
            else:
                logger.info(f"Actor subdirectory does not exist: {actor_path}")
        
        logger.info(f"Total checkpoint files found: {len(checkpoint_files)}")
        if checkpoint_files:
            logger.info(f"Example files: {[os.path.basename(f) for f in checkpoint_files[:3]]}")
        
        world_sizes = set()
        for file in checkpoint_files:
            filename = os.path.basename(file)
            # Match pattern: model_world_size_X_rank_Y.pt
            match = re.search(r'model_world_size_(\d+)_rank_\d+\.pt', filename)
            if match:
                world_size_found = int(match.group(1))
                world_sizes.add(world_size_found)
                logger.info(f"Found world_size {world_size_found} in file: {filename}")
        
        if world_sizes:
            detected_world_size = list(world_sizes)[0]
            logger.info(f"Auto-detected world_size: {detected_world_size}")
            return detected_world_size
        
        # Fallback: count rank files
        rank_files = [f for f in checkpoint_files if "rank_" in f]
        logger.info(f"Found {len(rank_files)} files with 'rank_' in name")
        
        if rank_files:
            max_rank = -1
            for file in rank_files:
                match = re.search(r'rank_(\d+)', os.path.basename(file))
                if match:
                    rank_found = int(match.group(1))
                    max_rank = max(max_rank, rank_found)
                    logger.info(f"Found rank {rank_found} in file: {os.path.basename(file)}")
            if max_rank >= 0:
                detected_world_size = max_rank + 1
                logger.info(f"Auto-detected world_size from rank files: {detected_world_size}")
                return detected_world_size
        
        # Debug: show what files we actually found
        logger.warning(f"Could not auto-detect world_size from {len(checkpoint_files)} files:")
        for file in checkpoint_files[:5]:  # Show first 5 files
            logger.warning(f"  Found file: {os.path.basename(file)}")
        
        # Default fallback
        logger.warning("Could not auto-detect world_size, defaulting to 2")
        return 2

    def load_fsdp_checkpoint(self, checkpoint_path: str, world_size: int = None) -> Dict:
        """
        Load FSDP distributed checkpoint and merge state dictionaries.
        Simple implementation based on the reference script.
        
        Args:
            checkpoint_path: Path to FSDP checkpoint directory
            world_size: Number of ranks/processes used during training (auto-detected if None)
            
        Returns:
            Merged state dictionary
        """
        logger.info(f"Loading FSDP checkpoint from {checkpoint_path}")
        
        # Auto-detect world size if not provided
        if world_size is None:
            world_size = self._auto_detect_world_size(checkpoint_path)
        
        # Use defaultdict(list) pattern from reference script
        state_dict = defaultdict(list)
        
        # Load each rank's checkpoint
        for rank in range(world_size):
            filepath = os.path.join(checkpoint_path, f"model_world_size_{world_size}_rank_{rank}.pt")
            
            # If file not found directly, check if we need to look in actor subdirectory
            if not os.path.exists(filepath):
                actor_filepath = os.path.join(checkpoint_path, "actor", f"model_world_size_{world_size}_rank_{rank}.pt")
                if os.path.exists(actor_filepath):
                    filepath = actor_filepath
                    logger.info(f"Found checkpoint in actor subdirectory: {filepath}")
                else:
                    logger.warning(f"Checkpoint file not found: {filepath}")
                    logger.warning(f"Also checked: {actor_filepath}")
                    continue
                
            logger.info(f"Loading {filepath}")
            this_state_dict = torch.load(filepath, map_location="cpu", weights_only=False)
            
            for key, value in this_state_dict.items():
                # Handle FSDP tensors that may need .to_local()
                if hasattr(value, 'to_local'):
                    value = value.to_local()
                state_dict[key].append(value)
        
        # Concatenate tensors from different ranks
        merged_state_dict = {}
        for key in state_dict:
            merged_state_dict[key] = torch.cat(state_dict[key], dim=0)
        
        logger.info(f"Merged state dict with {len(merged_state_dict)} parameters")
        return merged_state_dict
    
    def _has_safetensors_conversion(self, checkpoint_path: str) -> bool:
        """Check if SafeTensors conversion already exists in the same directory."""
        return (os.path.exists(os.path.join(checkpoint_path, "model.safetensors")) or
                os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")) or
                os.path.exists(os.path.join(checkpoint_path, "config.json")))
    
    def _convert_fsdp_to_safetensors_inplace(self, checkpoint_path: str) -> str:
        """
        Convert FSDP checkpoint to SafeTensors format and save to parent global_step directory.
        Deletes the actor subdirectory after conversion.
        
        Args:
            checkpoint_path: Path to FSDP checkpoint directory (e.g., /path/to/global_step_100/actor)
            
        Returns:
            Absolute path to parent directory containing SafeTensors (e.g., /path/to/global_step_100)
        """
        # Convert to absolute path for parallel execution safety
        checkpoint_path = os.path.abspath(checkpoint_path)
        logger.info(f"Converting FSDP checkpoint to SafeTensors: {checkpoint_path}")
        
        # Determine the target directory (parent of actor directory)
        if os.path.basename(checkpoint_path) == "actor":
            target_dir = os.path.dirname(checkpoint_path)
            logger.info(f"Saving SafeTensors to parent directory: {target_dir}")
        else:
            target_dir = checkpoint_path
            logger.info(f"Saving SafeTensors to same directory: {target_dir}")
        
        # Ensure target directory is also absolute
        target_dir = os.path.abspath(target_dir)
        
        # Use the existing convert_to_huggingface method which handles FSDP correctly
        self.convert_to_huggingface(checkpoint_path, target_dir)
        
        # Remove the actor subdirectory if it exists
        if os.path.basename(checkpoint_path) == "actor" and os.path.exists(checkpoint_path):
            logger.info(f"Removing actor subdirectory: {checkpoint_path}")
            shutil.rmtree(checkpoint_path)
        
        logger.info(f"Successfully converted FSDP checkpoint to SafeTensors in: {target_dir}")
        return target_dir
    
    def _cleanup_fsdp_files(self, checkpoint_path: str):
        """Delete FSDP .pt files to save space while keeping the directory structure."""
        logger.info(f"Cleaning up FSDP files in {checkpoint_path}")
        
        # Find and delete model_world_size_*.pt files in the current directory
        fsdp_files = glob.glob(os.path.join(checkpoint_path, "model_world_size_*.pt"))
        
        for fsdp_file in fsdp_files:
            try:
                os.remove(fsdp_file)
                logger.info(f"Deleted FSDP file: {fsdp_file}")
            except Exception as e:
                logger.warning(f"Could not delete {fsdp_file}: {e}")
        
        logger.info(f"Cleaned up {len(fsdp_files)} FSDP files from {checkpoint_path}")
    
    def convert_to_huggingface(self, checkpoint_path: str, output_path: str) -> str:
        """
        Convert FSDP checkpoint to HuggingFace format.
        Simple implementation based on the reference script.
        
        Args:
            checkpoint_path: Path to FSDP checkpoint directory
            output_path: Output directory for converted model
            
        Returns:
            Absolute path to converted model
        """
        # Convert to absolute paths for parallel execution safety
        checkpoint_path = os.path.abspath(checkpoint_path)
        output_path = os.path.abspath(output_path)
        
        logger.info(f"Converting FSDP checkpoint to HuggingFace format")
        logger.info(f"Source checkpoint: {checkpoint_path}")
        logger.info(f"Target output: {output_path}")
        
        # Load merged state dictionary
        state_dict = self.load_fsdp_checkpoint(checkpoint_path)
        
        # Load config and create model from config (avoids distributed issues)
        config = AutoConfig.from_pretrained(self.base_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config)
        
        # Load state dict into model
        model.load_state_dict(state_dict)
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(output_path, max_shard_size="5GB")
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Model saved to {output_path}")
        return output_path
    
    def _is_huggingface_model(self, path: str) -> bool:
        """
        Check if a path is a HuggingFace model directory.
        
        Args:
            path: Path to check
            
        Returns:
            True if it's a HuggingFace model directory
        """
        if not os.path.exists(path):
            return False
            
        # Check for typical HuggingFace model files
        hf_files = [
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        # Check if we have config.json (essential for HF models)
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            return False
        
        # Check if we have at least one model file
        model_files = [
            "pytorch_model.bin",
            "model.safetensors"
        ]
        
        for model_file in model_files:
            if os.path.exists(os.path.join(path, model_file)):
                logger.info(f"Detected HuggingFace model at {path} (found config.json and {model_file})")
                return True
                
        # Check for sharded model files
        if (glob.glob(os.path.join(path, "pytorch_model-*.bin")) or 
            glob.glob(os.path.join(path, "model-*.safetensors"))):
            logger.info(f"Detected HuggingFace model at {path} (found config.json and sharded model files)")
            return True
                
        return False
    
    def get_checkpoint_paths(self, checkpoint_dir: str, start_step: int = -1) -> List[str]:
        """
        Get all checkpoint paths from a directory, sorted by step/epoch.
        Automatically finds 'actor' subdirectories in global_step_X directories.
        Also handles HuggingFace model directories.
        
        Args:
            checkpoint_dir: Directory containing checkpoints or HuggingFace model
            start_step: Start from which step (-1 = all checkpoints, 100 = start from step 100)
            
        Returns:
            List of checkpoint paths sorted by training step/epoch, filtered by start_step
        """
            
        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
            return []
        
        # Check if this is a HuggingFace model directory
        if self._is_huggingface_model(checkpoint_dir):
            logger.info(f"Detected HuggingFace model directory: {checkpoint_dir}")
            return [checkpoint_dir]
        
        # Check if this is already a checkpoint directory (contains .pt files)
        if glob.glob(os.path.join(checkpoint_dir, "*.pt")):
            logger.info(f"Single FSDP checkpoint directory: {checkpoint_dir}")
            return [checkpoint_dir]
        
        # Find all checkpoint subdirectories
        checkpoint_paths = []
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path):
                # Check if it's a global_step_X directory with actor subdirectory
                if item.startswith("global_step_"):
                    actor_path = os.path.join(item_path, "actor")
                    logger.info(f"Checking for actor directory: {actor_path}")
                    if os.path.exists(actor_path):
                        actor_files = glob.glob(os.path.join(actor_path, "*.pt"))
                        logger.info(f"Found {len(actor_files)} .pt files in actor directory")
                        if actor_files:
                            checkpoint_paths.append(actor_path)
                            logger.info(f"Added actor checkpoint: {actor_path}")
                        else:
                            logger.warning(f"Actor directory exists but no .pt files found: {actor_path}")
                    else:
                        logger.warning(f"No actor subdirectory found in: {item_path}")
                        # Check if the global_step directory itself contains model files
                        step_files = glob.glob(os.path.join(item_path, "model_world_size_*.pt"))
                        if step_files:
                            logger.info(f"Found {len(step_files)} model files directly in global_step directory")
                            checkpoint_paths.append(item_path)
                        else:
                            # Check for other common checkpoint files
                            other_files = glob.glob(os.path.join(item_path, "*.pt"))
                            if other_files:
                                logger.info(f"Found {len(other_files)} .pt files in global_step directory")
                                checkpoint_paths.append(item_path)
                # Check if it contains checkpoint files directly (non-global_step directories)
                else:
                    model_files = glob.glob(os.path.join(item_path, "model_world_size_*.pt"))
                    if model_files:
                        checkpoint_paths.append(item_path)
                        logger.info(f"Found checkpoint directory with model files: {item_path}")
        
        # Sort by step number if available
        def extract_step(path):
            # Extract step from either direct path or parent path
            basename = os.path.basename(path)
            if basename == "actor":
                # If this is an actor directory, get step from parent
                parent_basename = os.path.basename(os.path.dirname(path))
                basename = parent_basename
                
            try:
                # Try to extract step number from directory name
                if "global_step_" in basename:
                    return int(basename.split("global_step_")[1].split("_")[0])
                elif "step_" in basename:
                    return int(basename.split("step_")[1].split("_")[0])
                elif "epoch_" in basename:
                    return int(basename.split("epoch_")[1].split("_")[0])
                else:
                    return 0
            except:
                return 0
        
        checkpoint_paths.sort(key=extract_step)
        logger.info(f"Found {len(checkpoint_paths)} checkpoint paths")
        
        # Filter by start_step if specified
        if start_step >= 0:
            filtered_paths = []
            logger.info(f"Filtering checkpoints with start_step={start_step}")
            for path in checkpoint_paths:
                step = extract_step(path)
                logger.info(f"Checkpoint {path} has step {step} (>= {start_step}? {step >= start_step})")
                if step >= start_step:
                    filtered_paths.append(path)
            
            logger.info(f"Filtered checkpoints: {len(filtered_paths)}/{len(checkpoint_paths)} checkpoints starting from step {start_step}")
            if filtered_paths:
                logger.info(f"First checkpoint to evaluate: {filtered_paths[0]}")
                logger.info(f"Last checkpoint to evaluate: {filtered_paths[-1]}")
            return filtered_paths
        else:
            logger.info(f"Using all {len(checkpoint_paths)} checkpoints (start_step={start_step})")
            return checkpoint_paths
    
    def load_model_from_checkpoint(self, checkpoint_path: str) -> AutoModelForCausalLM:
        """
        Load a model from either FSDP checkpoint or HuggingFace model directory.
        Automatically converts FSDP to SafeTensors on first load and deletes FSDP files.
        
        Args:
            checkpoint_path: Path to FSDP checkpoint or HuggingFace model directory
            
        Returns:
            Loaded model
        """
        # Convert to absolute path for parallel execution safety
        checkpoint_path = os.path.abspath(checkpoint_path)
        
        # Check if this is already a HuggingFace model
        if self._is_huggingface_model(checkpoint_path):
            logger.info(f"Loading HuggingFace model directly from: {checkpoint_path}")
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
            return model
        
        # Check if SafeTensors conversion already exists in the same directory
        if self._has_safetensors_conversion(checkpoint_path):
            logger.info(f"Loading existing SafeTensors model from: {checkpoint_path}")
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
            return model
        
        # Convert FSDP to SafeTensors format and load
        logger.info(f"Converting FSDP checkpoint to SafeTensors on first load: {checkpoint_path}")
        converted_path = self._convert_fsdp_to_safetensors_inplace(checkpoint_path)
        
        # Load the newly converted SafeTensors model
        logger.info(f"Loading newly converted SafeTensors model from: {converted_path}")
        model = AutoModelForCausalLM.from_pretrained(converted_path, trust_remote_code=True)
        return model