# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT Trainer with Online Thinking Generation

This trainer extends FSDPSFTTrainer to support online thinking generation during training.
The model generates thinking tokens between prompts and answers, but only receives
gradient signals from the answer tokens.

Training flow per batch:
1. Generate thinking: [prompt] + <think> → [reasoning tokens]</think> (no grad)
2. Reconstruct: [prompt] + [thinking] + <answer>[answer]</answer>
3. Train: Forward + backward with loss only on answer
"""

import os
import time
import torch
from contextlib import contextmanager
from tensordict import TensorDict
from omegaconf import OmegaConf
import hydra

# Import everything from the base trainer
from trainers.verl_sft_trainer import (
    FSDPSFTTrainer,
    create_sft_dataset,
    logger,
)
from verl.utils.device import is_cuda_available, is_npu_available
from verl.utils.distributed import initialize_global_process_group, destroy_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.tracking import Tracking
from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
from verl.utils.model import compute_position_id_with_mask
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM, AutoTokenizer


class FSDPSFTThinkingTrainer(FSDPSFTTrainer):
    """
    Extends FSDPSFTTrainer with online thinking generation capabilities.

    Key differences from parent:
    - Overrides training_step() to generate thinking before training
    - Adds helper methods for thinking generation and batch reconstruction
    - Loss is only applied to answer tokens, not thinking tokens
    """

    @contextmanager
    def _without_ulysses_patches(self):
        """
        Context manager to temporarily restore original flash attention during generation.

        Monkey patches modify _flash_attention_forward for Ulysses sequence parallel,
        which breaks .generate(). This temporarily restores the original implementation.

        Usage:
            with self._without_ulysses_patches():
                generated = self.fsdp_model.generate(...)
        """
        # Import original and patched versions
        try:
            from transformers.modeling_flash_attention_utils import _flash_attention_forward as original_flash_attn
        except ImportError:
            # Fallback for older transformers versions
            try:
                from transformers.models.llama.modeling_llama import _flash_attention_forward as original_flash_attn
            except ImportError:
                print("Warning: Could not import original flash attention, skipping patch removal")
                yield
                return

        try:
            from verl.models.transformers.monkey_patch import _ulysses_flash_attention_forward
        except ImportError:
            print("Warning: Could not import Ulysses flash attention, skipping patch removal")
            yield
            return

        # Save patched versions and restore originals
        saved_methods = {}
        module_count = 0

        for module in self.model.modules():
            if hasattr(module, '_flash_attention_forward'):
                module_id = id(module)
                saved_methods[module_id] = module._flash_attention_forward
                module._flash_attention_forward = original_flash_attn
                module_count += 1

        if module_count > 0:
            print(f">>> Temporarily restored original flash attention in {module_count} modules for generation <<<")

        try:
            yield
        finally:
            # Restore Ulysses patches
            restored_count = 0
            for module in self.model.modules():
                module_id = id(module)
                if module_id in saved_methods:
                    module._flash_attention_forward = saved_methods[module_id]
                    restored_count += 1

            if restored_count > 0:
                print(f">>> Restored Ulysses patches in {restored_count} modules after generation <<<")

    def _load_fsdp_model_for_testing(self, model_path):
        """
        Load a fresh FSDP model for generation testing (simpler than parent trainer).
        Based on working fsdp_vs_baseline_compare.py script.
        """
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

        rank = dist.get_rank() if dist.is_initialized() else 0
        device = torch.device(f"cuda:{rank}")

        # Load model directly to GPU (no meta tensors, no monkey patches)
        raw_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device)

        # Wrap with FSDP using simple config (like working script)
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32
        )

        fsdp_model = FSDP(
            raw_model,
            use_orig_params=True,  # Preserve original params (working script uses this)
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            device_id=device.index,
            sync_module_states=True,
            forward_prefetch=False,
        )

        return fsdp_model

    def __init__(self, config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset):
        super().__init__(config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset)

        # Get thinking generation config
        self.thinking_config = getattr(config, "thinking_generation", {})
        self.max_thinking_tokens = self.thinking_config.get("max_new_tokens", 200)
        self.thinking_temperature = self.thinking_config.get("temperature", 1.0)
        self.thinking_top_p = self.thinking_config.get("top_p", 0.9)
        self.thinking_do_sample = self.thinking_config.get("do_sample", True)

        # Debug counter - print examples for first 3 steps
        self.debug_step_counter = 0
        self.debug_print_steps = 50

        if self.device_mesh.get_rank() == 0:
            print(f"Thinking generation enabled:")
            print(f"  max_new_tokens: {self.max_thinking_tokens}")
            print(f"  temperature: {self.thinking_temperature}")
            print(f"  top_p: {self.thinking_top_p}")
            print(f"  do_sample: {self.thinking_do_sample}")
            print(f"  debug: Will print first {self.debug_print_steps} examples")
            print(f"  Using multi-token tag encoding (no special tokens added)")
        '''
            prompt = torch.tensor([[151644,   8948,    198,     32,  10435,   1948,   2657,    323,  21388,
            13,    576,   1196,  17064,    264,   3405,     11,    323,    279,
         21388,  67477,    432,    624,    785,  17847,   1156,  15482,    911,
           279,  32711,   1882,    304,    279,   3971,    323,   1221,   5707,
           279,   1196,    448,    279,   4226,     13,    576,  32711,   1882,
           323,   4226,    525,  43810,   2878,    366,  26865,     29,    690,
         26865,     29,    323,    366,   9217,     29,    690,   9217,     29,
          9492,     11,  15576,     11,    600,   1734,   2572,    366,  26865,
            29,  32711,   1882,   1588,    690,  26865,    397,     27,   9217,
            29,   9217,   1588,    522,   9217,    397,   5404,    537,  10339,
           697,  32711,   4766,    279,   4226,   9492,     11,   3410,   1172,
           279,   1590,   4226,     13,   3197,    458,   3110,    374,   3897,
            11,    498,   1265,  25470,   1795,    279,   3561,    315,    279,
          2550,     14,   9217,    304,    429,   3110,    624, 151645,    198,
        151644,    872,    198,  22043,    264,   6172,     11,    697,   2618,
           374,    311,   6923,    264,   1140,    315,   5424,    304,  41097,
          1973,     11,   5916,    504,    279,   1909,   7950,   2392,    382,
           785,  41097,   1973,    374,  65670,     11,   5916,    504,    279,
          1909,   7950,   9131,     13,   4398,  23638,    510,     12,   5145,
           504,    279,   1909,   7950,   9131,    323,   3271,   1290,    624,
            12,  14561,   1495,   6974,    279,   5622,   6701,   9131,    624,
            12,  14561,   2115,   6974,    279,   5622,   7950,   9131,    624,
            12,  14561,    705,   6974,    279,   1909,   6701,   9131,    624,
            12,  44801,    279,   7354,    369,    279,   9179,   5424,    315,
           279,   6172,   3080,   1449,   4343,    374,  11994,    382,   7771,
          2550,   1265,    387,    264,   3550,  72692,   1140,    315,  25780,
            11,    384,   1302,     13,    220,     16,    220,     17,    220,
            18,    220,     19,    220,     20,    220,     21,    271,   2461,
           279,   6172,   3685,     11,   1128,    374,    279,   1140,    315,
          5424,    304,  41097,   1973,   5267,     22,    220,     21,    198,
            21,    220,     21,    198, 151645,    198, 151644,  77091,    198,
         13708,    766,     29]]).cuda()

            # Compare three models:
            # 1. self.fsdp_model (from parent trainer with monkey patches, use_orig_params=False)
            # 2. fresh_fsdp_model (simple loading like working script, use_orig_params=True)
            # 3. regular model (non-FSDP baseline)

            print("\n" + "="*80)
            print("Generation Comparison Test")
            print("="*80)

            # Load fresh FSDP model using simple method
            print(">>> Loading fresh FSDP model (use_orig_params=True, no monkey patches) <<<")
            fresh_fsdp_model = self._load_fsdp_model_for_testing(config.model.partial_pretrain)

            # Load regular comparison model
            print(">>> Loading regular non-FSDP model <<<")
            regular_model = AutoModelForCausalLM.from_pretrained(
                config.model.partial_pretrain,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            ).cuda().eval()

            # Test 1: Generate from self.fsdp_model (parent trainer's model)
            print("\n=== Test 1: self.fsdp_model (parent trainer) ===")
            print(f"Strategy: {config.model.strategy}")
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            self.fsdp_model.eval()

            # FSDP2 fix: disable resharding and unshard before generation
            # Problem: with reshard_after_forward=True, parameters get resharded after EVERY forward pass
            # But .generate() does multiple forward passes (one per token), so we need to keep params unsharded
            is_fsdp2 = config.model.strategy == "fsdp2"
            if is_fsdp2:
                print(">>> FSDP2: Disabling reshard_after_forward for generation <<<")
                # FSDP2 wraps submodules, so we need to iterate and set property on each
                for module in self.fsdp_model.modules():
                    if hasattr(module, 'set_reshard_after_forward'):
                        module.set_reshard_after_forward(False)
                print(">>> FSDP2: Unsharding parameters <<<")
                for module in self.fsdp_model.modules():
                    if hasattr(module, 'unshard'):
                        module.unshard()

            with torch.no_grad():
                out_parent = self.fsdp_model.generate(
                    prompt,
                    top_p=1,
                    temperature=1.0,
                    max_new_tokens=400,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # FSDP2 fix: restore resharding behavior
            if is_fsdp2:
                print(">>> FSDP2: Re-enabling reshard_after_forward <<<")
                for module in self.fsdp_model.modules():
                    if hasattr(module, 'set_reshard_after_forward'):
                        module.set_reshard_after_forward(True)

            print(f"Parent FSDP output:")
            print(self.tokenizer.decode(out_parent[0], skip_special_tokens=False))

            # Test 2: Generate from fresh_fsdp_model (simple loading)

            # Test 3: Generate from regular model
            print("\n=== Test 3: regular_model (non-FSDP baseline) ===")
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            with torch.no_grad():
                out_regular = regular_model.generate(
                    prompt,
                    top_p=1,
                    temperature=1.0,
                    max_new_tokens=400,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            print(f"Regular model output:")
            print(self.tokenizer.decode(out_regular[0], skip_special_tokens=False))

            # Compare results
            print("\n" + "="*80)     
            print(f"  Parent FSDP == Regular:     {torch.equal(out_parent[0], out_regular[0])}")
            print("="*80)

            # Clean up
            del fresh_fsdp_model, regular_model
            torch.cuda.empty_cache()
        '''
    def _print_debug_example(self, original_batch, thinking_tokens, new_batch, prompt_text_from_gen):
        """
        Print debugging information for the first sample in the batch.
        Shows original sequence, generated thinking, and reconstructed sequence.

        Args:
            prompt_text_from_gen: The prompt text used for generation (with <think> tag added)
        """
        print("\n" + "="*80)
        print(f"DEBUG: Training Step {self.debug_step_counter + 1}")
        print("="*80)

        # Print first sample (index 0)
        sample_idx = 0

        print("\n0. PROMPT TEXT (used for generation with <think> tag):")
        print("-" * 80)
        print(f"Text: {prompt_text_from_gen}")
        print(f"Length: {len(self.tokenizer.encode(prompt_text_from_gen, add_special_tokens=False))} tokens")

        # Original sequence
        orig_ids = original_batch["input_ids"][sample_idx]
        orig_mask = original_batch["loss_mask"][sample_idx]
        orig_attn = original_batch["attention_mask"][sample_idx]

        # Remove padding
        valid_len = orig_attn.sum().item()
        orig_ids_valid = orig_ids[:valid_len]
        orig_mask_valid = orig_mask[:valid_len]

        print("\n1. ORIGINAL SEQUENCE (from dataset):")
        print("-" * 80)
        orig_text = self.tokenizer.decode(orig_ids_valid, skip_special_tokens=False)
        print(f"Text: {orig_text}")
        #print(f"\nTokens: {orig_ids_valid.tolist()}")
        #print(f"Loss mask: {orig_mask_valid.tolist()}")
        #loss_positions = (orig_mask_valid > 0).nonzero(as_tuple=True)[0].tolist()
        #print(f"Loss applied at positions: {loss_positions}")

        # Generated thinking
        print("\n2. GENERATED THINKING:")
        print("-" * 80)
        thinking_ids = thinking_tokens[sample_idx]
        thinking_text = self.tokenizer.decode(thinking_ids, skip_special_tokens=False)
        print(f"Text: {thinking_text}")
        #print(f"Tokens: {thinking_ids.tolist()}")
        #print(f"Length: {len(thinking_ids)} tokens")

        # Reconstructed sequence
        new_ids = new_batch["input_ids"][sample_idx]
        new_mask = new_batch["loss_mask"][sample_idx]
        new_attn = new_batch["attention_mask"][sample_idx]

        # Remove padding
        valid_len_new = new_attn.sum().item()
        new_ids_valid = new_ids[:valid_len_new]
        new_mask_valid = new_mask[:valid_len_new]

        print("\n3. RECONSTRUCTED SEQUENCE (for training):")
        print("-" * 80)
        new_text = self.tokenizer.decode(new_ids_valid, skip_special_tokens=False)
        print(f"Text: {new_text}")
        #print(f"\nTokens: {new_ids_valid.tolist()}")
        #print(f"Loss mask: {new_mask_valid.tolist()}")

        # Show which parts have loss
        '''
        loss_positions_new = (new_mask_valid > 0).nonzero(as_tuple=True)[0].tolist()
        print(f"Loss applied at positions: {loss_positions_new}")

        # Decode tokens with loss
        tokens_with_loss = new_ids_valid[new_mask_valid > 0]
        text_with_loss = self.tokenizer.decode(tokens_with_loss, skip_special_tokens=False)
        print(f"\nTokens WITH loss: {text_with_loss}")

        # Decode tokens without loss
        tokens_without_loss = new_ids_valid[new_mask_valid == 0]
        text_without_loss = self.tokenizer.decode(tokens_without_loss, skip_special_tokens=False)
        print(f"Tokens WITHOUT loss: {text_without_loss}")
        '''

        print("\n4. LOSS PATTERN VISUALIZATION:")
        print("-" * 80)
        # Create a visual representation
        visual = []
        for i, (token_id, has_loss) in enumerate(zip(new_ids_valid.tolist(), new_mask_valid.tolist())):
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            marker = "✅" if has_loss else "❌"
            visual.append(f"{marker} {token_text}")

        # Print in chunks to avoid too long lines
        chunk_size = 10
        for i in range(0, len(visual), chunk_size):
            chunk = visual[i:i+chunk_size]
            print(" | ".join(chunk))

        print("="*80 + "\n")

    def _get_prompt_lengths(self, batch):
        """
        Find where prompt ends and response begins by checking loss_mask.

        The original loss_mask has 0 for prompt region and 1 for response region.
        We find the first position where loss_mask becomes 1.

        Args:
            batch: TensorDict with "loss_mask" key [batch_size, seq_len]

        Returns:
            List of prompt lengths (int) for each sample in batch
        """
        loss_mask = batch["loss_mask"]  # [batch_size, seq_len]
        prompt_lengths = []

        for i in range(loss_mask.shape[0]):
            # Find first position with loss > 0 (start of response)
            response_positions = (loss_mask[i] > 0).nonzero(as_tuple=True)[0]
            if len(response_positions) > 0:
                # Prompt ends just before first response position
                # Note: loss_mask is already shifted by 1 (see dataset __getitem__)
                prompt_lengths.append(response_positions[0].item())
            else:
                # Entire sequence is prompt (shouldn't happen in normal training)
                prompt_lengths.append(loss_mask.shape[1])

        return prompt_lengths

    def _generate_thinking_for_batch(self, batch):
        """
        Generate thinking tokens for each prompt in the batch.

        Process:
        1. Decode prompts to text and add <think> tag if not present
        2. Generate thinking tokens
        3. Decode ONLY generated portion, find/add </think>
        4. Re-encode full thinking text to tokens

        Args:
            batch: Original batch from dataloader

        Returns:
            Tuple of (thinking_tokens_list, first_prompt_text) for debugging
        """
        device = batch["input_ids"].device

        # Extract prompts
        prompt_lengths = self._get_prompt_lengths(batch)
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]

        # Prepare prompt inputs for generation
        prompt_inputs = []
        first_prompt_text = None  # Save for debugging

        for i in range(batch_size):
            prompt_len = prompt_lengths[i]
            prompt_ids = input_ids[i, :prompt_len]

            # Debug: Print what we're decoding (only for first sample, first step)

            # Decode prompt to text
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)

            # Debug: Show decoded result
            if i == 0 and self.device_mesh.get_rank() == 0 and self.debug_step_counter == 0:
                print(f"Decoded prompt text (first 200 chars): {prompt_text[:200]}")
                print(f"Decoded prompt text (last 200 chars): {prompt_text[-200:]}")
                print(f"=== END DEBUG ===\n")

            # Don't add <think> to the prompt - let the model generate it
            # The model is trained on data where <think> comes right after <|im_start|>assistant
            # If we add it to the prompt, the model will generate it again, resulting in <think><think>
            # So we just use the prompt as-is, and the model will generate <think>...</think> naturally

            # Save first prompt for debugging
            if i == 0:
                first_prompt_text = prompt_text

            # Re-encode the prompt with tag
            prompt_with_tag = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_inputs.append(torch.tensor(prompt_with_tag, device=device))

        # Pad prompts to same length for batch generation
        max_prompt_len = max(len(p) for p in prompt_inputs)
        attention_masks = []
        padded_prompts = []

        for p in prompt_inputs:
            pad_len = max_prompt_len - len(p)
            if pad_len > 0:
                padded = torch.cat([
                    torch.full((pad_len,), self.tokenizer.pad_token_id, device=device),
                    p
                ])
                mask = torch.cat([
                    torch.zeros(pad_len, device=device),
                    torch.ones(len(p), device=device)
                ])
            else:
                padded = p
                mask = torch.ones(len(p), device=device)

            padded_prompts.append(padded)
            attention_masks.append(mask)

        padded_prompts = torch.stack(padded_prompts)
        attention_masks = torch.stack(attention_masks)


        # Switch to eval mode for generation
        self.fsdp_model.eval()

        # FSDP2 fix: disable resharding and unshard parameters before generation
        # Problem: with reshard_after_forward=True, parameters get resharded after EVERY forward pass
        # But .generate() does multiple forward passes (one per token), so we need to keep params unsharded
        is_fsdp2 = self.config.model.strategy == "fsdp2"
        if is_fsdp2:
            print(">>> FSDP2: Disabling reshard_after_forward for generation <<<")
            # FSDP2 wraps submodules, so we need to iterate and set property on each
            for module in self.fsdp_model.modules():
                if hasattr(module, 'set_reshard_after_forward'):
                    module.set_reshard_after_forward(False)
            print(">>> FSDP2: Unsharding parameters <<<")
            for module in self.fsdp_model.modules():
                if hasattr(module, 'unshard'):
                    module.unshard()

        # Temporarily restore original flash attention for generation
        with self._without_ulysses_patches():
            with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                print(">>> Generating from self.fsdp_model <<<")
                generated = self.fsdp_model.generate(
                    padded_prompts,
                    attention_mask=attention_masks,
                    max_new_tokens=self.max_thinking_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=self.thinking_temperature,
                    top_p=self.thinking_top_p,
                    do_sample=self.thinking_do_sample,
                )


        # FSDP2 fix: restore resharding behavior and switch back to train mode
        if is_fsdp2:
            print(">>> FSDP2: Re-enabling reshard_after_forward <<<")
            for module in self.fsdp_model.modules():
                if hasattr(module, 'set_reshard_after_forward'):
                    module.set_reshard_after_forward(True)

        # Switch back to train mode for training
        self.fsdp_model.train()

        # Extract and clean thinking for each sample
        thinking_tokens_list = []
        for i in range(batch_size):
            # Get ONLY the generated part (after prompt) - this is key!
            gen_start_idx = len(prompt_inputs[i])
            generated_only = generated[i, gen_start_idx:]

            # Remove padding
            generated_only = generated_only[generated_only != self.tokenizer.pad_token_id]

            # Decode ONLY the generated portion to text
            generated_text = self.tokenizer.decode(generated_only, skip_special_tokens=False)

            # Find the assistant's response (after <|im_start|>assistant)
            if "<|im_start|>assistant" in generated_text:
                assistant_start = generated_text.find("<|im_start|>assistant") + len("<|im_start|>assistant")
                assistant_response = generated_text[assistant_start:]
            else:
                # No assistant marker found, use entire generated text
                assistant_response = generated_text

            # Find <think> and </think> in the assistant's response
            if "<think>" in assistant_response and "</think>" in assistant_response:
                think_start = assistant_response.find("<think>")
                think_end = assistant_response.find("</think>") + len("</think>")
                full_thinking_text = assistant_response[think_start:think_end]
            elif "<think>" in assistant_response:
                # <think> found but no </think>, take from <think> onwards and add </think>
                think_start = assistant_response.find("<think>")
                full_thinking_text = assistant_response[think_start:] + "</think>"
            else:
                # No <think> found, wrap entire response
                full_thinking_text = "<think>" + assistant_response + "</think>"

            # Re-encode to tokens
            thinking_tokens = self.tokenizer.encode(full_thinking_text, add_special_tokens=False)
            thinking_tokens_list.append(torch.tensor(thinking_tokens, device=device))

        return thinking_tokens_list, first_prompt_text

    def _build_single_sample(self, prompt_ids, thinking_ids, answer_ids):
        """
        Construct a single training sample: [prompt] + [thinking] + <answer>[answer]</answer>

        Uses text-based approach to avoid multi-token tag matching issues.

        Args:
            prompt_ids: Tensor of prompt token IDs
            thinking_ids: Tensor of thinking token IDs (includes <think>...</think>)
            answer_ids: Tensor of answer token IDs (without <answer> tags)

        Returns:
            Dict with input_ids, attention_mask, position_ids, loss_mask
        """
        device = prompt_ids.device

        # Decode all inputs to text
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        thinking_text = self.tokenizer.decode(thinking_ids, skip_special_tokens=False)
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=False)

        # Build the sequence in two parts:
        # 1. Part with NO loss: prompt + thinking (ends with </think>)
        # 2. Part with loss: <answer> + answer + </answer>
        no_loss_text = prompt_text + thinking_text
        loss_text = "<answer>" + answer_text + "</answer>"
        full_text = no_loss_text + loss_text

        # Encode the no_loss portion to find the boundary
        no_loss_tokens = self.tokenizer.encode(no_loss_text, add_special_tokens=False)
        boundary_idx = len(no_loss_tokens)

        # Encode the full sequence
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Add EOS token
        eos_id = self.tokenizer.eos_token_id
        input_ids = torch.tensor(full_tokens + [eos_id], device=device)

        # Build attention mask (all 1s for actual content)
        attention_mask = torch.ones_like(input_ids)

        # Build loss mask based on boundary
        # Everything before boundary_idx: no loss (prompt + thinking)
        # Everything from boundary_idx onwards: apply loss (answer + eos)
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[boundary_idx:] = 1

        # Build position IDs
        position_ids = torch.arange(len(input_ids), device=device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask
        }

    def _reconstruct_batch_with_thinking(self, original_batch, thinking_tokens_list):
        """
        Rebuild batch by inserting generated thinking between prompt and answer.

        Process:
        1. Extract prompt and answer from original batch
        2. Decode answer to text, strip tags with string operations
        3. Rebuild all tensors (input_ids, attention_mask, position_ids, loss_mask)
        4. Pad to max length in batch

        Args:
            original_batch: Original batch from dataloader
            thinking_tokens_list: List of thinking tensors (one per sample)

        Returns:
            New TensorDict with reconstructed batch
        """
        batch_size = original_batch["input_ids"].shape[0]
        prompt_lengths = self._get_prompt_lengths(original_batch)
        device = original_batch["input_ids"].device

        # Build each sample with thinking
        new_samples = []
        for i in range(batch_size):
            prompt_len = prompt_lengths[i]

            # Get prompt (everything before response)
            prompt_ids = original_batch["input_ids"][i, :prompt_len]

            # Get answer (everything after prompt, excluding padding)
            answer_start = prompt_len
            answer_ids = original_batch["input_ids"][i, answer_start:]
            answer_mask = original_batch["attention_mask"][i, answer_start:]
            answer_len = answer_mask.sum().item()
            answer_ids = answer_ids[:answer_len]

            # Decode answer to text for easy tag removal
            answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=False)

            # Remove ALL tags using string operations
            # The original data has <think>...</think><answer>...</answer>
            # But we're generating thinking online, so we need to extract ONLY the answer content

            # Find <answer> and </answer> tags to extract just the answer content
            if "<answer>" in answer_text and "</answer>" in answer_text:
                answer_start = answer_text.find("<answer>") + len("<answer>")
                answer_end = answer_text.find("</answer>")
                answer_text = answer_text[answer_start:answer_end]
            else:
                # No answer tags found, try to strip thinking tags
                # Remove <think>...</think> if present
                if "<think>" in answer_text and "</think>" in answer_text:
                    think_start = answer_text.find("<think>")
                    think_end = answer_text.find("</think>") + len("</think>")
                    answer_text = answer_text[:think_start] + answer_text[think_end:]

                # Clean up any remaining tags
                answer_text = answer_text.replace("<answer>", "").replace("</answer>", "")

            # The tokenizer.decode might include EOS as a special string, but we set skip_special_tokens=False
            # So we don't need to worry about it here - it will be stripped

            # Re-encode cleaned answer text back to tokens
            answer_ids_clean = self.tokenizer.encode(answer_text, add_special_tokens=False)
            answer_ids = torch.tensor(answer_ids_clean, device=device)

            # Build sample with thinking
            sample = self._build_single_sample(
                prompt_ids=prompt_ids,
                thinking_ids=thinking_tokens_list[i],
                answer_ids=answer_ids
            )
            new_samples.append(sample)

        # Find max length in batch (for padding)
        max_len_in_batch = max(len(s["input_ids"]) for s in new_samples)

        # Pad each sample to max length in batch
        for sample in new_samples:
            seq_len = len(sample["input_ids"])
            if seq_len < max_len_in_batch:
                pad_len = max_len_in_batch - seq_len
                sample["input_ids"] = torch.cat([
                    sample["input_ids"],
                    torch.full((pad_len,), self.tokenizer.pad_token_id, device=device)
                ])
                sample["attention_mask"] = torch.cat([
                    sample["attention_mask"],
                    torch.zeros(pad_len, dtype=sample["attention_mask"].dtype, device=device)
                ])
                sample["position_ids"] = torch.cat([
                    sample["position_ids"],
                    torch.zeros(pad_len, dtype=sample["position_ids"].dtype, device=device)
                ])
                sample["loss_mask"] = torch.cat([
                    sample["loss_mask"],
                    torch.zeros(pad_len, dtype=sample["loss_mask"].dtype, device=device)
                ])

        # Stack into batch tensors
        return TensorDict({
            "input_ids": torch.stack([s["input_ids"] for s in new_samples]),
            "attention_mask": torch.stack([s["attention_mask"] for s in new_samples]),
            "position_ids": torch.stack([s["position_ids"] for s in new_samples]),
            "loss_mask": torch.stack([s["loss_mask"] for s in new_samples])
        }, batch_size=batch_size)

    def training_step(self, batch: TensorDict):
        """
        Override training_step to add thinking generation phase.

        Flow:
        1. Generate thinking (no grad, eval mode)
        2. Reconstruct batch with thinking
        3. Train on reconstructed batch (forward + backward)

        Returns:
            Dict with training metrics
        """
        start_time = time.time()

        # Step 1: Generate thinking tokens (no gradients)
        thinking_tokens, first_prompt_text = self._generate_thinking_for_batch(batch)

        # Step 2: Reconstruct batch with generated thinking
        new_batch = self._reconstruct_batch_with_thinking(batch, thinking_tokens)

        # Debug: Print example for first few steps (only rank 0)
        if self.device_mesh.get_rank() == 0 and self.debug_step_counter < self.debug_print_steps:
            self._print_debug_example(batch, thinking_tokens, new_batch, first_prompt_text)
            self.debug_step_counter += 1

        # Step 3: Training phase (same as parent, but on new_batch)
        self.fsdp_model.train()
        self.optimizer.zero_grad()

        # Split into micro-batches
        micro_batches = new_batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)

        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            step_loss += loss.item()

        # Gradient clipping
        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(
                self.fsdp_model.parameters(),
                max_norm=self.config.optim.clip_grad
            )
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        # Optimizer step (skip if grad is not finite)
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]

        # Reduce loss across all ranks
        step_loss = torch.tensor(step_loss).to(self.device_name)

        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)

        end_time = time.time()
        spend_time_per_step = end_time - start_time

        return {
            "train/loss": step_loss.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/time(s)": spend_time_per_step,
        }


def run_sft_thinking(config):
    """
    Run SFT training with online thinking generation.

    Same as run_sft() from base trainer, but uses FSDPSFTThinkingTrainer.
    """
    from verl.utils.device import get_device_name

    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )

    # Build tokenizer and datasets
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    logger.info(f"Train dataset length: {len(train_dataset)}")

    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)
    logger.info(f"Val dataset length: {len(val_dataset)}")

    # Use thinking trainer instead of base trainer
    trainer = FSDPSFTThinkingTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path="configs/sft", config_name="sft_thinking_trainer", version_base=None)
def main(config):
    """Main entry point for thinking trainer."""
    OmegaConf.resolve(config)
    from pprint import pprint
    pprint(OmegaConf.to_container(config, resolve=True))

    run_sft_thinking(config)


if __name__ == "__main__":
    main()
