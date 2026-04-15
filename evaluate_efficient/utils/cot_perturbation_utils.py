"""
Simplified Chain-of-Thought Perturbation

This module provides simple perturbation functions that work as post-processing
on already generated teacher responses.
"""

import re
import logging
import random

logger = logging.getLogger(__name__)

def truncate_thinking_first(thinking: str, ratio: float = 0.5) -> str:
    """
    Truncate the first half of thinking trace
    
    Args:
        thinking: Original thinking trace
        ratio: Fraction of tokens to remove from the beginning (default 0.5)
    
    Returns:
        Thinking trace with first half removed
    """
    if not thinking or not thinking.strip():
        return thinking
        
    tokens = thinking.split()
    if len(tokens) <= 1:
        return thinking
    
    # Remove tokens from the beginning
    tokens_to_remove = int(len(tokens) * ratio)
    remaining_tokens = tokens[tokens_to_remove:]
    
    return ' '.join(remaining_tokens) if remaining_tokens else thinking

def truncate_thinking_second(thinking: str, ratio: float = 0.5) -> str:
    """
    Truncate the second half of thinking trace
    
    Args:
        thinking: Original thinking trace
        ratio: Fraction of tokens to remove from the end (default 0.5)
    
    Returns:
        Thinking trace with second half removed
    """
    if not thinking or not thinking.strip():
        return thinking
        
    tokens = thinking.split()
    if len(tokens) <= 1:
        return thinking
    
    # Keep tokens from the beginning
    tokens_to_keep = int(len(tokens) * (1 - ratio))
    tokens_to_keep = max(1, tokens_to_keep)  # Keep at least 1 token
    truncated_tokens = tokens[:tokens_to_keep]
    
    return ' '.join(truncated_tokens)

def truncate_thinking_random(thinking: str, keep_prob: float = 0.5) -> str:
    """
    Randomly truncate tokens from thinking trace
    
    Args:
        thinking: Original thinking trace
        keep_prob: Probability of keeping each token (default 0.5)
    
    Returns:
        Thinking trace with randomly selected tokens
    """
    if not thinking or not thinking.strip():
        return thinking
        
    tokens = thinking.split()
    if len(tokens) <= 1:
        return thinking
    
    # Randomly decide whether to keep each token
    kept_tokens = []
    for token in tokens:
        if random.random() < keep_prob:
            kept_tokens.append(token)
    
    # Ensure at least one token is kept
    if not kept_tokens and tokens:
        kept_tokens = [tokens[0]]
    
    return ' '.join(kept_tokens)

def truncate_thinking(thinking: str, ratio: float = 0.5) -> str:
    """
    Legacy truncate function - truncates from the end (same as truncate_second)

    Args:
        thinking: Original thinking trace
        ratio: Fraction of tokens to keep (default 0.5 for half)

    Returns:
        Truncated thinking trace
    """
    return truncate_thinking_second(thinking, 1 - ratio)

def remove_thinking(thinking: str) -> str:
    """
    Remove all thinking content, returning an empty string

    This creates prompts with <think></think><answer> to test whether
    the model can produce correct answers without any reasoning trace.

    Args:
        thinking: Original thinking trace (ignored)

    Returns:
        Empty string
    """
    return ""

def add_filler_tokens(thinking: str, filler_token: str = "...") -> str:
    """
    Replace all words in thinking trace with filler tokens
    
    Args:
        thinking: Original thinking trace
        filler_token: Token to replace words with (default "...")
    
    Returns:
        Thinking trace with all words replaced by filler tokens
    """
    if not thinking or not thinking.strip():
        return thinking
        
    # Split by whitespace to get tokens
    tokens = thinking.split()
    
    # Replace all tokens with filler tokens
    filler_tokens = [filler_token] * len(tokens)
    
    return ' '.join(filler_tokens)

def shuffle_thinking(thinking: str) -> str:
    """
    Shuffle all words in thinking trace randomly
    
    Args:
        thinking: Original thinking trace
    
    Returns:
        Thinking trace with all words shuffled randomly
    """
    if not thinking or not thinking.strip():
        return thinking
        
    # Split by whitespace to get tokens
    tokens = thinking.split()
    
    if len(tokens) <= 1:
        return thinking
    
    # Shuffle the tokens randomly
    shuffled_tokens = tokens.copy()
    random.shuffle(shuffled_tokens)
    
    return ' '.join(shuffled_tokens)

def add_expert_thinking(expert_thinking: str) -> str:
    """
    Add expert thinking to the end of the thinking trace
    """
    expert_thinking = expert_thinking.replace("<think>", "")
    return f" {expert_thinking}"

def add_replacement_thinking(replacement_thinking: str) -> str:
    """
    Add replacement thinking trace
    """
    replacement_thinking = replacement_thinking.replace("<think>", "")
    return f" {replacement_thinking}"


def create_perturbed_prompt(question_with_think_tag: str, perturbed_thinking: str) -> str:
    """
    Create a new prompt with perturbed thinking trace
    
    Args:
        question_with_think_tag: Original question ending with <think> tag
        perturbed_thinking: Perturbed thinking trace to insert
    
    Returns:
        Complete prompt with perturbed thinking ready for inference
    """
    # The question already ends with "<think>", so we append the perturbed thinking
    # and close the think tag, then ask for an answer
    prompt = f"{question_with_think_tag}{perturbed_thinking}</think>\n<answer>"
    return prompt

def perturb_teacher_responses(teacher_responses, perturbation_types=["truncate", "filler"]):
    """
    Add perturbation variants to teacher responses data

    Args:
        teacher_responses: List of teacher response dictionaries (now with k_responses structure)
        perturbation_types: List of perturbation types to apply
                          Supported types: "truncate", "truncate_first", "truncate_second",
                          "truncate_random", "filler", "shuffle", "remove_thinking",
                          "expert_thinking", "replace"

    Returns:
        Dictionary with original and perturbed versions

    Note:
        - "expert_thinking": Prepends expert model thinking traces (from expert_thinking_traces field)
        - "replace": Prepends replacement thinking traces and updates answer for scoring (from replacement_thinking_traces field)
          For "replace", the answer field is updated to the replacement answer for correct reward calculation!
    """
    result = {
        "original": teacher_responses,
        "perturbed": {}
    }

    for pert_type in perturbation_types:
        perturbed_responses = []

        for response in teacher_responses:
            # Handle both old format (teacher_thinking at top level) and new format (k_responses)
            if 'k_responses' in response:
                # New format: iterate through all k responses
                k_responses_list = response['k_responses']
            else:
                # Old format: wrap in a list for uniform processing
                k_responses_list = [response]

            # Process each k response
            for k_idx, k_response in enumerate(k_responses_list):
                if 'teacher_thinking' not in k_response:
                    continue

                original_thinking = k_response['teacher_thinking']

                # Apply perturbation based on type
                if pert_type == "truncate":
                    perturbed_thinking = truncate_thinking(original_thinking)
                elif pert_type == "truncate_first":
                    perturbed_thinking = truncate_thinking_first(original_thinking)
                elif pert_type == "truncate_second":
                    perturbed_thinking = truncate_thinking_second(original_thinking)
                elif pert_type == "truncate_random":
                    perturbed_thinking = truncate_thinking_random(original_thinking)
                elif pert_type == "filler":
                    perturbed_thinking = add_filler_tokens(original_thinking)
                elif pert_type == "shuffle":
                    perturbed_thinking = shuffle_thinking(original_thinking)
                elif pert_type == "remove_thinking":
                    perturbed_thinking = remove_thinking(original_thinking)
                elif pert_type == "expert_thinking":
                    # Handle multiple expert thinking traces - create separate perturbations for each expert model
                    expert_traces = response.get('expert_thinking_traces', {})
                    if expert_traces:
                        # Create a perturbation for each expert model
                        for expert_model_name, expert_trace in expert_traces.items():
                            perturbed_thinking = add_expert_thinking(expert_trace)

                            # Create perturbed prompt
                            perturbed_prompt = create_perturbed_prompt(response['full_prompt'], perturbed_thinking)

                            # Create new response entry for this expert model
                            expert_perturbed_response = response.copy()
                            expert_perturbed_response.update({
                                'question': perturbed_prompt,
                                'teacher_thinking': perturbed_thinking,
                                'perturbation_type': f'expert_thinking_{expert_model_name}',
                                'expert_model_name': expert_model_name,
                                'original_thinking': original_thinking,
                                'answer': response['gold_answer'],
                                'teacher_answer': k_response.get('teacher_answer', ''),
                                'k_idx': k_idx  # Track which k response this came from
                            })

                            perturbed_responses.append(expert_perturbed_response)
                        continue  # Skip the normal processing for expert_thinking
                    else:
                        # Fallback to single expert_thinking field if available
                        perturbed_thinking = add_expert_thinking(response.get('expert_thinking', ''))
                elif pert_type == "replace":
                    # Handle replacement thinking traces - similar to expert_thinking
                    replacement_traces = response.get('replacement_thinking_traces', {})
                    if replacement_traces:
                        # Create a perturbation for each replacement instance
                        for replacement_id, replacement_data in replacement_traces.items():
                            perturbed_thinking = add_replacement_thinking(replacement_data.get('thinking', ''))

                            # Create perturbed prompt
                            perturbed_prompt = create_perturbed_prompt(response['full_prompt'], perturbed_thinking)

                            # Create new response entry for this replacement
                            replacement_perturbed_response = response.copy()
                            replacement_perturbed_response.update({
                                'question': perturbed_prompt,
                                'teacher_thinking': perturbed_thinking,
                                'perturbation_type': f'replace_{replacement_id}',
                                'replacement_id': replacement_id,
                                'original_thinking': original_thinking,
                                'answer': replacement_data.get('gold_answer', replacement_data.get('answer', response['gold_answer'])),  # Use replacement answer for scoring!
                                'original_answer': response['gold_answer'],  # Keep track of original answer
                                'replacement_answer': replacement_data.get('answer', ''),  # The replacement's answer
                                'teacher_answer': k_response.get('teacher_answer', ''),
                                'k_idx': k_idx  # Track which k response this came from
                            })

                            perturbed_responses.append(replacement_perturbed_response)
                        continue  # Skip the normal processing for replace
                    else:
                        logger.warning(f"No replacement_thinking_traces found for replace perturbation")
                        continue
                else:
                    logger.warning(f"Unknown perturbation type: {pert_type}")
                    continue

                # Create perturbed prompt
                perturbed_prompt = create_perturbed_prompt(response['full_prompt'], perturbed_thinking)

                # Create new response entry for inference
                perturbed_response = response.copy()
                perturbed_response.update({
                    'question': perturbed_prompt,  # This becomes the new input for inference
                    'teacher_thinking': perturbed_thinking,
                    'k_idx': k_idx,  # Track which k response this came from
                    'perturbation_type': pert_type,
                    'original_thinking': original_thinking,
                    'answer': response['gold_answer'],
                    'teacher_answer': k_response.get('teacher_answer', '')
                })

                perturbed_responses.append(perturbed_response)
        
        result["perturbed"][pert_type] = perturbed_responses
    
    return result