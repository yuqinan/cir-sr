"""
Prompt Manager

Handles creation of prompts for teacher models with different configurations.
Separated from teacher_pipeline.py for better organization.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

FORMATING_PROMPT = {
    # ===== your originals (kept verbatim) =====
    "futoshiki": "Ensure your answer follows the same format as the puzzle above, just replace blanks (_) with the correct value for the cell.\n",
    "mini_sudoku": "Format your response as the puzzle above, with spaces separating each number within a row, and newlines separating rows.\n",
    "spiral_matrix": "Your output should be a space-separated list of integers, e.g. 1 2 3 4 5 6",
    "family_relationships": "Answer with a single word.\n",
    "simple_equations": "Answer in one number.\n",
    "rotate_matrix": "Your output should be a matrix in the same format as the input.\n",
    "arc_1d": "Your final answer should be just the test output grid itself.\n",
    "rush_hour": "Specify moves in the format: 'F+1 K+1 M-1 C+3 H+2 ...'\nwhere the letter is the vehicle and +/- number is spaces to move right/left or down/up.",
    "simple_geometry": "Return only the angle as your answer.Do not give the units in your answer.\n",
    "puzzle24": "Final answer format instructions:\n1. Provide your final answer as a arithmetic expression (no '=' sign).\n2. Do not include the target number in the expression.\n3. Use '*' for multiplication.\n4. Use '/' for division.\n",
    "knight_swap": "Answer Format:\n- For impossible puzzles: \"No\"\n- For possible puzzles: List moves as [\"color,from,to\", ...]\n  Example: [\"w,A1,B3\"] means white knight moves A1→B3\n",
    "chain_sum": "",
    "complex_arithmetic": "",
    "basic_arithmetic": "",

    # ===== filled-in entries for all tasks in your TASKS list =====
    "advanced_geometry": "Return only the requested numeric value (angle/length/etc.). Do not include units or explanations.\n",
    "base_conversion": "Return only the converted number string (no prefixes like 0x/0b and no spaces).\n",
    "binary_alternation": "Return only the resulting binary string.\n",
    "binary_matrix": "Return the resulting matrix formatted exactly like the input (rows on newlines, space-separated values).\n",
    "bitwise_arithmetic": "Evaluate the expression and return a single integer in decimal.\n",
    "caesar_cipher": "Return only the transformed text (no quotes or extra text).\n",
    "count_bits": "Answer with a single integer (the number of 1-bits).\n",
    "countdown": "Provide your final answer as an arithmetic expression that equals the target (no '=' sign). Use each given number at most once; allowed ops: + - * /.\n",
    "cryptarithm": "Provide a comma-separated mapping from letters to digits. Format exactly: A=1,B=2,C=3 (no spaces).\n",
    "decimal_arithmetic": "Solve to at most 12 significant digits (round half up). Reply with the final value only.\n",
    "fraction_simplification": "Give only the simplified fraction as your final answer (e.g. 15/113).\n",
    "largest_island": "Return a single integer: the maximum island area (0 if none).\n",
    "leg_counting": "Return only the total number of legs as an integer.\n",
    "letter_counting": "Return a single integer: the count.\n",
    "letter_jumble": "Return the unscrambled sentence only, preserving capitalization and punctuation.\n",
    "manipulate_matrix": "Return the resulting matrix in the same format as the input.\n",
    "number_format": "Return only the correctly formatted number string; no explanation.\n",
    "number_sorting": "Your output should be a comma-separated list of numbers (no brackets), e.g. 1, 2, 3.\n",
    "palindrome_generation": "Output a single palindrome string using all given letters; no spaces or punctuation.\n",
    "palindrome_partitioning": "Return a valid partition of the string into palindromic substrings as a comma-separated list, in order.\n",
    "polynomial_multiplication": "Return the simplified expanded expression only. Use ** for exponents and include * in all multiplications.\n",
    "pool_matrix": "Your output should be a matrix in the same format as the input. If averages are needed, give values to 2 decimal places.\n",
    "quantum_lock": "Return only the final code as a contiguous string (e.g. 0427), no spaces.\n",
    "simple_integration": "Return only the antiderivative expression (include + C). Use ** for exponents and include * in multiplications.\n",
    "string_manipulation": "Return only the resulting string; do not include explanation or quotes.\n",
    "string_splitting": "Output six space-separated integers in the order: A B C X Y Z.\n",
    "string_synthesis": "Return only the synthesized string; no extra text.\n",
    "tsumego": "Return only the move coordinates using the same format as the puzzle (e.g. D4); no extra text.\n",
    "word_sequence_reversal": "Provide your answer as a comma-separated list of words with a space after each comma.\n",
    "word_sorting": "Your output should be a comma-separated list of words (no brackets), e.g. apple, banana, pear.\n",
}

class PromptManager:
    """Manages prompt creation for different model types and configurations."""
    
    def __init__(self, teacher_config: Dict[str, Any], developer_prompt: str):
        """
        Initialize prompt manager.
        
        Args:
            teacher_config: Teacher model configuration
            developer_prompt: System/developer prompt to use
        """
        self.teacher_config = teacher_config
        self.developer_prompt = developer_prompt
        self.developer_role = teacher_config.get('developer_role', 'system')
        self.preappend_token = teacher_config.get('preappend_token', None)
        self.use_chat_template = teacher_config.get('use_chat_template', True)

    def _load_cot_verifier_config(self, task: str) -> Dict[str, Any]:
        """
        Load task-specific configuration for cot_verifier_accuracy prompts.

        Args:
            task: Task name (e.g., 'mini_sudoku', 'spiral_matrix')

        Returns:
            Dictionary with optional keys:
                - custom_instructions: str
                - examples: List[Dict]
                - evaluation_criteria: str
                - format_override: str
            Returns empty dict if no config file exists or on error.
        """
        # Get the path to the config file
        current_file = Path(__file__)
        prompts_dir = current_file.parent.parent / 'prompts' / 'cot_verifier'
        config_path = prompts_dir / f'{task}.yaml'

        # Return empty dict if config doesn't exist
        if not config_path.exists():
            logger.debug(f"No task-specific cot_verifier config found for {task}")
            return {}

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                logger.warning(f"Invalid cot_verifier config for {task}: not a dictionary")
                return {}

            logger.info(f"Loaded task-specific cot_verifier config for {task}")
            return config

        except Exception as e:
            logger.error(f"Error loading cot_verifier config for {task}: {e}")
            return {}

    def create_teacher_prompt(self, question: str, tokenizer=None, is_perturbation: bool = False) -> str:
        """
        Create a prompt for teacher model inference.
        
        Args:
            question: The question/input to process
            tokenizer: Model tokenizer (required for chat template)
            is_perturbation: Whether this is for CoT perturbation (disables chat template)
            
        Returns:
            Formatted prompt ready for model
        """
        # Disable chat template for perturbations (already formatted)
        use_template = self.use_chat_template and not is_perturbation
        
        if use_template and tokenizer:
            return self._create_chat_template_prompt(question, tokenizer)
        else:
            return self._create_simple_prompt(question, is_perturbation)
    
    def create_openai_prompt(self, question: str) -> str:
        """
        Create a prompt for OpenAI API.
        
        Args:
            question: The question to process
            
        Returns:
            Formatted prompt for OpenAI
        """
        if self.developer_prompt:
            # Return the question - OpenAI client handles message formatting
            return self.developer_prompt + "\n\n"+question + self.preappend_token
        else:
            return question
    
    def _create_chat_template_prompt(self, question: str, tokenizer) -> str:
        """Create prompt using chat template."""
        try:
            chat = []
            
            if self.developer_prompt:
                chat.append({"role": self.developer_role, "content": self.developer_prompt})
            
            chat.append({"role": "user", "content": question})
            
            prompt = tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            if self.preappend_token:
                prompt = prompt + self.preappend_token
            
            logger.debug("Applied chat template successfully")
            return prompt
            
        except Exception as e:
            logger.warning(f"Chat template failed: {e}, falling back to simple format")
            return self._create_simple_prompt(question, is_perturbation=False)
    
    def _create_simple_prompt(self, question: str, is_perturbation: bool) -> str:
        """Create simple prompt without chat template."""
        if is_perturbation:
            # For perturbation, question is already a complete formatted prompt
            logger.debug("Using pre-formatted perturbation prompt")
            return question
        else:
            # Simple format with developer prompt
            if self.developer_prompt:
                return f"{self.developer_prompt}\\n\\n{question}"
            else:
                return question

    #################################VERIFIER ACCURACY PROMPTS########################################################
    # Verifier accuracy: answer | question, thinking (WITH question provided)
    def create_verifier_prompt_with_question(self, question: str, thinking: str, task: str) -> str:
        """Create prompt for cot_verifier_accuracy with question provided.

        Tests: answer | question, thinking
        Provides question + reasoning traces. Model must follow the traces
        and return the answer. Used by cot_verifier_accuracy metric.

        Supports modular task-specific customization through YAML config files.
        """
        # Load task-specific config
        config = self._load_cot_verifier_config(task)

        # Build prompt modularly
        prompt_parts = []

        # 1. Base template (always included) - NOW INCLUDES QUESTION
        prompt_parts.append("""You are an expert verifier. You are given:
- A Question
- Reasoning Traces from another model

Your job:
1. Following exactly the reasoning traces for the question and return the answer the other model generated.
2. Do not make any assumptions or use any external knowledge, solely follow the reasoning traces step by step without any deviation.
3. If you can't answer, reply no answer found
4. Place your answer between <answer> and </answer>.""")

        # 2. Custom instructions (if provided)
        if config.get('custom_instructions'):
            prompt_parts.append("\nAdditional Instructions:")
            prompt_parts.append(config['custom_instructions'].strip())

        # 3. Evaluation criteria (if provided)
        if config.get('evaluation_criteria'):
            prompt_parts.append("\nEvaluation Criteria:")
            prompt_parts.append(config['evaluation_criteria'].strip())

        # 4. Examples (if provided)
        if config.get('examples'):
            prompt_parts.append("\nExamples:")
            for i, example in enumerate(config['examples'], 1):
                prompt_parts.append(f"\nExample {i}:")
                if 'reasoning' in example:
                    prompt_parts.append(f"Reasoning traces: {example['reasoning']}")
                if 'expected_answer' in example:
                    prompt_parts.append(f"Expected answer: {example['expected_answer']}")
                if 'notes' in example:
                    prompt_parts.append(f"Notes: {example['notes']}")

        # 5. Format instructions (custom or default)
        format_instructions = config.get('format_override', FORMATING_PROMPT.get(task, ''))
        if format_instructions:
            prompt_parts.append(f"\n{format_instructions.strip()}")

        # 6. Add the question, reasoning traces, and output marker
        prompt_parts.append(f"\nQuestion: {question}")
        prompt_parts.append(f"Reasoning traces: {thinking}")
        prompt_parts.append("\nOutput:")

        return '\n'.join(prompt_parts)


    # Verifier accuracy: answer | thinking (WITHOUT question)
    def create_verifier_prompt_without_question(self, thinking: str, task: str) -> str:
        """Create prompt for cot_verifier_accuracy without question.

        Tests: answer | thinking
        Only provides reasoning traces, no question.
        The model must follow the traces and extract the answer.
        Used for the "answer | thinking" variant of verifier accuracy.

        Supports modular task-specific customization through YAML config files.
        """
        # Load task-specific config
        config = self._load_cot_verifier_config(task)

        # Build prompt modularly
        prompt_parts = []

        # 1. Base template (always included) - WITHOUT QUESTION
        prompt_parts.append("""You are an expert verifier. You are given:
- Reasoning Traces from another model

Your job:
1. Following exactly the reasoning traces and return the answer the other model generated.
2. Do not make any assumptions or use any external knowledge, solely follow the reasoning traces step by step without any deviation.
3. If you can't answer, reply no answer found
4. Place your answer between <answer> and </answer>.""")

        # 2. Custom instructions (if provided)
        if config.get('custom_instructions'):
            prompt_parts.append("\nAdditional Instructions:")
            prompt_parts.append(config['custom_instructions'].strip())

        # 3. Evaluation criteria (if provided)
        if config.get('evaluation_criteria'):
            prompt_parts.append("\nEvaluation Criteria:")
            prompt_parts.append(config['evaluation_criteria'].strip())

        # 4. Examples (if provided)
        if config.get('examples'):
            prompt_parts.append("\nExamples:")
            for i, example in enumerate(config['examples'], 1):
                prompt_parts.append(f"\nExample {i}:")
                if 'reasoning' in example:
                    prompt_parts.append(f"Reasoning traces: {example['reasoning']}")
                if 'expected_answer' in example:
                    prompt_parts.append(f"Expected answer: {example['expected_answer']}")
                if 'notes' in example:
                    prompt_parts.append(f"Notes: {example['notes']}")

        # 5. Format instructions (custom or default)
        format_instructions = config.get('format_override', FORMATING_PROMPT.get(task, ''))
        if format_instructions:
            prompt_parts.append(f"\n{format_instructions.strip()}")

        # 6. Add the reasoning traces and output marker (NO question)
        prompt_parts.append(f"\nReasoning traces: {thinking}")
        prompt_parts.append("\nOutput:")

        return '\n'.join(prompt_parts)

    def create_answer_and_question_removal_prompt(self, thinking: str, answer: str, question: str) -> str:
        """Create prompt for removing both answer and question from thinking trace.

        Args:
            thinking: The reasoning trace
            answer: The final answer to remove
            question: The original question to remove

        Returns:
            A prompt for GPT-4o-mini to clean the thinking trace
        """
        prompt = f"""You are given a reasoning trace and the original question. Your task is to remove any part of the reasoning trace that include or restate the question such as mentioning the exact number of the question or the instruction.

Additionally, remove the final answer from the reasoning trace.

Original Question:
{question if question else 'N/A'}

Answer to Remove:
{answer if answer else 'N/A'}

Reasoning Trace:
{thinking}

Please return ONLY the cleaned reasoning trace. Keep all the actual reasoning steps, intermediate calculations, and logic. Do not rewrite any part of the reasoning trace.

Cleaned Reasoning Trace:"""

        return prompt
