"""
Prompt Manager

Handles creation of prompts for teacher models with different configurations.
Separated from teacher_pipeline.py for better organization.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

FORMATING_PROMPT = {
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
    "knight_swap": "Answer Format:\n- For impossible puzzles: \"No\"\n- For possible puzzles: List moves as [\"color,from,to\", ...]\n  Example: [\"w,A1,B3\"] means white knight moves A1\u2192B3\n",
    "chain_sum": "",
    "complex_arithmetic": "",
    "basic_arithmetic": "",
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

################################THE FOLLOWING TEST GENERALIZABILITY IN THE NEW QUESTION########################################################

    ## this one gives the unedited explanation, the original question and test generalizability in the new question
    def create_whole_explanation_question_generalize_to_new_question_prompt(self, question: str, thinking: str, new_question: str, task: str) -> str:
        """Create prompt for whole_explanation_question_generalize_to_new_question informativeness evaluation using meta prompt template."""
        return f"""You are given:
- A sample Question
- Its Reasoning Traces
- A New Question from the same task

Your job:
1. Read the sample Question and Reasoning Trace.
2. Following the reasoning process, apply it to the New Question.
3. Do not make any assumptions or use any external knowledge, solely follow the reasoning traces step by step without any deviation
4. If you can't answer, reply no answer found
5. Output only the answer to the New Question. You may optionally include concise reasoning steps. Place your answer between <answer> and </answer>.

{FORMATING_PROMPT[task]}

Question: {question}
Reasoning traces: {thinking}
New question: {new_question}

Output:
"""
    ## this one gives the answer, the original question and test generalizability in the new question
    def create_answer_generalize_to_new_question_prompt(self, question: str, thinking: str, new_question: str, task: str) -> str:
        """Create prompt for answer_generalize_to_new_question informativeness evaluation."""
        # Placeholder - will be implemented later
        return f"""You are given:
- A sample Question
- The answer to the sample Question
- A New Question from the same task

Your job:
1. Read the sample Question and Answer.
2. Internally deduce the general task instruction they imply.
3. Apply this inferred instruction to the New Question.
4. If you can't answer, reply no answer found
5. Output only the answer to the New Question. You may optionally include concise reasoning steps. Place your answer between <answer> and </answer>.

{FORMATING_PROMPT[task]}

Question: {question}
Answer: {thinking}
New question: {new_question}

Output:
"""
  # the same as the first one but remove the answer from the reasoning traces  
    def create_answer_removed_explanation_question_generalize_to_new_question_prompt(self, question: str, thinking: str, new_question: str, task: str) -> str:
        """Create prompt for answer_removed_explanation_question_generalize_to_new_question informativeness evaluation.
        
        Provides original question, reasoning traces, and new question.
        Similar to whole_explanation but with answer removed from reasoning traces.
        """
        return f"""You are given:
- A sample Question
- Its Reasoning Traces 
- A New Question from the same task

Your job:
1. Read the sample Question and Reasoning Trace.
2. Following the reasoning process, apply it to the New Question.
3. Do not make any assumptions or use any external knowledge, solely follow the reasoning traces step by step without any deviation
4. If you can't answer, reply no answer found
5. Output only the answer to the New Question. You may optionally include concise reasoning steps. Place your answer between <answer> and </answer>.

{FORMATING_PROMPT[task]}

Question: {question}
Reasoning traces: {thinking}
New question: {new_question}

Output:
"""

    def create_answer_removed_explanation_generalize_to_new_question_prompt(self, thinking: str, new_question: str, task: str) -> str:
        """Create prompt for answer_removed_explanation_generalize_to_new_question informativeness evaluation.
        
        Only provides reasoning traces and new question, no original question.
        The model must use the reasoning to solve the new question.
        """
        return f"""You are given:
- Reasoning Traces
- A New Question from the same task

Your job:
1. Read the Reasoning Traces carefully.
2. Following the reasoning process, apply it to the New Question.
3. Do not make any assumptions or use any external knowledge, solely follow the reasoning traces step by step without any deviation
4. If you can't answer, reply no answer found
5. Output only the answer to the New Question. You may optionally include concise reasoning steps. Place your answer between <answer> and </answer>.

{FORMATING_PROMPT[task]}

Reasoning traces: {thinking}
New question: {new_question}

Output:
"""

    #################################VERIFIER ACCURACY PROMPTS########################################################
    # Verifier accuracy: answer | question, thinking (WITH question provided)
    def create_verifier_prompt_with_question(self, question: str, thinking: str, task: str) -> str:
        """Create prompt for cot_verifier_accuracy with question provided.

        Tests: answer | question, thinking
        Provides question + reasoning traces. Model must follow the traces
        and return the answer.
        """
        return f"""You are an expert verifier. You are given:
- A Question
- Reasoning Traces from another model

Your job:
1. Following exactly the reasoning traces and return the answer the other model generated.
2. Do not make any assumptions or use any external knowledge, solely follow the reasoning traces step by step without any deviation.
3. If you can't answer, reply no answer found
4. Place your answer between <answer> and </answer>.

{FORMATING_PROMPT[task]}

Question: {question}
Reasoning traces: {thinking}

Output:
"""

    # Verifier accuracy: answer | thinking (WITHOUT question)
    def create_verifier_prompt_without_question(self, thinking: str, task: str) -> str:
        """Create prompt for cot_verifier_accuracy without question.

        Tests: answer | thinking
        Only provides reasoning traces, no question.
        The model must follow the traces and extract the answer.
        """
        return f"""You are an expert verifier. You are given:
- Reasoning Traces from another model

Your job:
1. Following exactly the reasoning traces and return the answer the other model generated.
2. Do not make any assumptions or use any external knowledge, solely follow the reasoning traces step by step without any deviation.
3. If you can't answer, reply no answer found
4. Place your answer between <answer> and </answer>.

{FORMATING_PROMPT.get(task, '')}

Reasoning traces: {thinking}

Output:
"""

    # this one test whether the reasoning traces help with just the original question without generalizing to the new question
#    def create_answer_removed_explanation_only_prompt(self, thinking: str, task: str) -> str:
#        """Create prompt for answer_removed_explanation_only informativeness evaluation.
#        
#        Only provides the reasoning traces no question.
#        The model must infer what was being asked and provide the answer.
#        """
#        return f"""You are given:
#- Reasoning Traces 
#
#Your job:
#1. Following exactly the reasoning traces and deduct the answer.
#2. Do not make any assumptions or use any external knowledge, solely follow the reasoning traces step by step without any deviation
#3. If you can't answer, reply no answer found
#4. Output only the answer to the Question. You may optionally include concise reasoning steps. Place your answer between <answer> and </answer>.
#
#{FORMATING_PROMPT[task]}
#
#Reasoning traces: {thinking}
#
#Output:
#"""
#
#    def create_answer_removed_explanation_only_quality_prompt(self, thinking: str, task: str) -> str:
#        return f"""You are tasked with evaluating the reasoning quality of the following model-generated reasoning traces.
#
#Your evaluation criteria are as follows:
#
#1. Focus Only on Reasoning Quality:
#   - Ignore the correctness of the final answer.
#   - Only assess the clarity, coherence, and structure of the reasoning process.
#   - Consider whether each step logically follows from the previous one and contributes meaningfully to the overall reasoning.
#
#2. Scoring Guidelines:
#   - Output must be a single number between 0 and 1, inclusive.
#   - A score of 1.0 represents perfectly clear, logically sound, and well-structured reasoning throughout.
#   - A score of 0.0 represents incoherent, unclear, or completely unstructured reasoning.
#   - You may use intermediate values to reflect partial quality (e.g., 0.3, 0.65, etc.).
#
#3. Formatting Requirements:
#   - Output only the numeric score on a single line.
#   - Do not include any explanation or commentary.
#
#Thinking Traces:
#
#{thinking}
#
#Score:  
#        """

#    def create_answer_removed_explanation_only_quality_prompt_specific(self, thinking: str, question: str) -> str:
#        
#        def _format_scoring_examples():
#            if not examples:
#                return "Here are some examples of scoring:"
#            text = "Here are some examples of scoring:"
#            for example in examples:
#                example_question = str(example.get("question", "")).strip()
#                example_thinking = str(example.get("thinking", "")).strip()
#                example_score = example.get("score", "")
#                example_score_str = str(example_score).strip() if example_score is not None else ""
#                text += (
#                    "\n\n" +
#                    "Input question:\n\n" +
#                    example_question + "\n" +
#                    "\nThinking Traces:\n\n" +
#                    example_thinking + "\n" +
#                    "\nScore:\n" +
#                    example_score_str
#                )
#            return text
#
#        examples_block = _format_scoring_examples()
#
#        return f"""You are tasked with evaluating the reasoning quality of the following model-generated reasoning traces.

#1. Scoring Guidelines:
#   - Output must be a single number between 0 and 1, inclusive.
#   - A score of 1.0 represents perfectly clear, logically and well-structured reasoning that is specific to the input question.
#   - A score of 0.0 represents incoherent, unclear, or completely unstructured reasoning.
#   - A score of 0.0 also represents the reasoning process is generic for all of input and not specific enough.
#   - You may use intermediate values to reflect partial quality (e.g., 0.3, 0.65, etc.).
#
#2. Formatting Requirements:
#   - Output only the numeric score on a single line.
#   - Do not include any explanation or commentary.
#
#{examples_block}
#
#Input question:
#{question}
#
#Thinking Traces:\n
#{thinking}
#
#Score:  
#        """
#

    def create_answer_removed_explanation_only_quality_prompt_specific(self, thinking: str, question: str, examples) -> str:

        return f"""
Your task is to evaluate the reasoning quality of the following model-generated reasoning traces on the task of Spiral Matrix.
Output a single number between 0 and 1 without any additional text.

The instruction of the task is:

Given a matrix, your job is to generate a list of elements in spiral order, starting from the top-left element.

The spiral order is clockwise, starting from the top-left corner. More precisely:
- Start from the top-left corner and move right.
- Move down towards the bottom-right corner.
- Move left towards the bottom-left corner.
- Move up towards the top-right corner.
- Repeat the steps for the inner elements of the matrix until every entry is visited.

Your output should be a space-separated list of integers, e.g. 1 2 3 4 5 6


Here are some examples of the task:

QUESTION 1:
6 6
0 6

REASONING TRACE 1:
For a 2×2 matrix, spiral order goes clockwise from top-left.

First row: 6, 6
Then down the right side: 6
Then left on bottom: 0
Then up (nothing left)

SCORE: 0.55


QUESTION 2:
1 2 3
4 5 6
7 8 9

REASONING TRACE 2:
Let me work through this systematically for the 3×3 matrix.

Starting position: (0,0)
- Move right along row 0: 1, 2, 3
- Move down along column 2: 6, 9
- Move left along row 2: 8, 7
- Move up along column 0: 4
- Move right along row 1: 5

Complete sequence: 1 2 3 6 9 8 7 4 5

Verification:
- Outer layer: 1→2→3→6→9→8→7→4 (8 elements)
- Inner layer: 5 (1 element)
- Total: 9 elements ✓

SCORE: 0.92


QUESTION 3:
5 10
15 20

REASONING TRACE 3:
Going clockwise: 5 15 20 10

SCORE: 0.08


QUESTION 4:
2 4 6
8 10 12

REASONING TRACE 4:
This is a 2×3 matrix.

Spiral traversal:
- Right across top: 2, 4, 6
- Down right side: 12
- Left across bottom: 10, 8

Result: 2 4 6 12 10 8

SCORE: 0.78


QUESTION 5:
7 14
21 28

REASONING TRACE 5:
Matrix is:
7 14
21 28

Top row: 7, 14
Then down: 28
Then left: 21

Wait, I need to reconsider. Let me trace again:
- Start at (0,0): 7
- Right to (0,1): 14
- Down to (1,1): 28
- Left to (1,0): 21

Sequence: 7 14 28 21

SCORE: 0.65


QUESTION 6:
3 6 9
12 15 18
21 24 27

REASONING TRACE 6:
Let me trace the 3×3 matrix carefully.

Step 1 - Outer ring:
- Top row (left to right): 3, 6, 9
- Right column (top to bottom, excluding top): 18, 27
- Bottom row (right to left, excluding right): 24, 21
- Left column (bottom to top, excluding bottom and top): 12

Step 2 - Inner element:
- Center: 15

Full sequence: 3 6 9 18 27 24 21 12 15

Let me verify all 9 elements are included:
3, 6, 9, 12, 15, 18, 21, 24, 27 ✓

SCORE: 0.88


QUESTION 7:
8 16
24 32

REASONING TRACE 7:
To solve this problem, I need to traverse the matrix in a spiral order. This means I should move right along the top row, then down along the right column, then left along the bottom row, and finally up along the left column. I should continue this process iteratively, reducing the size of the matrix at each step, until all elements have been visited.

SCORE: 0.05


QUESTION 8:
1 3
5 7

REASONING TRACE 8:
Starting top-left and going around: 1, 3, 7, 5

SCORE: 0.18

QUESTION 9:
1 2
3 4

REASONING TRACE 9:
To solve this problem, I need to traverse the given 2D matrix in a spiral order. This involves repeating four actions: moving right along the top row, then moving down along the rightmost column, then moving left along the bottom row (excluding the last element since it has already been visited), and finally moving up along the leftmost column (excluding the first element as it has already been visited). I will continue these steps until all elements are visited.

SCORE: 0.1


QUESTION 10:
{question}

REASONING TRACE 10:
{thinking}

SCORE:
"""