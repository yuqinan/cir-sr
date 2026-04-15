from verl.workers.fsdp_workers import RewardModelWorker
from verl import DataProto  # Correct import path
from vllm import LLM, SamplingParams  # Correct vLLM imports
import torch
import random
# Add necessary imports for the decorators and utilities
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.debug import DistProfiler
from verl.utils.device import get_device_id
# Import extraction functions from data.utils
from data.utils import extract_answer, extract_answer_and_think, extract_answer_and_think_auto
# Import RewardCalculator for proper task-specific scoring
from trainers.reward_calculator import RewardCalculator
# Import template for student prompts
from data.template import STUDENT_TEMPLATE_MODIFIED
import numpy as np

class VLLMRewardModelWorker(RewardModelWorker):
    """
    A RewardModelWorker that:
      - pulls the teacher's (actor) generated text
      - feeds it to a student LLM via vLLM
      - computes student's accuracy and returns it as reward for the teacher
    """

    def __init__(self, config, device=None, **kwargs):
        # Call parent constructor first
        super().__init__(config)
        print(f"VLLMRewardModelWorker config: {config}")
        
        # Set device properly - this is missing in the parent class
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Initialize vLLM model with correct API
        model_path = config.model.path
        max_length = config.max_length

        # Store model name for format detection
        self.model_name = model_path

        print(f"Initializing VLLMRewardModelWorker with model: {model_path}")

        self.vllm_model = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.4,  # Reduced memory usage to prevent OOM
            max_model_len=min(max_length, 2048),  # Cap max length to prevent OOM
            quantization=None,
            enforce_eager=True,  # Use eager execution to reduce memory overhead
            disable_log_stats=True  # Disable logging to save memory
        )
        
        # Set up sampling parameters from config
        temperature = getattr(config, 'temperature', 1.0)
        top_p = getattr(config, 'top_p', 0.9)
        top_k = getattr(config, 'top_k', 50)
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        print("VLLMRewardModelWorker initialized successfully")

        self.vllm_tokenizer = self.vllm_model.get_tokenizer()
        
        # Initialize RewardCalculator for proper task-specific scoring
        # Use task_types from reward_model config (passed from verl_grpo_trainer.py)
        # VLLMRewardModelWorker only receives config.reward_model, so use fallback for task
        task_types = getattr(config, 'task_types', ['mini_sudoku'])
        self.reward_calculator = RewardCalculator(
            task="reasoning_gym", 
            task_type=task_types
        )

    def cleanup_memory(self):
        """
        Clean up GPU memory to prevent OOM issues
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Override parent's init_model to also check tokenizers
        """
        # print("[DEBUG] VLLMRewardModelWorker.init_model called")
        
        # Call parent's init_model
        super().init_model()


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data_proto: DataProto):
        """
        Compute reward scores using vLLM model.
        This method name and signature matches the parent class expectation.
        """
        # if get_device_id() == 0:
        #     # print(f"[DEBUG] compute_rm_score called with data_proto: {type(data_proto)}")
        
        data_proto = data_proto.to(get_device_id())
        
        # Initialize reward tensor following verl_grpo_trainer.py pattern - ensure same device
        reward_tensor = torch.zeros_like(data_proto.batch["responses"], dtype=torch.float32)
        
        
        #decoded_prompts = data_proto.non_tensor_batch["decoded_prompts"]
        #decoded_responses = data_proto.non_tensor_batch["decoded_responses"]
        
        # Process each item to create student prompts
        student_prompts = []
        response_lengths = []
        
        for i in range(len(data_proto)):
            data_item = data_proto[i]  # DataProtoItem
            
            # Use pre-decoded text from teacher tokenizer
            prompt_str = data_item.non_tensor_batch["decoded_prompts"]
            response_str = data_item.non_tensor_batch["decoded_responses"]
            
            # Calculate response length from token data for reward placement
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            response_lengths.append(valid_response_length)
            
            # Extract thinking traces from teacher response for student prompt
            extracted = extract_answer_and_think_auto(response_str, model_name=self.model_name)
            thinking_trace = extracted[0] if extracted else ""  # First element is thinking
            
            # Create and format student prompt with chat template
            #student_prompt = STUDENT_TEMPLATE_MODIFIED.format(
            #    question=data_item.non_tensor_batch["question"],
            #    explanation=thinking_trace
            #)
            
            chat_messages = [
                {"role": "system", "content": self.developer_prompt},
                {"role": "user", "content": data_item.non_tensor_batch["question"]}]
            formatted_prompt = self.vllm_tokenizer.apply_chat_template(
                chat_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            formatted_prompt = formatted_prompt + "<think>" + thinking_trace + "</think>\n<answer>"
            
            student_prompts.append(formatted_prompt)
            
            # Simple debug for first item only from worker 0
            # if get_device_id() == 0:
            #     if i == 0:
            #         # print(f"[COMPUTE_RM_SCORE DEBUG] prompt length: {len(prompt_str)}", flush=True)
            #         # clean_prompt = prompt_str.replace('\n', ' ')
            #         # print(f"[COMPUTE_RM_SCORE DEBUG] prompt: {clean_prompt}", flush=True)
            #         # print(f"[COMPUTE_RM_SCORE DEBUG] response length: {len(response_str)}", flush=True)
            #         # clean_response = response_str.replace('\n', ' ')
            #         # print(f"[COMPUTE_RM_SCORE DEBUG] response: {clean_response}", flush=True)
        
        data_proto.non_tensor_batch["student_prompt"] = np.array(student_prompts, dtype=object)

        # if get_device_id() == 0:
        #     # print(f"[FINISH GENERATING PROMPTS DEBUG] Created {len(data_proto)} student prompts, sending to vLLM...", flush=True)
        
        # Generate responses for the whole batch using vLLM (student model)
        # Clear GPU cache before generation to prevent OOM
        torch.cuda.empty_cache()
        
        # Process in smaller batches to prevent OOM
        student_answers = []
        total_student_prompts = len(data_proto)
        batch_size = min(16, total_student_prompts)  # Small batch size
        for batch_start in range(0, total_student_prompts, batch_size):
            batch_end = min(batch_start + batch_size, total_student_prompts)
            batch_prompts = data_proto[batch_start:batch_end]
            
            # Generate with vLLM for the mini-batch
            outputs = self.vllm_model.generate(batch_prompts.non_tensor_batch["student_prompt"], self.sampling_params, use_tqdm=False)
            batch_answers = [output.outputs[0].text.strip() for output in outputs]
            student_answers.extend(batch_answers)
            
            # Clear cache after each batch
            torch.cuda.empty_cache()

        # if get_device_id() == 0:
        #     # clean_batch_prompt = data_proto[0].non_tensor_batch['student_prompt'].replace('\n', ' ')
        #     # clean_student_answer = student_answers[0].replace('\n', ' ')
        #     # print(f"[DECODING STUDENT ANSWERS DEBUG] batch_prompts: {clean_batch_prompt}", flush=True)       
        #     # print(f"[DECODING STUDENT ANSWERS DEBUG] student_answers: {clean_student_answer}", flush=True)
        
        # if get_device_id() == 0:
        #     # print(f"[FINISH GENERATING ANSWERS DEBUG] Generated {len(student_answers)} student answers, now scoring...", flush=True)
        
        data_proto.non_tensor_batch["student_answer"] = np.array(student_answers, dtype=object)

        # Compute scores for each item using RewardCalculator (like verl_grpo_trainer.py)
        for i in range(len(data_proto)):
            data_item = data_proto[i]
            student_response = student_answers[i]
            valid_response_length = response_lengths[i]
            student_response = "<answer>" + student_response
            
            # Extract answer from student response before evaluation
            student_answer = extract_answer(student_response, "answer")
            if not student_answer:
                # If no answer tag found, use the raw response
                student_answer = student_response.strip()
            
            # Reconstruct entry for RewardCalculator (similar to verl_grpo_trainer.py)
            index = data_item.non_tensor_batch.get("index", i)
            
            entry = {
                'answer': data_item.non_tensor_batch.get("answer", ""),
                'metadata': data_item.non_tensor_batch.get("metadata", {}),
                'data_source': data_item.non_tensor_batch.get("data_source", "mini_sudoku"),
                'index': index
            }
            
            # Ensure metadata has source_dataset for RewardCalculator
            if 'metadata' not in entry or 'source_dataset' not in entry['metadata']:
                entry['metadata'] = {'source_dataset': entry.get('data_source', 'mini_sudoku')}
            
            # Use RewardCalculator for proper task-specific scoring (student version - no format bonus)
            score = self.reward_calculator.calculate_student_reward(student_answer, entry)
            
            # Set reward at the end of the sequence like verl_grpo_trainer.py
            reward_tensor[i, valid_response_length - 1] = score
            
            # Enhanced debug output showing RewardCalculator scoring
            if get_device_id() == 0:
                if i == 0:
                    answer = entry.get('answer', '')
                    clean_question = data_item.non_tensor_batch['decoded_prompts'].replace('\n', ' ')
                    clean_teacher_response = data_item.non_tensor_batch['decoded_responses'].replace('\n', ' ')
                    print(f"[FINAL DEBUG] Item {i}:", flush=True)
                    print(f"[FINAL DEBUG] Question: '{clean_question}'", flush=True)
                    print(f"[FINAL DEBUG] Teacher Response: '{clean_teacher_response}'", flush=True)
                    print(f"[FINAL DEBUG] Student Prompt: '{student_prompt}'", flush=True)
                    print(f"[FINAL DEBUG] Student Answer: '{student_answer}'", flush=True)
                    print(f"[FINAL DEBUG] Ground Truth: '{str(answer)}'", flush=True)
                    print(f"[FINAL DEBUG] Score: {score:.3f}", flush=True)
                    print(flush=True)

        
        # Clean up memory after processing
        self.cleanup_memory()
        
        # Return DataProto with rm_scores key (not raw tensor)
        output = DataProto.from_dict({"rm_scores": reward_tensor})
        return output.to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="green")
    def compute_few_shot_reward(self, data_proto: DataProto, n_shot: int = 3):
        """
        Compute few-shot reward by:
        1. Extract thinking traces from teacher responses 
        2. Form n-shot examples from different indices
        3. Test student on new sudoku problems using these examples
        4. Return student accuracy
        """
        # if get_device_id() == 0:
        #     # print(f"[DEBUG] compute_few_shot_reward called with n_shot={n_shot}")
        
        data_proto = data_proto.to(get_device_id())
        
        # Use pre-decoded text from the teacher tokenizer
        decoded_prompts = data_proto.non_tensor_batch["decoded_prompts"]
        decoded_responses = data_proto.non_tensor_batch["decoded_responses"]
        
        # Step 1: Extract thinking traces and answers from teacher responses
        examples = []
        for i in range(len(data_proto)):
            data_item = data_proto[i]
            response_str = decoded_responses[i]
            
            # Extract thinking trace and answer using data.utils functions
            extracted = extract_answer_and_think_auto(response_str, model_name=self.model_name)
            if extracted:
                think, answer = extracted  # Correct order: (thinking, answer)
            else:
                # Try to extract just the answer
                answer = extract_answer(response_str, "answer")
                think = extract_answer(response_str, "think")
                
                # If extraction fails, use empty strings
                if not answer:
                    answer = ""
                if not think:
                    think = ""
            
            # Get the original question from prompt
            prompt_str = decoded_prompts[i]
            
            examples.append({
                'question': prompt_str,
                'thinking': think,
                'answer': answer,
                'index': i
            })
        
        # if get_device_id() == 0:
        #     # print(f"[DEBUG] Extracted {len(examples)} examples with thinking traces")
        
        # Step 2: Form n-shot examples from different indices
        if len(examples) < n_shot + 1:  # Need at least n_shot examples + 1 for testing
            # if get_device_id() == 0:
            #     # print(f"[DEBUG] Not enough examples ({len(examples)}) for {n_shot}-shot learning")
            return DataProto.from_dict({"student_acc": 0.0}).to("cpu")
        
        # Randomly select n_shot examples for demonstration
        demo_indices = random.sample(range(len(examples)), n_shot)
        demo_examples = [examples[i] for i in demo_indices]
        
        # Use remaining examples for testing (select a few for evaluation)
        test_indices = [i for i in range(len(examples)) if i not in demo_indices]
        test_examples = random.sample(test_indices, min(10, len(test_indices)))  # Test on up to 10 examples
        
        # if get_device_id() == 0:
        #     # print(f"[DEBUG] Using {len(demo_examples)} demo examples and {len(test_examples)} test examples")
        
        # Step 3: Create few-shot prompts and test student
        correct_predictions = 0
        total_predictions = 0
        
        for test_idx in test_examples:
            test_example = examples[test_idx]
            
            # Build few-shot prompt
            few_shot_prompt = "Here are some examples of solving mini sudoku puzzles:\n\n"
            
            for i, demo in enumerate(demo_examples):
                few_shot_prompt += f"Example {i+1}:\n"
                few_shot_prompt += f"Question: {demo['question'][:200]}...\n"
                if demo['thinking']:
                    few_shot_prompt += f"Thinking: {demo['thinking']}\n"
                few_shot_prompt += f"Answer: {demo['answer']}\n\n"
            
            few_shot_prompt += f"Now solve this new puzzle:\n"
            few_shot_prompt += f"Question: {test_example['question'][:200]}...\n"
            few_shot_prompt += f"Answer with just the final sudoku grid:\n"
            
            # Generate student response using vLLM
            torch.cuda.empty_cache()
            
            # Use longer max_tokens for sudoku solutions
            few_shot_sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                top_k=50,
                max_tokens=200,  # Longer for full sudoku grid
                stop=["Question:", "Example", "Now solve"]
            )
            
            outputs = self.vllm_model.generate([few_shot_prompt], few_shot_sampling_params, use_tqdm=False)
            student_response = outputs[0].outputs[0].text.strip()
            
            # Extract answer from student response before evaluation
            student_answer = extract_answer(student_response, "answer")
            if not student_answer:
                # If no answer tag found, use the raw response
                student_answer = student_response.strip()
            
            # Get entry from data_proto and use RewardCalculator
            data_item = data_proto[test_idx]
            index = data_item.non_tensor_batch.get("index", test_idx)
            
            entry = {
                'answer': data_item.non_tensor_batch.get("answer", ""),
                'metadata': data_item.non_tensor_batch.get("metadata", {}),
                'data_source': data_item.non_tensor_batch.get("data_source", "mini_sudoku"),
                'index': index
            }
            
            # Ensure metadata has source_dataset for RewardCalculator
            if 'metadata' not in entry or 'source_dataset' not in entry['metadata']:
                entry['metadata'] = {'source_dataset': entry.get('data_source', 'mini_sudoku')}
            
            # Use RewardCalculator for consistent scoring (student version - no format bonus)
            score = self.reward_calculator.calculate_student_reward(student_answer, entry)
            
            # Consider it correct if score is above threshold (no format bonus for student)
            is_correct = score >= 1.0  # Exact match gets 1.0 (no format bonus)
            
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            # Debug first few predictions
            # if total_predictions <= 3:
            #     answer = entry.get('answer', '')
            #     # print(f"[DEBUG] Test {total_predictions}: Student='{student_answer[:30]}...' vs Answer='{answer[:30]}...'")
            #     # print(f"[DEBUG] RewardCalculator Score={score:.3f}, Correct={is_correct}")
        
        # Calculate student accuracy
        student_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # if get_device_id() == 0:
        #     # print(f"[DEBUG] Few-shot evaluation: {correct_predictions}/{total_predictions} = {student_accuracy:.3f}")
        
        # Clean up memory
        self.cleanup_memory()
        
        # Return student accuracy
        output = DataProto.from_dict({"student_acc": student_accuracy})
        return output.to("cpu")
