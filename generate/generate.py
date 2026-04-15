#!/usr/bin/env python3
"""
Simplified Teacher Evaluation Script

Clean, focused script for running teacher model evaluation.
Reduced from 726 lines to ~300 lines by extracting orchestration logic.
"""

import os
import sys
import logging
from pprint import pprint
from datetime import datetime
import hydra
from omegaconf import OmegaConf
import pandas as pd
import re
import json
from data.template import SYSTEM_PROMPTS
import asyncio
import aiohttp
from tqdm.asyncio import tqdm as async_tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.teacher_evaluator import TeacherEvaluator
from evaluate.utils.answer_removal import remove_answers_batch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('teacher_eval.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def filter_and_save_to_parquet(config):
    tokenizer = AutoTokenizer.from_pretrained(config.evaluation.base_model_path)

    with open(config.evaluation.output_dir + "/teacher/step_0/teacher_responses_step_0.json", "r") as f:
        data = json.load(f)
    #filtered_data = [item for item in data if item["answer_removed_explanation_only_score"] == 1]
    for idx, item in enumerate(data):
        data[idx]['reward_score'] = item['k_responses'][0]['reward_score']
        data[idx]['teacher_thinking'] = item['k_responses'][0]['teacher_thinking']
        data[idx]['teacher_answer'] = item['k_responses'][0]['teacher_answer']
        data[idx]['teacher_response'] = item['k_responses'][0]['teacher_response']

    #filtered_data = [item for item in data if item["answer_removed_explanation_only_score"] == 1]
    filtered_data = [item for item in data if item["reward_score"] == 1]
    def filter_teacher_think(item):
        item['teacher_thinking_without_answer'] = re.sub(item['teacher_answer'], "", item['teacher_thinking'])
        return item
    filtered_data = [filter_teacher_think(item) for item in filtered_data if item["teacher_response"] is not None]

    print(f"Total filtered data: {len(filtered_data)}")
    filtered_data = pd.DataFrame(filtered_data)

    train_data = filtered_data
    val_data = filtered_data
    
    try:
        postpend = config.evaluation.postpend
    except:
        postpend = ""
    # Create directories
    os.makedirs(f"generate/train_data_{postpend}/{config.evaluation.teacher_model.openai_model_name}_{config.evaluation.student_model.model_path}", exist_ok=True)
    os.makedirs(f"generate/val_data_{postpend}/{config.evaluation.teacher_model.openai_model_name}_{config.evaluation.student_model.model_path}", exist_ok=True)
    
    # Save train data
    train_save_path = re.sub('results', f'train_data_{postpend}', config.evaluation.output_dir)
    train_data.to_parquet(train_save_path + ".parquet")
    # Save train data as JSON too
    train_data.to_json(train_save_path + ".json", orient='records', indent=2)
    
    # Save val data
    val_save_path = re.sub('results', f'val_data_{postpend}', config.evaluation.output_dir)
    val_data.to_parquet(val_save_path + ".parquet")
    # Save val data as JSON too
    val_data.to_json(val_save_path + ".json", orient='records', indent=2)


'''
def make_perturbed_inputs(data, config, tokenizer):
    import random
    
    system_chat = SYSTEM_PROMPTS["DeepSeekZero"]
    
    for idx, item in enumerate(data):
        # Randomly pick another instance from the data
        other_items = [other_item for other_item in data if other_item.get("question") != item.get("question")]
        if other_items:
            # Use a fixed seed based on index for reproducibility
            random.seed(idx)
            random_other = random.choice(other_items)
            other_thinking = random_other.get('teacher_thinking_without_answer', '')
            other_answer = random_other.get('teacher_answer', '')
        else:
            other_thinking = ''
            other_answer = ''
        
        # Create the perturbed input by adding the other instance's thinking
        if system_chat is not None:
            prompt_chat = [{'role': 'system', 'content': system_chat}, {"role": "user", 'content': item['question']}]
        else:
            prompt_chat = [{"role": "user", "content": item['question']}]

        # Convert to string
        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False
        )
        
        # Add the other instance's thinking to create perturbed input
        data[idx]['perturbed_input'] = prompt_chat_str + other_thinking + "</think>"
        data[idx]["perturbed_response"] = "<answer>" + other_answer + "</answer>"
        
    return data
'''


def filter_and_save_to_parquet_perturbed(config):
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.evaluation.base_model_path)

    with open(config.evaluation.output_dir + "/teacher/step_0/teacher_responses_step_0.json", "r") as f:
        data = json.load(f)
    #filtered_data = [item for item in data if item["answer_removed_explanation_only_score"] == 1]
    for idx, item in enumerate(data):
        data[idx]['reward_score'] = item['k_responses'][0]['reward_score']
        data[idx]['teacher_thinking'] = item['k_responses'][0]['teacher_thinking']
        data[idx]['teacher_answer'] = item['k_responses'][0]['teacher_answer']
        data[idx]['teacher_response'] = item['k_responses'][0]['teacher_response']

    filtered_data = [item for item in data if item["reward_score"] == 1]
    def filter_teacher_think(item):
        item['teacher_thinking_without_answer'] = re.sub(item['teacher_answer'], "", item['teacher_thinking'])
        return item

    filtered_data = [filter_teacher_think(item) for item in filtered_data if item["teacher_response"] is not None]

    print(f"Total filtered data: {len(filtered_data)}")
    
    # Shuffle the data randomly
    filtered_data  = make_perturbed_inputs(filtered_data , config)

    # Convert back to DataFrame
    filtered_data = pd.DataFrame(filtered_data)

    train_data = filtered_data
    val_data = filtered_data

    try:
        postpend = config.evaluation.postpend
    except:
        postpend = ""
    # Create directories
    os.makedirs(f"generate/train_data_{postpend}/{config.evaluation.teacher_model.openai_model_name}_{config.evaluation.student_model.model_path}", exist_ok=True)
    os.makedirs(f"generate/val_data_{postpend}/{config.evaluation.teacher_model.openai_model_name}_{config.evaluation.student_model.model_path}", exist_ok=True)
    
    # Save train data
    train_save_path = re.sub('results', f'train_data_{postpend}', config.evaluation.output_dir)
    train_data.to_parquet(train_save_path + ".parquet")
    # Save train data as JSON too
    train_data.to_json(train_save_path + ".json", orient='records', indent=2)
    
    # Save val data
    val_save_path = re.sub('results', f'val_data_{postpend}', config.evaluation.output_dir)
    val_data.to_parquet(val_save_path + ".parquet")
    # Save val data as JSON too
    val_data.to_json(val_save_path + ".json", orient='records', indent=2)



def generate_dataset_only(config):
    """
    Generate reasoning gym dataset and save to parquet without running evaluation.
    Activated when config.evaluation.dataset_only is True.
    """
    logger.info("Running dataset-only mode - generating data without evaluation")

    # Import data generator
    from evaluate.utils.data_generator import DataGenerator

    # Get dataset configuration
    dataset_config = OmegaConf.to_container(config.evaluation.teacher_dataset, resolve=True)

    task_params = {k: v for k, v in dataset_config.items()
                  if k not in ['task_name', 'seed', 'size', 'val_start']}

    # Initialize data generator
    data_generator = DataGenerator(dataset_config['task_name'], task_params)

    # Generate dataset
    logger.info(f"Generating {dataset_config['size']} examples for task: {dataset_config['task_name']}")
    data = data_generator.generate_teacher_dataset(
        dataset_config['seed'],
        dataset_config['size'],
        dataset_config.get('val_start', 0)
    )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    logger.info(f"Generated {len(df)} examples")

    # Determine output path
    try:
        postpend = config.evaluation.postpend
    except:
        postpend = "dataset"

    # Create output directory
    output_dir = f"generate/dataset_{postpend}"
    os.makedirs(output_dir, exist_ok=True)

    # Save to parquet and json
    task_name = dataset_config['task_name']
    output_path = os.path.join(output_dir, task_name)

    df.to_parquet(output_path + ".parquet")
    df.to_json(output_path + ".json", orient='records', indent=2)

    logger.info(f"Saved dataset to {output_path}.parquet and {output_path}.json")
    logger.info(f"Dataset generation complete!")

    return df

def make_perturbed_inputs(data, config):
    import random

    for idx, item in enumerate(data):
        # Randomly pick another instance from the data
        other_items = [other_item for other_item in data if other_item.get("question") != item.get("question")]
        if other_items:
            # Use a fixed seed based on index for reproducibility
            random.seed(idx)
            random_other = random.choice(other_items)
            other_thinking = random_other.get('teacher_thinking_without_answer', '')
            other_answer = random_other.get('teacher_answer', '')
        else:
            other_thinking = ''
            other_answer = ''

        data[idx]['perturbed_thinking'] = other_thinking + "</think>" + "<answer>"
        data[idx]['perturbed_response'] = data[idx]['perturbed_thinking']  + other_answer + "</answer>"
        data[idx]['perturbed_answer'] = other_answer + "</answer>"

    return data

def make_perturb_another_task_input(task1, task2):
    import random

    df1 = pd.read_parquet(task1 + ".parquet")
    df2 = pd.read_parquet(task2 + ".parquet")

    df3 = df1.copy()

    # Convert df2 to list for random sampling
    df2_data = df2.to_dict('records')

    for idx in range(len(df3)):
        # Use a fixed seed based on index for reproducibility
        random.seed(idx)
        random_other = random.choice(df2_data)

        other_thinking = random_other.get('teacher_thinking_without_answer', '')
        other_answer = random_other.get('teacher_answer', '')

        df3.at[idx, 'perturbed_thinking'] = other_thinking + "</think>" + "<answer>"
        df3.at[idx, 'perturbed_response'] = df3.at[idx, 'perturbed_thinking'] + other_answer + "</answer>"
        df3.at[idx, 'perturbed_answer'] = other_answer + "</answer>"

    return df3

def load_and_filter_responses(config, from_existing=False):
    """
    Load teacher responses and filter for correct answers.

    Args:
        config: Hydra config object
        from_existing: If True, load from existing formatted data using read_in_postpend

    Returns:
        List of filtered data entries
    """
    
    if from_existing:
        # Load from existing formatted data
        read_in_postpend = config.evaluation.get('read_in_postpend', None)
        if not read_in_postpend:
            logger.error("read_in_postpend not specified in config when from_existing=True")
            raise ValueError("read_in_postpend required when loading from existing data")

        # Construct path to existing formatted data
        read_in_path = re.sub('results', f'train_data_{read_in_postpend}', config.evaluation.output_dir)
        json_path = read_in_path + ".json"

        logger.info(f"Loading existing formatted data from {json_path}")
        with open(json_path, "r") as f:
            filtered_data = json.load(f)

        logger.info(f"Loaded {len(filtered_data)} entries from existing data")
        return filtered_data

    else:
        # Load from raw teacher responses
        with open(config.evaluation.output_dir + "/teacher/step_0/teacher_responses_step_0.json", "r") as f:
            data = json.load(f)
        #filtered_data = [item for item in data if item["answer_removed_explanation_only_score"] == 1]
        for idx, item in enumerate(data):
            data[idx]['reward_score'] = item['k_responses'][0]['reward_score']
            data[idx]['teacher_thinking'] = item['k_responses'][0]['teacher_thinking']
            data[idx]['teacher_answer'] = item['k_responses'][0]['teacher_answer']
            data[idx]['teacher_response'] = item['k_responses'][0]['teacher_response']


        # Filter for correct responses
        filtered_data = [item for item in data if item["reward_score"] == 1]

        # Use GPT-4o-mini to remove answers (with fallback to regex if API key not available)
        filtered_data = remove_answers_batch(filtered_data, use_llm=True)
        for idx, i in enumerate(filtered_data):
            filtered_data[idx]['teacher_thinking_without_answer'] = filtered_data[idx]['teacher_thinking_without_answer'] + "</think>"
        logger.info(f"Filtered to {len(filtered_data)} correct responses")

        return filtered_data


def save_train_val_data(filtered_data, config, postpend=None):
    """
    Save filtered data as train and val parquet/json files.

    Args:
        filtered_data: DataFrame or list of data to save
        config: Hydra config object
        postpend: Optional postpend string for output directory
    """
    # Convert to DataFrame if needed
    if not isinstance(filtered_data, pd.DataFrame):
        filtered_data = pd.DataFrame(filtered_data)

    if postpend is None:
        postpend = config.evaluation.get('postpend', '')

    # Create directories
    os.makedirs(f"generate/train_data_{postpend}/{config.evaluation.teacher_model.openai_model_name}_{config.evaluation.student_model.model_path}", exist_ok=True)
    os.makedirs(f"generate/val_data_{postpend}/{config.evaluation.teacher_model.openai_model_name}_{config.evaluation.student_model.model_path}", exist_ok=True)

    # Save train data
    train_save_path = re.sub('results', f'train_data_{postpend}', config.evaluation.output_dir)
    filtered_data.to_parquet(train_save_path + ".parquet")
    filtered_data.to_json(train_save_path + ".json", orient='records', indent=2)

    # Save val data
    val_save_path = re.sub('results', f'val_data_{postpend}', config.evaluation.output_dir)
    filtered_data.to_parquet(val_save_path + ".parquet")
    filtered_data.to_json(val_save_path + ".json", orient='records', indent=2)

    logger.info(f"Saved data to {train_save_path}.parquet and {val_save_path}.parquet")


@hydra.main(config_path="configs", config_name="mini_sudoku", version_base=None)
def main(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Get mode from config (default to "expert_traces")
    mode = config.evaluation.get('mode', 'expert_traces')
    logger.info(f"Running in mode: {mode}")

    if mode == "dataset_only":
        # Mode 1: Generate dataset only without evaluation
        logger.info("=== DATASET ONLY MODE ===")
        generate_dataset_only(config)

    elif mode == "expert_traces":
        # Mode 2: Default mode - run evaluation to generate expert traces
        logger.info("=== EXPERT TRACES MODE ===")
        evaluator = TeacherEvaluator(config)
        evaluator.run_evaluation()

        # Load and filter responses from raw teacher outputs
        filtered_data = load_and_filter_responses(config, from_existing=False)

        # Save without perturbations
        save_train_val_data(filtered_data, config)

    elif mode == "perturb_expert_traces":
        # Mode 3: Load existing expert traces and add same-task perturbations
        logger.info("=== PERTURB EXPERT TRACES MODE ===")
        logger.info("Loading existing expert traces from read_in_postpend")

        # Load from existing formatted data (requires read_in_postpend in config)
        filtered_data = load_and_filter_responses(config, from_existing=True)

        # Add same-task perturbations
        filtered_data = make_perturbed_inputs(filtered_data, config)

        # Save with perturbations
        save_train_val_data(filtered_data, config)

    elif mode == "perturb_other_traces":
        # Mode 4: Load existing expert traces and add other-task perturbations
        logger.info("=== PERTURB OTHER TRACES MODE ===")
        logger.info("Loading existing expert traces from read_in_postpend")

        # Get other traces config
        other_traces_config = config.evaluation.get('other_traces_dir', {})
        if not other_traces_config:
            logger.error("other_traces_dir not specified in config for perturb_other_traces mode")
            raise ValueError("other_traces_dir required for perturb_other_traces mode")

        # Load from existing formatted data (requires read_in_postpend in config)
        filtered_data = load_and_filter_responses(config, from_existing=True)

        # Add perturbations from other tasks
        filtered_data = add_perturb_other_traces(filtered_data, other_traces_config)

        # Save with perturbations
        postpend = config.evaluation.get('postpend', 'perturb_other_traces')
        save_train_val_data(filtered_data, config, postpend=postpend)

    else:
        logger.error(f"Unknown mode: {mode}")
        raise ValueError(f"Mode must be one of: dataset_only, expert_traces, perturb_expert_traces, perturb_other_traces")





if __name__ == "__main__":
    main()