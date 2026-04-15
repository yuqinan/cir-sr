#!/usr/bin/env python3
"""
Combined plot showing CoT importance and Verifier Accuracy evolution across training.

Usage:
    python plot_cot_verifier_combined.py [model_size]
    python plot_cot_verifier_combined.py 3b
    python plot_cot_verifier_combined.py 7b
    python plot_cot_verifier_combined.py 1.5b
    python plot_cot_verifier_combined.py 3b --local  # for local testing
    python plot_cot_verifier_combined.py 7b --local  # for local testing with 7b model
    python plot_cot_verifier_combined.py 1.5b --local  # for local testing with 1.5b model
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set font to serif for consistency
plt.rcParams['font.family'] = 'serif'


def load_task_categories(json_path):
    """Load task categories from JSON file."""
    with open(json_path, 'r') as f:
        categories = json.load(f)

    # Create a mapping from task name to category
    task_to_category = {}
    for category, tasks in categories.items():
        for task in tasks:
            task_to_category[task] = category

    return task_to_category


def is_math_task(task_name, task_to_category):
    """Check if a task is a math task (Algebra, Arithmetic, or Geometry)."""
    category = task_to_category.get(task_name, "")
    return category in ["Algebra", "Arithmetic", "Geometry"]


def load_jsonl_file(file_path):
    """Load a JSONL file and return list of entries."""
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def calculate_average_accuracy(entries):
    """Calculate average score from JSONL entries."""
    scores = []
    for entry in entries:
        if 'score' in entry:
            scores.append(entry['score'])
        elif 'reward_score' in entry:
            scores.append(entry['reward_score'])
        elif 'accuracy' in entry:
            scores.append(entry['accuracy'])
    if scores:
        return np.mean(scores)
    return 0.0


def load_step_accuracy(base_dir, task_name, step, variant="original", batch="8", debug=False):
    """Load accuracy for a specific step."""
    rollout_dir = Path(base_dir) / task_name / variant / variant / batch
    if debug:
        print(f"  Checking rollout_dir: {rollout_dir}")
        print(f"  Rollout dir exists: {rollout_dir.exists()}")
    if not rollout_dir.exists():
        return None
    step_file = rollout_dir / f"{step}.jsonl"
    if debug:
        print(f"  Checking step_file: {step_file}")
        print(f"  Step file exists: {step_file.exists()}")
    if not step_file.exists():
        return None
    try:
        entries = load_jsonl_file(step_file)
        acc = calculate_average_accuracy(entries)
        if debug:
            print(f"  Loaded accuracy: {acc}")
        return acc
    except Exception as e:
        if debug:
            print(f"  Error loading: {e}")
        return None


def load_cot_importance_at_step(base_dir, task_name, step):
    """Load CoT importance for a specific step. Returns value or None."""
    folder_name = f"{task_name}_cot_importance"
    paths = [
        os.path.join(base_dir, task_name, "-1.0", "teacher", f"step_{step}", f"teacher_responses_step_{step}.json"),
        os.path.join(base_dir, folder_name, "-1.0", "teacher", f"step_{step}", f"teacher_responses_step_{step}.json"),
    ]

    for json_path in paths:
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Check if number of examples is less than 95
                if len(data) < 95:
                    return None

                # Sample at 11 positions: 0%, 10%, 20%, ..., 100%
                percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                instance_vals = []
                for item in data:
                    if 'cot_importance_evaluation' in item:
                        js_divs = item['cot_importance_evaluation'].get('js_divergences', [])
                        if len(js_divs) >= 2:
                            sampled_values = []
                            for p in percentages:
                                idx = max(0, min(int((p/100.0)*len(js_divs))-1, len(js_divs)-1))
                                # Special handling for 0% - use index 0
                                if p == 0:
                                    idx = 0
                                sampled_values.append(js_divs[idx])
                            # Average the 11 sampled values
                            instance_vals.append(np.mean(sampled_values))
                return np.mean(instance_vals) if instance_vals else None
            except:
                pass
    return None


def load_verifier_accuracy_at_step(base_dir, task_name, step):
    """
    Load average verifier accuracy from verifier_comparison.answers_match for a task at a specific training step.

    Special handling: If answers_match is True but both with_question_answer and without_question_answer
    are "no answer found", count as 0 (incorrect).

    Args:
        base_dir: Base directory containing task results
        task_name: Name of the task folder
        step: Training step number

    Returns:
        Average verifier accuracy or None if data not found
    """
    # Try different possible paths
    possible_paths = [
        os.path.join(base_dir, task_name, "-1.0", "teacher", f"step_{step}", f"teacher_responses_step_{step}.json"),
        os.path.join(base_dir, task_name, "1.0", "teacher", f"step_{step}", f"teacher_responses_step_{step}.json"),
        os.path.join(base_dir, f"{task_name}_cot_importance", "-1.0", "teacher", f"step_{step}", f"teacher_responses_step_{step}.json"),
    ]

    json_path = None
    for path in possible_paths:
        if os.path.exists(path):
            json_path = path
            break

    if not json_path:
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return None

    verifier_scores = []

    for item in data:
        if 'verifier_comparison' in item:
            verifier_comp = item['verifier_comparison']
            answers_match = verifier_comp.get('answers_match', False)
            answers_match_original = verifier_comp.get('answers_match_original', False)

            # Check if both answers are "no answer found"
            with_q_answer = verifier_comp.get('with_question_answer', '').strip().lower()
            without_q_answer = verifier_comp.get('without_question_answer', '').strip().lower()

            # If both are "no answer found", treat as incorrect (0) even if answers_match is True
            if answers_match and with_q_answer == 'no answer found' and without_q_answer == 'no answer found':
                verifier_scores.append(0)
            else:
                verifier_scores.append(1 if answers_match and answers_match_original else 0)

    if verifier_scores:
        return np.mean(verifier_scores)
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description='Plot CoT importance and Verifier Accuracy combined')
    parser.add_argument('model_size', nargs='?', default='3b', help='Model size: 1.5b, 3b, or 7b (default: 3b)')
    parser.add_argument('--local', action='store_true', help='Use local paths (for testing)')
    parser.add_argument('--model', type=str, default=None, help='Model name for rollout data (default: auto-detected from model_size)')
    parser.add_argument('--thinking_dir', type=str, default='grpo', help='Directory name for thinking variant')
    parser.add_argument('--variant', type=str, default='original', help='Variant folder name (default: original)')
    parser.add_argument('--batch', type=str, default='8', help='Batch folder name (default: 8)')
    parser.add_argument('--base_dir', type=str, default=None, help='Override base directory for eval results (e.g. /nlp/scr/.../grpo_llama-3-3b-instruct)')
    parser.add_argument('--steps', type=int, nargs='+', default=None, help='Training steps to use (default: [2, 156])')

    args = parser.parse_args()

    # Auto-detect model name from model_size if not provided
    if args.model is None:
        if args.model_size == '1.5b':
            args.model = 'q1.5b-instruct'
        elif args.model_size == '3b':
            args.model = 'q3b-instruct'
        elif args.model_size == '7b':
            args.model = 'q7b-instruct'
        elif args.model_size == 'llama-3-3b':
            args.model = 'llama-3-3b-instruct'
        else:
            # Default fallback
            args.model = f'q{args.model_size}-instruct'

    # Define training steps (only before and after training)
    if args.steps:
        training_steps = args.steps
    elif args.model_size == 'llama-3-3b':
        training_steps = [3, 156]
    else:
        training_steps = [2, 156]
    step_labels = {training_steps[0]: "Before Training", training_steps[-1]: "After Training"}

    # Load task categories
    task_category_path = os.path.join(os.path.dirname(__file__), 'task_category.json')
    if not os.path.exists(task_category_path):
        print(f"Warning: task_category.json not found at {task_category_path}")
        task_to_category = {}
    else:
        task_to_category = load_task_categories(task_category_path)
        print(f"Loaded task categories from {task_category_path}")

    # Determine result directory name based on model
    if args.model_size == 'llama-3-3b':
        result_dir_name = 'grpo_llama-3-3b-instruct'
    else:
        result_dir_name = f'grpo_qwen-{args.model_size}-instruct'

    # Set base directories
    if args.base_dir:
        base_dir = os.path.join(args.base_dir, 'cot_importance')
        rollout_base = os.path.join(args.base_dir, 'rollout_data')
        output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis/graph'
    elif args.local:
        base_dir = f'/Users/qinanyu/Desktop/rl-explanations/evaluate/results/{result_dir_name}/cot_importance'
        rollout_base = f'/Users/qinanyu/Desktop/rl-explanations/trainers/{args.thinking_dir}/{args.model}/rollout_data'
        output_dir = '/Users/qinanyu/Desktop/rl-explanations/analysis/graph'
    else:
        base_dir = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/{result_dir_name}/cot_importance'
        rollout_base = f'/nlp/scr/qinanyu/rl-explanations/trainers/{args.thinking_dir}/{args.model}/rollout_data'
        output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis/graph'

    print(f"\n{'='*80}")
    print(f"DEBUG: Configuration")
    print(f"{'='*80}")
    print(f"Model size: {args.model_size}")
    print(f"Model name: {args.model}")
    print(f"Base directory: {base_dir}")
    print(f"Rollout base: {rollout_base}")
    print(f"Base dir exists: {os.path.exists(base_dir)}")
    print(f"Rollout base exists: {os.path.exists(rollout_base)}")

    if not os.path.exists(base_dir):
        print(f"ERROR: Base directory does not exist: {base_dir}")
        return

    print(f"\nBase directory contents: {os.listdir(base_dir)[:10]}")  # Show first 10 items

    # Collect data
    task_folders = [item.replace('_cot_importance', '')
                    for item in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, item))]

    print(f"\n{'='*80}")
    print(f"DEBUG: Task folders found")
    print(f"{'='*80}")
    print(f"Number of task folders: {len(task_folders)}")
    print(f"Task folders: {sorted(task_folders)[:20]}")  # Show first 20

    # Debug first task in detail
    if task_folders:
        sample_task = sorted(task_folders)[0]
        print(f"\n{'='*80}")
        print(f"DEBUG: Sample task detailed check - {sample_task}")
        print(f"{'='*80}")

        print(f"Checking CoT importance at step {training_steps[0]}...")
        cot_val = load_cot_importance_at_step(base_dir, sample_task, training_steps[0])
        print(f"Result: {cot_val}")

        print(f"\nChecking verifier accuracy at step {training_steps[0]}...")
        verifier_val = load_verifier_accuracy_at_step(base_dir, sample_task, training_steps[0])
        print(f"Result: {verifier_val}")

    # Collect CoT importance data
    cot_task_data = {}
    tasks_with_cot_data = 0

    for task_name in sorted(task_folders):
        # Special case: simple_geometry uses steps 40 and 100 instead of 30 and 90
        if task_name == 'simple_geometry':
            task_training_steps = [2, 40, 60, 100, 120, 156]
        else:
            task_training_steps = training_steps

        step_values = {}
        for step in task_training_steps:
            cot_importance = load_cot_importance_at_step(base_dir, task_name, step)
            if cot_importance is not None:
                step_values[step] = cot_importance

        if step_values:
            tasks_with_cot_data += 1

        # Only include tasks that have data for at least first or last step
        if any(s in step_values for s in training_steps):
            cot_task_data[task_name] = {
                'steps': step_values,
                'training_steps': task_training_steps
            }

    print(f"\n{'='*80}")
    print(f"DEBUG: CoT importance data collection")
    print(f"{'='*80}")
    print(f"Tasks with CoT data: {tasks_with_cot_data}")
    print(f"Tasks included in cot_task_data: {len(cot_task_data)}")
    if cot_task_data:
        print(f"Sample tasks: {list(cot_task_data.keys())[:10]}")

    # Collect Verifier Accuracy data
    verifier_task_data = {}
    tasks_with_verifier_data = 0

    for task_name in sorted(task_folders):
        # Load verifier accuracy for all steps
        step_values = {}
        for step in training_steps:
            verifier_acc = load_verifier_accuracy_at_step(base_dir, task_name, step)
            if verifier_acc is not None:
                step_values[step] = verifier_acc

        if step_values:
            tasks_with_verifier_data += 1

        # Only include tasks that have data for at least first or last step
        if any(s in step_values for s in training_steps):
            verifier_task_data[task_name] = {
                'steps': step_values
            }

    print(f"\n{'='*80}")
    print(f"DEBUG: Verifier accuracy data collection")
    print(f"{'='*80}")
    print(f"Tasks with verifier data: {tasks_with_verifier_data}")
    print(f"Tasks included in verifier_task_data: {len(verifier_task_data)}")
    if verifier_task_data:
        print(f"Sample tasks: {list(verifier_task_data.keys())[:10]}")

    if not cot_task_data and not verifier_task_data:
        print(f"\nERROR: No data found for either CoT or Verifier")
        return

    # Sort tasks by initial CoT importance (lowest to highest)
    def get_initial_cot(item):
        task_name, data = item
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 1:
            return steps[available_steps[0]]
        return 0

    sorted_cot_tasks = sorted(cot_task_data.items(), key=get_initial_cot)
    cot_task_names = [task for task, _ in sorted_cot_tasks]

    # Sort tasks by initial verifier accuracy (lowest to highest)
    def get_initial_verifier(item):
        task_name, data = item
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 1:
            return steps[available_steps[0]]
        return 0

    sorted_verifier_tasks = sorted(verifier_task_data.items(), key=get_initial_verifier)
    verifier_task_names = [task for task, _ in sorted_verifier_tasks]

    # Analyze CIR changes for bottom 1/3, middle 1/3, top 1/3
    print("\n" + "="*80)
    print("CIR ANALYSIS: Bottom 1/3 vs Middle 1/3 vs Top 1/3 (sorted by initial CIR)")
    print("="*80)

    # Calculate overall increase/decrease percentages for CIR
    cir_tasks_increased = 0
    cir_tasks_decreased = 0
    cir_tasks_unchanged = 0
    for task_name, data in sorted_cot_tasks:
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            initial = steps[available_steps[0]]
            final = steps[available_steps[-1]]
            if final > initial:
                cir_tasks_increased += 1
            elif final < initial:
                cir_tasks_decreased += 1
            else:
                cir_tasks_unchanged += 1

    # Calculate percentage of tasks with CIR < 0.1 and SR < 0.1 at final step
    final_step = training_steps[-1]
    cir_below_threshold = sum(1 for _, data in sorted_cot_tasks
                              if final_step in data['steps'] and data['steps'][final_step] < 0.1)
    sr_below_threshold = sum(1 for _, data in sorted_verifier_tasks
                             if final_step in data['steps'] and data['steps'][final_step] < 0.1)
    total_cir_final = sum(1 for _, data in sorted_cot_tasks if final_step in data['steps'])
    total_sr_final = sum(1 for _, data in sorted_verifier_tasks if final_step in data['steps'])

    print(f"\n{'='*80}")
    print(f"CIR AND SR PER TASK AT STEP {final_step}")
    print(f"{'='*80}")
    all_tasks = sorted(set(list(cot_task_data.keys()) + list(verifier_task_data.keys())))
    print(f"{'Task':<35} {'CIR':>8} {'SR':>8}")
    print("-" * 55)
    for task_name in all_tasks:
        cir_val = cot_task_data.get(task_name, {}).get('steps', {}).get(final_step, float('nan'))
        sr_val = verifier_task_data.get(task_name, {}).get('steps', {}).get(final_step, float('nan'))
        print(f"{task_name:<35} {cir_val:>8.4f} {sr_val:>8.4f}")

    print(f"\n{'='*80}")
    print(f"LOW CIR / LOW SR AT STEP {final_step} (threshold < 0.1)")
    print(f"{'='*80}")
    if total_cir_final > 0:
        print(f"  CIR < 0.1: {cir_below_threshold}/{total_cir_final} tasks ({100*cir_below_threshold/total_cir_final:.1f}%)")
    if total_sr_final > 0:
        print(f"  SR  < 0.1: {sr_below_threshold}/{total_sr_final} tasks ({100*sr_below_threshold/total_sr_final:.1f}%)")

    total_cir_tasks = len(sorted_cot_tasks)
    print(f"\nOverall CIR trends:")
    print(f"  Total tasks: {total_cir_tasks}")
    print(f"  Tasks with CIR increase: {cir_tasks_increased}/{total_cir_tasks} ({100*cir_tasks_increased/total_cir_tasks:.1f}%)")
    print(f"  Tasks with CIR decrease: {cir_tasks_decreased}/{total_cir_tasks} ({100*cir_tasks_decreased/total_cir_tasks:.1f}%)")
    if cir_tasks_unchanged > 0:
        print(f"  Tasks with CIR unchanged: {cir_tasks_unchanged}/{total_cir_tasks} ({100*cir_tasks_unchanged/total_cir_tasks:.1f}%)")

    # Analyze CIR changes by math vs non-math tasks
    print(f"\nCIR trends by task type:")
    cir_math_increased = 0
    cir_math_total = 0
    cir_nonmath_increased = 0
    cir_nonmath_total = 0

    for task_name, data in sorted_cot_tasks:
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            initial = steps[available_steps[0]]
            final = steps[available_steps[-1]]

            if is_math_task(task_name, task_to_category):
                cir_math_total += 1
                if final > initial:
                    cir_math_increased += 1
            else:
                cir_nonmath_total += 1
                if final > initial:
                    cir_nonmath_increased += 1

    if cir_math_total > 0:
        print(f"  Math tasks with CIR increase: {cir_math_increased}/{cir_math_total} ({100*cir_math_increased/cir_math_total:.1f}%)")
    else:
        print(f"  Math tasks with CIR increase: N/A (no math tasks)")

    if cir_nonmath_total > 0:
        print(f"  Non-math tasks with CIR increase: {cir_nonmath_increased}/{cir_nonmath_total} ({100*cir_nonmath_increased/cir_nonmath_total:.1f}%)")
    else:
        print(f"  Non-math tasks with CIR increase: N/A (no non-math tasks)")

    third_cot = len(sorted_cot_tasks) // 3
    bottom_third_cot = sorted_cot_tasks[:third_cot]
    middle_third_cot = sorted_cot_tasks[third_cot:2*third_cot]
    top_third_cot = sorted_cot_tasks[2*third_cot:]

    # Bottom 1/3 CIR analysis
    bottom_cot_increases = []
    for task_name, data in bottom_third_cot:
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            initial = steps[available_steps[0]]
            final = steps[available_steps[-1]]
            change = final - initial
            if change > 0:
                bottom_cot_increases.append(change)

    print(f"\nBottom 1/3 (n={len(bottom_third_cot)} tasks):")
    print(f"  Tasks with CIR increase: {len(bottom_cot_increases)}/{len(bottom_third_cot)}")
    if bottom_cot_increases:
        print(f"  Average increase: {np.mean(bottom_cot_increases):.4f}")
    else:
        print(f"  Average increase: N/A (no increases)")

    # Middle 1/3 CIR analysis
    middle_cot_increases = []
    for task_name, data in middle_third_cot:
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            initial = steps[available_steps[0]]
            final = steps[available_steps[-1]]
            change = final - initial
            if change > 0:
                middle_cot_increases.append(change)

    print(f"\nMiddle 1/3 (n={len(middle_third_cot)} tasks):")
    print(f"  Tasks with CIR increase: {len(middle_cot_increases)}/{len(middle_third_cot)}")
    if middle_cot_increases:
        print(f"  Average increase: {np.mean(middle_cot_increases):.4f}")
    else:
        print(f"  Average increase: N/A (no increases)")

    # Top 1/3 CIR analysis
    top_cot_increases = []
    for task_name, data in top_third_cot:
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            initial = steps[available_steps[0]]
            final = steps[available_steps[-1]]
            change = final - initial
            if change > 0:
                top_cot_increases.append(change)

    print(f"\nTop 1/3 (n={len(top_third_cot)} tasks):")
    print(f"  Tasks with CIR increase: {len(top_cot_increases)}/{len(top_third_cot)}")
    if top_cot_increases:
        print(f"  Average increase: {np.mean(top_cot_increases):.4f}")
    else:
        print(f"  Average increase: N/A (no increases)")

    # Analyze SR changes for bottom 1/3, middle 1/3, top 1/3
    print("\n" + "="*80)
    print("SR ANALYSIS: Bottom 1/3 vs Middle 1/3 vs Top 1/3 (sorted by initial SR)")
    print("="*80)

    # Calculate overall increase/decrease percentages for SR
    sr_tasks_increased = 0
    sr_tasks_decreased = 0
    sr_tasks_unchanged = 0
    for task_name, data in sorted_verifier_tasks:
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            initial = steps[available_steps[0]]
            final = steps[available_steps[-1]]
            if final > initial:
                sr_tasks_increased += 1
            elif final < initial:
                sr_tasks_decreased += 1
            else:
                sr_tasks_unchanged += 1

    total_sr_tasks = len(sorted_verifier_tasks)
    print(f"\nOverall SR trends:")
    print(f"  Total tasks: {total_sr_tasks}")
    print(f"  Tasks with SR increase: {sr_tasks_increased}/{total_sr_tasks} ({100*sr_tasks_increased/total_sr_tasks:.1f}%)")
    print(f"  Tasks with SR decrease: {sr_tasks_decreased}/{total_sr_tasks} ({100*sr_tasks_decreased/total_sr_tasks:.1f}%)")
    if sr_tasks_unchanged > 0:
        print(f"  Tasks with SR unchanged: {sr_tasks_unchanged}/{total_sr_tasks} ({100*sr_tasks_unchanged/total_sr_tasks:.1f}%)")

    # Analyze SR changes by math vs non-math tasks
    print(f"\nSR trends by task type:")
    sr_math_increased = 0
    sr_math_total = 0
    sr_nonmath_increased = 0
    sr_nonmath_total = 0

    for task_name, data in sorted_verifier_tasks:
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            initial = steps[available_steps[0]]
            final = steps[available_steps[-1]]

            if is_math_task(task_name, task_to_category):
                sr_math_total += 1
                if final > initial:
                    sr_math_increased += 1
            else:
                sr_nonmath_total += 1
                if final > initial:
                    sr_nonmath_increased += 1

    if sr_math_total > 0:
        print(f"  Math tasks with SR increase: {sr_math_increased}/{sr_math_total} ({100*sr_math_increased/sr_math_total:.1f}%)")
    else:
        print(f"  Math tasks with SR increase: N/A (no math tasks)")

    if sr_nonmath_total > 0:
        print(f"  Non-math tasks with SR increase: {sr_nonmath_increased}/{sr_nonmath_total} ({100*sr_nonmath_increased/sr_nonmath_total:.1f}%)")
    else:
        print(f"  Non-math tasks with SR increase: N/A (no non-math tasks)")

    third_verifier = len(sorted_verifier_tasks) // 3
    bottom_third_verifier = sorted_verifier_tasks[:third_verifier]
    middle_third_verifier = sorted_verifier_tasks[third_verifier:2*third_verifier]
    top_third_verifier = sorted_verifier_tasks[2*third_verifier:]

    # Bottom 1/3 SR analysis
    bottom_verifier_increases = []
    for task_name, data in bottom_third_verifier:
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            initial = steps[available_steps[0]]
            final = steps[available_steps[-1]]
            change = final - initial
            if change > 0:
                bottom_verifier_increases.append(change)

    print(f"\nBottom 1/3 (n={len(bottom_third_verifier)} tasks):")
    print(f"  Tasks with SR increase: {len(bottom_verifier_increases)}/{len(bottom_third_verifier)}")
    if bottom_verifier_increases:
        print(f"  Average increase: {np.mean(bottom_verifier_increases):.4f}")
    else:
        print(f"  Average increase: N/A (no increases)")

    # Middle 1/3 SR analysis
    middle_verifier_increases = []
    for task_name, data in middle_third_verifier:
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            initial = steps[available_steps[0]]
            final = steps[available_steps[-1]]
            change = final - initial
            if change > 0:
                middle_verifier_increases.append(change)

    print(f"\nMiddle 1/3 (n={len(middle_third_verifier)} tasks):")
    print(f"  Tasks with SR increase: {len(middle_verifier_increases)}/{len(middle_third_verifier)}")
    if middle_verifier_increases:
        print(f"  Average increase: {np.mean(middle_verifier_increases):.4f}")
    else:
        print(f"  Average increase: N/A (no increases)")

    # Top 1/3 SR analysis
    top_verifier_increases = []
    for task_name, data in top_third_verifier:
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            initial = steps[available_steps[0]]
            final = steps[available_steps[-1]]
            change = final - initial
            if change > 0:
                top_verifier_increases.append(change)

    print(f"\nTop 1/3 (n={len(top_third_verifier)} tasks):")
    print(f"  Tasks with SR increase: {len(top_verifier_increases)}/{len(top_third_verifier)}")
    if top_verifier_increases:
        print(f"  Average increase: {np.mean(top_verifier_increases):.4f}")
    else:
        print(f"  Average increase: N/A (no increases)")

    print("\n" + "="*80)
    print(f"DEBUG: Starting plotting")
    print("="*80)
    print(f"CoT tasks to plot: {len(cot_task_names)}")
    print(f"Verifier tasks to plot: {len(verifier_task_names)}")
    print("="*80 + "\n")

    # Create the combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 10))

    # Colors for before and after training
    step_colors = {
        training_steps[0]: (0.678, 0.847, 0.902),    # Light blue for "Before Training"
        training_steps[-1]: (0.098, 0.275, 0.471),  # Dark blue for "After Training"
    }

    # LEFT SUBPLOT: CoT Importance
    x_positions_cot = np.arange(len(cot_task_names))

    # Add legend entries for both steps (dummy points outside plot)
    ax1.scatter([], [], s=80, color=step_colors[training_steps[0]], edgecolors='black', linewidth=1.5, label='Before Training')
    ax1.scatter([], [], s=80, color=step_colors[training_steps[-1]], edgecolors='black', linewidth=1.5, label='After Training')

    for i, (task, data) in enumerate(zip(cot_task_names, [d for _, d in sorted_cot_tasks])):
        step_values = data['steps']
        task_training_steps = data['training_steps']

        # Get all available step values for this task (only steps 2 and 156)
        available_steps = []
        y_values = []
        colors = []
        for step in training_steps:
            if step in step_values:
                available_steps.append(step)
                y_values.append(step_values[step])
                colors.append(step_colors[step])

        # Determine if increasing or decreasing
        if len(available_steps) > 1:
            first_val = y_values[0]
            last_val = y_values[-1]
            is_increasing = last_val > first_val
            line_color = '#2E86AB' if is_increasing else '#D32F2F'  # Blue for increase, red for decrease

            # Draw line with multiple chevron arrows
            num_chevrons = 5
            for idx in range(num_chevrons):
                start_frac = idx / num_chevrons
                end_frac = (idx + 1) / num_chevrons

                y_start = y_values[0] + start_frac * (y_values[-1] - y_values[0])
                y_end = y_values[0] + end_frac * (y_values[-1] - y_values[0])

                # Draw chevron arrow segment
                ax1.annotate('', xy=(i, y_end), xytext=(i, y_start),
                           arrowprops=dict(arrowstyle='->', color=line_color, lw=2.5, alpha=0.7,
                                         mutation_scale=15), zorder=2)

        # Plot dots at each available step with step colors
        for step, y_val, color in zip(available_steps, y_values, colors):
            ax1.scatter(i, y_val, s=80, color=color,
                       edgecolors='black', linewidth=1.5, zorder=3)

    # Formatting for CoT subplot
    ax1.set_xticks(x_positions_cot)
    ax1.set_xticklabels(cot_task_names, rotation=70, ha='right', fontsize=19, family='serif')

    # Make math task names bold
    for i, (label, task_name) in enumerate(zip(ax1.get_xticklabels(), cot_task_names)):
        if is_math_task(task_name, task_to_category):
            label.set_fontweight('bold')
    ax1.set_xlabel('Tasks (sorted by initial CIR: lowest → highest)',
                  fontsize=35, fontweight='bold', family='serif')
    ax1.set_ylabel('CIR',
                  fontsize=35, fontweight='bold', family='serif')
    ax1.set_title(f'Evolution of CIR: Before vs After Training',
                 fontsize=35, fontweight='bold', pad=20, family='serif')

    for label in ax1.get_yticklabels():
        label.set_family('serif')
        label.set_fontsize(19)

    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y', zorder=0)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), framealpha=0.9, prop={'family': 'serif', 'size': 24}, markerscale=2)

    # RIGHT SUBPLOT: Verifier Accuracy
    x_positions_verifier = np.arange(len(verifier_task_names))

    # Add legend entries for both steps (dummy points outside plot)
    ax2.scatter([], [], s=80, color=step_colors[training_steps[0]], edgecolors='black', linewidth=1.5, label='Before Training')
    ax2.scatter([], [], s=80, color=step_colors[training_steps[-1]], edgecolors='black', linewidth=1.5, label='After Training')

    for i, (task, data) in enumerate(zip(verifier_task_names, [d for _, d in sorted_verifier_tasks])):
        step_values = data['steps']

        # Get all available step values for this task (only steps 2 and 156)
        available_steps = []
        y_values = []
        colors = []
        for step in training_steps:
            if step in step_values:
                available_steps.append(step)
                y_values.append(step_values[step])
                colors.append(step_colors[step])

        # Determine if increasing or decreasing
        if len(available_steps) > 1:
            first_val = y_values[0]
            last_val = y_values[-1]
            is_increasing = last_val > first_val
            line_color = '#2E86AB' if is_increasing else '#D32F2F'

            # Draw line with multiple chevron arrows
            num_chevrons = 5
            for idx in range(num_chevrons):
                start_frac = idx / num_chevrons
                end_frac = (idx + 1) / num_chevrons

                y_start = y_values[0] + start_frac * (y_values[-1] - y_values[0])
                y_end = y_values[0] + end_frac * (y_values[-1] - y_values[0])

                ax2.annotate('', xy=(i, y_end), xytext=(i, y_start),
                           arrowprops=dict(arrowstyle='->', color=line_color, lw=2.5, alpha=0.7,
                                         mutation_scale=15), zorder=2)

        # Plot dots at each available step with step colors
        for step, y_val, color in zip(available_steps, y_values, colors):
            ax2.scatter(i, y_val, s=80, color=color,
                       edgecolors='black', linewidth=1.5, zorder=3)

    # Formatting for Verifier subplot
    ax2.set_xticks(x_positions_verifier)
    ax2.set_xticklabels(verifier_task_names, rotation=70, ha='right', fontsize=19, family='serif')

    # Make math task names bold
    for i, (label, task_name) in enumerate(zip(ax2.get_xticklabels(), verifier_task_names)):
        if is_math_task(task_name, task_to_category):
            label.set_fontweight('bold')
    ax2.set_xlabel('Tasks (sorted by initial SR: lowest → highest)',
                  fontsize=35, fontweight='bold', family='serif')
    ax2.set_ylabel('SR',
                  fontsize=35, fontweight='bold', family='serif')
    ax2.set_title(f'Evolution of SR: Before vs After Training',
                 fontsize=35, fontweight='bold', pad=20, family='serif')

    for label in ax2.get_yticklabels():
        label.set_family('serif')
        label.set_fontsize(19)

    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y', zorder=0)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), framealpha=0.9, prop={'family': 'serif', 'size': 24}, markerscale=2)

    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f'plot_3_combined_{args.model_size}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"DEBUG: Plot saved successfully")
    print(f"{'='*80}")
    print(f"PDF saved to: {pdf_path}")

    png_path = os.path.join(output_dir, f'plot_3_combined_{args.model_size}_match_original.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"PNG saved to: {png_path}")
    print(f"{'='*80}\n")

    plt.show()


if __name__ == "__main__":
    main()
