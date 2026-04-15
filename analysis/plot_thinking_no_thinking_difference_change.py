#!/usr/bin/env python3
"""
Plot initial and final Thinking-NoThinking accuracy difference for each task as a scatter plot.
Shows how the performance gap between Thinking and NoThinking evolves from beginning to end.

Usage:
    python plot_thinking_no_thinking_difference_change.py
    python plot_thinking_no_thinking_difference_change.py --local  # for local testing
    python plot_thinking_no_thinking_difference_change.py --begin_step 1 --end_step 156
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Special tasks that load from evaluate results instead of rollout data
SPECIAL_TASKS = ['futoshiki', 'mini_sudoku', 'simple_equations']


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


def load_step_accuracy(base_dir, task_name, step, variant="original", batch="8", from_evaluate=False, model_size="3b", local=False):
    """Load accuracy for a specific step."""
    if from_evaluate:
        # Load from evaluate results for special tasks
        if local:
            eval_base = f'/Users/qinanyu/Desktop/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct/cot_importance'
        else:
            eval_base = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct/cot_importance'

        grpo_task = f"{task_name}_cot_importance"
        grpo_path = os.path.join(eval_base, grpo_task, "-1.0", "teacher", f"step_{step}", f"teacher_responses_step_{step}.json")
        if os.path.exists(grpo_path):
            try:
                with open(grpo_path, 'r') as f:
                    data = json.load(f)
                scores = []
                for item in data:
                    if 'k_responses' in item and item['k_responses']:
                        best = max(item['k_responses'], key=lambda x: x.get('reward_score', 0))
                        scores.append(best.get('reward_score', 0))
                    elif 'reward_score' in item:
                        scores.append(item['reward_score'])
                return np.mean(scores) if scores else None
            except:
                return None
        return None
    else:
        # Load from rollout data
        rollout_dir = Path(base_dir) / task_name / variant / variant / batch
        if not rollout_dir.exists():
            return None
        step_file = rollout_dir / f"{step}.jsonl"
        if not step_file.exists():
            return None
        try:
            entries = load_jsonl_file(step_file)
            return calculate_average_accuracy(entries)
        except:
            return None


def load_verifier_accuracy_at_step(base_dir, task_name, step, model_size="3b", local=False):
    """Load average verifier accuracy from verifier_comparison.answers_match."""
    if local:
        eval_base = f'/Users/qinanyu/Desktop/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct/cot_importance'
    else:
        eval_base = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct/cot_importance'

    possible_paths = [
        os.path.join(eval_base, task_name, "-1.0", "teacher", f"step_{step}", f"teacher_responses_step_{step}.json"),
        os.path.join(eval_base, task_name, "1.0", "teacher", f"step_{step}", f"teacher_responses_step_{step}.json"),
        os.path.join(eval_base, f"{task_name}_cot_importance", "-1.0", "teacher", f"step_{step}", f"teacher_responses_step_{step}.json"),
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

            with_q_answer = verifier_comp.get('with_question_answer', '').strip().lower()
            without_q_answer = verifier_comp.get('without_question_answer', '').strip().lower()

            if answers_match and with_q_answer == 'no answer found' and without_q_answer == 'no answer found':
                verifier_scores.append(0)
            else:
                verifier_scores.append(1 if answers_match else 0)

    if verifier_scores:
        return np.mean(verifier_scores)
    else:
        return None


def find_common_tasks(thinking_base, no_thinking_base, training_steps, variant="original", batch="8"):
    """Find tasks that exist in both Thinking and NoThinking directories with required steps."""
    thinking_tasks = set()
    no_thinking_tasks = set()

    thinking_path = Path(thinking_base)
    no_thinking_path = Path(no_thinking_base)

    if thinking_path.exists():
        for task_dir in thinking_path.iterdir():
            if task_dir.is_dir():
                rollout_dir = task_dir / variant / variant / batch
                if rollout_dir.exists():
                    # Check if at least first and last step exist
                    if (rollout_dir / f"{training_steps[0]}.jsonl").exists() or (rollout_dir / f"{training_steps[-1]}.jsonl").exists():
                        thinking_tasks.add(task_dir.name)

    if no_thinking_path.exists():
        for task_dir in no_thinking_path.iterdir():
            if task_dir.is_dir():
                rollout_dir = task_dir / variant / variant / batch
                if rollout_dir.exists():
                    # Check if at least first and last step exist
                    if (rollout_dir / f"{training_steps[0]}.jsonl").exists() or (rollout_dir / f"{training_steps[-1]}.jsonl").exists():
                        no_thinking_tasks.add(task_dir.name)

    return sorted(thinking_tasks & no_thinking_tasks)


def main():
    parser = argparse.ArgumentParser(description='Plot Thinking-NoThinking difference change across training steps')
    parser.add_argument('--local', action='store_true', help='Use local paths (for testing)')
    parser.add_argument('--model', type=str, default='q3b-instruct', help='Model name (default: q3b-instruct)')
    parser.add_argument('--model_size', type=str, default='3b', help='Model size for evaluate results (default: 3b)')
    parser.add_argument('--thinking_dir', type=str, default='grpo', help='Directory name for thinking variant')
    parser.add_argument('--no_thinking_dir', type=str, default='direct', help='Directory name for no-thinking variant')
    parser.add_argument('--variant', type=str, default='original', help='Variant folder name (default: original)')
    parser.add_argument('--batch', type=str, default='8', help='Batch folder name (default: 8)')
    parser.add_argument('--analyze_cir', action='store_true', help='Analyze thinking gap by CIR change')

    args = parser.parse_args()

    # Define training steps (only initial and final)
    training_steps = [2, 156]

    # Set base directories
    if args.local:
        thinking_base = f'/Users/qinanyu/Desktop/rl-explanations/trainers/{args.thinking_dir}/{args.model}/rollout_data'
        no_thinking_base = f'/Users/qinanyu/Desktop/rl-explanations/trainers/{args.no_thinking_dir}/{args.model}/rollout_data'
        output_dir = '/Users/qinanyu/Desktop/rl-explanations/analysis'
    else:
        thinking_base = f'/nlp/scr/qinanyu/rl-explanations/trainers/{args.thinking_dir}/{args.model}/rollout_data'
        no_thinking_base = f'/nlp/scr/qinanyu/rl-explanations/trainers/{args.no_thinking_dir}/{args.model}/rollout_data'
        output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis'

    # Find common tasks
    tasks = find_common_tasks(thinking_base, no_thinking_base, training_steps, args.variant, args.batch)
    if not tasks:
        return

    # Load data for all tasks
    task_data = {}
    for task_name in tasks:
        # Check if this is a special task that needs evaluate results
        is_special = task_name in SPECIAL_TASKS

        # Load accuracy differences for all steps
        step_diffs = {}
        thinking_end_acc = None

        for step in training_steps:
            # For thinking (GRPO), use from_evaluate for special tasks
            thinking_acc = load_step_accuracy(thinking_base, task_name, step, args.variant, args.batch,
                                             from_evaluate=is_special, model_size=args.model_size, local=args.local)
            # For no_thinking (Direct), always use rollout data
            no_thinking_acc = load_step_accuracy(no_thinking_base, task_name, step, args.variant, args.batch,
                                                from_evaluate=False, model_size=args.model_size, local=args.local)

            if thinking_acc is not None and no_thinking_acc is not None:
                step_diffs[step] = thinking_acc - no_thinking_acc
                if step == 156:
                    thinking_end_acc = thinking_acc

        # Only include tasks that have data for at least first and last step
        if (2 in step_diffs or 156 in step_diffs or 150 in step_diffs) and thinking_end_acc is not None:
            task_data[task_name] = {
                'steps': step_diffs,
                'thinking_end': thinking_end_acc
            }

    if not task_data:
        return

    # Load CIR and SR data for final step to determine coloring
    # Red: CIR < 0.15 AND SR < 0.15
    # Gray: Everything else
    task_colors = {}
    cir_base = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance' if not args.local else f'/Users/qinanyu/Desktop/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'

    for task_name in task_data.keys():
        # Load final values (step 156)
        final_cir = load_cot_importance_at_step(cir_base, task_name, 156)
        final_sr = load_verifier_accuracy_at_step(None, task_name, 156, args.model_size, args.local)

        # Determine color: Red only if BOTH CIR < 0.15 AND SR < 0.15
        if final_cir is not None and final_sr is not None:
            if final_cir < 0.15 and final_sr < 0.15:
                task_colors[task_name] = '#E57373'  # Red
                task_data[task_name]['category'] = 'red'
            else:
                task_colors[task_name] = '#808080'  # Gray
                task_data[task_name]['category'] = 'gray'
        else:
            # Default to gray if data missing
            task_colors[task_name] = '#808080'
            task_data[task_name]['category'] = 'gray'

    # Sort tasks by final thinking accuracy
    sorted_tasks = sorted(task_data.items(), key=lambda x: x[1]['thinking_end'], reverse=True)
    task_names = [task for task, _ in sorted_tasks]

    # Print all tasks being analyzed
    print("\n" + "="*80)
    print("ALL TASKS BEING ANALYZED")
    print("="*80)
    print(f"Total tasks: {len(task_names)}")
    print(f"Tasks (sorted by final thinking accuracy):")
    for idx, task in enumerate(task_names, 1):
        print(f"  {idx}. {task}")
    print("="*80)

    # Analyze tasks with significant thinking advantage
    print("\n" + "="*80)
    print("THINKING ADVANTAGE ANALYSIS")
    print("="*80)

    # Calculate initial and final thinking advantages
    initial_advantages = []
    final_advantages = []
    tasks_with_significant_gain = []
    tasks_with_positive_final = []
    tasks_with_negative_final = []

    for task_name, data in sorted_tasks:
        step_diffs = data['steps']
        available_steps = sorted(step_diffs.keys())

        if len(available_steps) >= 1:
            initial_diff = step_diffs[available_steps[0]]
            final_diff = step_diffs[available_steps[-1]]

            initial_advantages.append(initial_diff)
            final_advantages.append(final_diff)

            if final_diff > 0.1:
                tasks_with_significant_gain.append((task_name, final_diff))

            if final_diff > 0:
                tasks_with_positive_final.append((task_name, final_diff))
            else:
                tasks_with_negative_final.append((task_name, final_diff))

    print(f"\nTotal tasks analyzed: {len(task_data)}")
    print(f"\nTasks with significant thinking advantage (final > 0.1): {len(tasks_with_significant_gain)}/{len(task_data)}")
    if tasks_with_significant_gain:
        print(f"Average advantage for significant tasks: {np.mean([adv for _, adv in tasks_with_significant_gain]):.4f}")
        print(f"Tasks:")
        for task, adv in sorted(tasks_with_significant_gain, key=lambda x: x[1], reverse=True):
            print(f"  - {task}: {adv:.4f}")

    print(f"\nTasks with positive final thinking advantage (final > 0): {len(tasks_with_positive_final)}/{len(task_data)}")
    if tasks_with_positive_final:
        print(f"Average advantage: {np.mean([adv for _, adv in tasks_with_positive_final]):.4f}")

    print(f"\nTasks with negative final thinking advantage (final < 0): {len(tasks_with_negative_final)}/{len(task_data)}")
    if tasks_with_negative_final:
        print(f"Average disadvantage: {np.mean([adv for _, adv in tasks_with_negative_final]):.4f}")
        print(f"Tasks:")
        for task, adv in sorted(tasks_with_negative_final, key=lambda x: x[1]):
            print(f"  - {task}: {adv:.4f}")

    if initial_advantages and final_advantages:
        print(f"\nAverage initial thinking advantage: {np.mean(initial_advantages):.4f}")
        print(f"Average final thinking advantage: {np.mean(final_advantages):.4f}")
        print(f"Average change: {np.mean(final_advantages) - np.mean(initial_advantages):.4f}")

    # Calculate proportion of tasks where no-CoT performs as well as CoT
    tasks_no_cot_as_good = len([adv for adv in final_advantages if adv <= 0.05])
    proportion = tasks_no_cot_as_good / len(task_data) if len(task_data) > 0 else 0
    print(f"\nTo validate this, we run RLVR with and without CoT reasoning in models, and we find that in {proportion:.1%} ({tasks_no_cot_as_good}/{len(task_data)}) of tasks, models without CoT perform as well as models with CoT.")

    print("="*80 + "\n")

    # Analyze by CIR change if requested
    if args.analyze_cir:
        print("\n" + "="*80)
        print("ANALYSIS BY CIR CHANGE")
        print("="*80)

        # Set CIR base directory
        if args.local:
            cir_base_dir = f'/Users/qinanyu/Desktop/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'
        else:
            cir_base_dir = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'

        # Load CIR data for tasks
        tasks_with_cir_increase = []
        tasks_with_cir_decrease = []
        tasks_missing_cir = []

        for task_name, data in sorted_tasks:
            step_diffs = data['steps']
            available_steps = sorted(step_diffs.keys())

            if len(available_steps) >= 1:
                initial_step = available_steps[0]
                final_step = available_steps[-1]

                # Load CIR values
                initial_cir = load_cot_importance_at_step(cir_base_dir, task_name, initial_step)
                final_cir = load_cot_importance_at_step(cir_base_dir, task_name, final_step)

                if initial_cir is not None and final_cir is not None:
                    cir_change = final_cir - initial_cir
                    final_thinking_gap = step_diffs[final_step]

                    if cir_change > 0:
                        tasks_with_cir_increase.append({
                            'task': task_name,
                            'cir_change': cir_change,
                            'thinking_gap': final_thinking_gap
                        })
                    else:
                        tasks_with_cir_decrease.append({
                            'task': task_name,
                            'cir_change': cir_change,
                            'thinking_gap': final_thinking_gap
                        })
                else:
                    tasks_missing_cir.append(task_name)

        if tasks_with_cir_increase:
            thinking_gaps_increase = [t['thinking_gap'] for t in tasks_with_cir_increase]
            print(f"\nCIR INCREASE: {np.mean(thinking_gaps_increase):.4f}")

        if tasks_with_cir_decrease:
            thinking_gaps_decrease = [t['thinking_gap'] for t in tasks_with_cir_decrease]
            print(f"CIR DECREASE: {np.mean(thinking_gaps_decrease):.4f}\n")

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot lines and dots for each task
    x_positions = np.arange(len(task_names))

    # Track if we've added legend entries
    added_legend = {'red': False, 'gray': False}

    # Track values for each category to calculate averages
    red_values = []
    gray_values = []

    for i, (task, data) in enumerate(zip(task_names, [d for _, d in sorted_tasks])):
        step_diffs = data['steps']
        task_color = task_colors[task]
        category = data.get('category', 'unknown')

        # Get final value (Step 156)
        if 156 in step_diffs:
            final_val = step_diffs[156]

            # Track values by category
            if category == 'red':
                red_values.append(final_val)
            elif category == 'gray':
                gray_values.append(final_val)

            # Plot final dot only
            label = None
            if category == 'red' and not added_legend['red']:
                label = 'CIR < 0.15 & SR < 0.15'
                added_legend['red'] = True
            elif category == 'gray' and not added_legend['gray']:
                label = 'Others'
                added_legend['gray'] = True

            ax.scatter(i, final_val, s=200, color=task_color,
                       edgecolors='black', linewidth=1.5, zorder=3,
                       label=label)

    # Calculate and print averages for red and gray categories
    print("\n" + "="*80)
    print("CATEGORY-WISE ANALYSIS")
    print("="*80)
    if red_values:
        print(f"\nRed dots (CIR < 0.15 & SR < 0.15):")
        print(f"  Count: {len(red_values)}")
        print(f"  Average final thinking advantage: {np.mean(red_values):.4f}")
        print(f"  Std dev: {np.std(red_values):.4f}")
    else:
        print(f"\nRed dots (CIR < 0.15 & SR < 0.15): 0 tasks")

    if gray_values:
        print(f"\nGray dots (Others):")
        print(f"  Count: {len(gray_values)}")
        print(f"  Average final thinking advantage: {np.mean(gray_values):.4f}")
        print(f"  Std dev: {np.std(gray_values):.4f}")
    else:
        print(f"\nGray dots (Others): 0 tasks")

    if red_values and gray_values:
        diff = np.mean(gray_values) - np.mean(red_values)
        print(f"\nDifference (Gray avg - Red avg): {diff:.4f}")

    print("="*80 + "\n")

    # Add horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=1)

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(task_names, rotation=70, ha='right', fontsize=19, family='serif')
    ax.set_xlabel('Tasks (sorted by final accuracy with Thinking)',
                  fontsize=25, fontweight='bold', family='serif')
    ax.set_ylabel('Final Acc Diff',
                  fontsize=35, fontweight='bold', family='serif')

    ax.set_title(f'Difference between training to reason\nand training to directly answer',
                 fontsize=40, fontweight='bold', pad=20, family='serif')

    # Set y-axis tick labels to serif font
    for label in ax.get_yticklabels():
        label.set_family('serif')
        label.set_fontsize(19)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y', zorder=0)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.9, prop={'family': 'serif', 'size': 24}, markerscale=2)

    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'figure1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, 'figure1.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
