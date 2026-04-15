#!/usr/bin/env python3
"""
Analyze the relationship between CIR decrease and thinking-no thinking performance gap.

For all tasks where CIR decreases from initial to final, calculate the final thinking-no thinking
performance difference.

Usage:
    python analyze_cir_decrease_thinking_gap.py
    python analyze_cir_decrease_thinking_gap.py --model_size 3b
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path


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


def load_step_accuracy_from_rollout(base_dir, task_name, step, variant="original", batch="8"):
    """Load accuracy for a specific step from rollout data."""
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


def main():
    parser = argparse.ArgumentParser(description='Analyze CIR decrease vs thinking-no thinking gap')
    parser.add_argument('--model_size', type=str, default='3b', help='Model size (default: 3b)')
    args = parser.parse_args()

    # Paths
    cot_base_dir = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'
    rollout_base_grpo = '/nlp/scr/qinanyu/rl-explanations/trainers/grpo/q3b-instruct/rollout_data'
    rollout_base_direct = '/nlp/scr/qinanyu/rl-explanations/trainers/direct/q3b-instruct/rollout_data'

    # Steps to analyze
    initial_step = 2
    final_step = 156

    print("=" * 80)
    print("ANALYZING CIR DECREASE VS THINKING-NO THINKING GAP")
    print("=" * 80)
    print(f"Model size: {args.model_size}")
    print(f"Initial step: {initial_step}")
    print(f"Final step: {final_step}")
    print("=" * 80)

    # Get all task folders
    task_folders = [item.replace('_cot_importance', '')
                    for item in os.listdir(cot_base_dir)
                    if os.path.isdir(os.path.join(cot_base_dir, item))]

    # Collect data
    tasks_with_cir_decrease = []
    tasks_with_cir_increase = []
    tasks_missing_data = []

    for task_name in sorted(task_folders):
        # Load CIR values
        initial_cir = load_cot_importance_at_step(cot_base_dir, task_name, initial_step)
        final_cir = load_cot_importance_at_step(cot_base_dir, task_name, final_step)

        # Load thinking and no-thinking accuracies at final step
        thinking_acc_final = load_step_accuracy_from_rollout(rollout_base_grpo, task_name, final_step)
        no_thinking_acc_final = load_step_accuracy_from_rollout(rollout_base_direct, task_name, final_step)

        # Check if all data is available
        if initial_cir is None or final_cir is None:
            tasks_missing_data.append((task_name, 'CIR'))
            continue

        if thinking_acc_final is None or no_thinking_acc_final is None:
            tasks_missing_data.append((task_name, 'accuracy'))
            continue

        # Calculate changes
        cir_change = final_cir - initial_cir
        thinking_gap = thinking_acc_final - no_thinking_acc_final

        # Categorize by CIR change
        if cir_change < 0:
            tasks_with_cir_decrease.append({
                'task': task_name,
                'initial_cir': initial_cir,
                'final_cir': final_cir,
                'cir_change': cir_change,
                'thinking_gap': thinking_gap,
                'thinking_acc': thinking_acc_final,
                'no_thinking_acc': no_thinking_acc_final
            })
        else:
            tasks_with_cir_increase.append({
                'task': task_name,
                'initial_cir': initial_cir,
                'final_cir': final_cir,
                'cir_change': cir_change,
                'thinking_gap': thinking_gap,
                'thinking_acc': thinking_acc_final,
                'no_thinking_acc': no_thinking_acc_final
            })

    # Print results
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print("=" * 80)
    print(f"Total tasks analyzed: {len(tasks_with_cir_decrease) + len(tasks_with_cir_increase)}")
    print(f"Tasks with CIR decrease: {len(tasks_with_cir_decrease)}")
    print(f"Tasks with CIR increase/unchanged: {len(tasks_with_cir_increase)}")
    print(f"Tasks with missing data: {len(tasks_missing_data)}")

    if tasks_with_cir_decrease:
        print(f"\n{'=' * 80}")
        print("TASKS WITH CIR DECREASE")
        print("=" * 80)

        thinking_gaps = [t['thinking_gap'] for t in tasks_with_cir_decrease]
        cir_changes = [t['cir_change'] for t in tasks_with_cir_decrease]

        print(f"\nFinal Thinking-NoThinking Gap Statistics:")
        print(f"  Mean: {np.mean(thinking_gaps):.4f}")
        print(f"  Median: {np.median(thinking_gaps):.4f}")
        print(f"  Std: {np.std(thinking_gaps):.4f}")
        print(f"  Min: {np.min(thinking_gaps):.4f}")
        print(f"  Max: {np.max(thinking_gaps):.4f}")

        print(f"\nCIR Change Statistics:")
        print(f"  Mean: {np.mean(cir_changes):.4f}")
        print(f"  Median: {np.median(cir_changes):.4f}")
        print(f"  Min: {np.min(cir_changes):.4f}")
        print(f"  Max: {np.max(cir_changes):.4f}")

        # Count positive vs negative thinking gaps
        positive_gaps = [t for t in tasks_with_cir_decrease if t['thinking_gap'] > 0]
        negative_gaps = [t for t in tasks_with_cir_decrease if t['thinking_gap'] < 0]
        zero_gaps = [t for t in tasks_with_cir_decrease if t['thinking_gap'] == 0]

        print(f"\nThinking Gap Distribution:")
        print(f"  Positive (thinking > no-thinking): {len(positive_gaps)}/{len(tasks_with_cir_decrease)} ({100*len(positive_gaps)/len(tasks_with_cir_decrease):.1f}%)")
        if positive_gaps:
            print(f"    Average gap: {np.mean([t['thinking_gap'] for t in positive_gaps]):.4f}")
        print(f"  Negative (thinking < no-thinking): {len(negative_gaps)}/{len(tasks_with_cir_decrease)} ({100*len(negative_gaps)/len(tasks_with_cir_decrease):.1f}%)")
        if negative_gaps:
            print(f"    Average gap: {np.mean([t['thinking_gap'] for t in negative_gaps]):.4f}")
        if zero_gaps:
            print(f"  Zero (thinking = no-thinking): {len(zero_gaps)}/{len(tasks_with_cir_decrease)}")

        # List all tasks
        print(f"\nDetailed Task List (sorted by thinking gap):")
        sorted_tasks = sorted(tasks_with_cir_decrease, key=lambda x: x['thinking_gap'], reverse=True)
        for i, task_info in enumerate(sorted_tasks, 1):
            print(f"  {i}. {task_info['task']}")
            print(f"     CIR: {task_info['initial_cir']:.4f} → {task_info['final_cir']:.4f} (Δ={task_info['cir_change']:.4f})")
            print(f"     Thinking gap: {task_info['thinking_gap']:.4f} (T={task_info['thinking_acc']:.4f}, NT={task_info['no_thinking_acc']:.4f})")

    if tasks_with_cir_increase:
        print(f"\n{'=' * 80}")
        print("COMPARISON: TASKS WITH CIR INCREASE/UNCHANGED")
        print("=" * 80)

        thinking_gaps_increase = [t['thinking_gap'] for t in tasks_with_cir_increase]

        print(f"\nFinal Thinking-NoThinking Gap Statistics:")
        print(f"  Mean: {np.mean(thinking_gaps_increase):.4f}")
        print(f"  Median: {np.median(thinking_gaps_increase):.4f}")

        positive_gaps_increase = [t for t in tasks_with_cir_increase if t['thinking_gap'] > 0]
        print(f"\nThinking Gap Distribution:")
        print(f"  Positive (thinking > no-thinking): {len(positive_gaps_increase)}/{len(tasks_with_cir_increase)} ({100*len(positive_gaps_increase)/len(tasks_with_cir_increase):.1f}%)")

    if tasks_missing_data:
        print(f"\n{'=' * 80}")
        print("TASKS WITH MISSING DATA")
        print("=" * 80)
        for task, reason in sorted(tasks_missing_data):
            print(f"  - {task}: missing {reason}")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
