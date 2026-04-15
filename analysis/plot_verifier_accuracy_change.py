#!/usr/bin/env python3
"""
Plot initial and final Verifier Accuracy for each task as vertical lines with dots.

Usage:
    python plot_verifier_accuracy_change.py [model_size]
    python plot_verifier_accuracy_change.py 3b
    python plot_verifier_accuracy_change.py 1.5b
    python plot_verifier_accuracy_change.py --local  # for local testing
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
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


def load_step_accuracy(base_dir, task_name, step, variant="original", batch="8"):
    """Load accuracy for a specific step."""
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

            # Check if both answers are "no answer found"
            with_q_answer = verifier_comp.get('with_question_answer', '').strip().lower()
            without_q_answer = verifier_comp.get('without_question_answer', '').strip().lower()

            # If both are "no answer found", treat as incorrect (0) even if answers_match is True
            if answers_match and with_q_answer == 'no answer found' and without_q_answer == 'no answer found':
                verifier_scores.append(0)
            else:
                verifier_scores.append(1 if answers_match else 0)

    if verifier_scores:
        return np.mean(verifier_scores)
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description='Plot Verifier Accuracy change across training steps')
    parser.add_argument('model_size', nargs='?', default='3b', help='Model size (default: 3b)')
    parser.add_argument('--local', action='store_true', help='Use local paths (for testing)')
    parser.add_argument('--model', type=str, default='q3b-instruct', help='Model name for rollout data (default: q3b-instruct)')
    parser.add_argument('--thinking_dir', type=str, default='grpo', help='Directory name for thinking variant')
    parser.add_argument('--variant', type=str, default='original', help='Variant folder name (default: original)')
    parser.add_argument('--batch', type=str, default='8', help='Batch folder name (default: 8)')

    args = parser.parse_args()

    # Define training steps
    training_steps = [2, 30, 60, 90, 120, 156]

    # Set base directories
    if args.local:
        base_dir = f'/Users/qinanyu/Desktop/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'
        rollout_base = f'/Users/qinanyu/Desktop/rl-explanations/trainers/{args.thinking_dir}/{args.model}/rollout_data'
        output_dir = '/Users/qinanyu/Desktop/rl-explanations/analysis/graph'
    else:
        base_dir = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'
        rollout_base = f'/nlp/scr/qinanyu/rl-explanations/trainers/{args.thinking_dir}/{args.model}/rollout_data'
        output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis/graph'

    if not os.path.exists(base_dir):
        return

    # Collect data
    task_data = {}
    task_folders = [item.replace('_cot_importance', '')
                    for item in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, item))]

    for task_name in sorted(task_folders):
        # Load verifier accuracy for all steps
        step_values = {}
        for step in training_steps:
            verifier_acc = load_verifier_accuracy_at_step(base_dir, task_name, step)
            if verifier_acc is not None:
                step_values[step] = verifier_acc

        # Load final thinking accuracy for sorting (use step 156)
        thinking_acc = load_step_accuracy(rollout_base, task_name, 156, args.variant, args.batch)

        # Only include tasks that have data for at least first and last step
        if (2 in step_values or 156 in step_values or 150 in step_values) and thinking_acc is not None:
            task_data[task_name] = {
                'steps': step_values,
                'thinking_end': thinking_acc
            }

    if not task_data:
        return

    # Sort tasks by change in verifier accuracy (highest drop to highest improvement)
    # Change = last_step_value - first_step_value
    def get_verifier_change(item):
        task_name, data = item
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            return steps[available_steps[-1]] - steps[available_steps[0]]
        return 0

    sorted_tasks = sorted(task_data.items(), key=get_verifier_change)
    task_names = [task for task, _ in sorted_tasks]

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot lines and dots for each task
    x_positions = np.arange(len(task_names))

    # Create blue gradient (light to dark)
    num_steps = len(training_steps)
    blue_gradient = [
        (0.678, 0.847, 0.902),  # Light blue for step 2
        (0.529, 0.808, 0.922),  # Step 30
        (0.416, 0.678, 0.839),  # Step 60
        (0.282, 0.525, 0.671),  # Step 90
        (0.176, 0.396, 0.573),  # Step 120
        (0.098, 0.275, 0.471),  # Dark blue for step 156
    ]

    for i, (task, data) in enumerate(zip(task_names, [d for _, d in sorted_tasks])):
        step_values = data['steps']

        # Get all available step values for this task
        available_steps = []
        y_values = []
        colors = []
        for step_idx, step in enumerate(training_steps):
            if step in step_values:
                available_steps.append(step)
                y_values.append(step_values[step])
                colors.append(blue_gradient[step_idx])

        # Determine if increasing or decreasing
        if len(available_steps) > 1:
            first_val = y_values[0]
            last_val = y_values[-1]
            is_increasing = last_val > first_val
            line_color = '#2E86AB' if is_increasing else '#D32F2F'  # Blue for increase, red for decrease

            # Draw line with multiple chevron arrows
            num_chevrons = 5  # Number of chevron arrows along the line
            for idx in range(num_chevrons):
                # Calculate start and end positions for each chevron segment
                start_frac = idx / num_chevrons
                end_frac = (idx + 1) / num_chevrons

                y_start = y_values[0] + start_frac * (y_values[-1] - y_values[0])
                y_end = y_values[0] + end_frac * (y_values[-1] - y_values[0])

                # Draw chevron arrow segment
                ax.annotate('', xy=(i, y_end), xytext=(i, y_start),
                           arrowprops=dict(arrowstyle='->', color=line_color, lw=2.5, alpha=0.7,
                                         mutation_scale=15), zorder=2)

        # Plot dots at each available step with gradient colors
        for step_idx, (step, y_val, color) in enumerate(zip(available_steps, y_values, colors)):
            # Only add label once per step (for first task)
            step_original_idx = training_steps.index(step)
            ax.scatter(i, y_val, s=80, color=color,
                       edgecolors='black', linewidth=1.5, zorder=3,
                       label=f'Step {step}' if i == 0 else '')

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(task_names, rotation=70, ha='right', fontsize=19, family='serif')
    ax.set_xlabel('Tasks (sorted by verifier accuracy change: highest drop → highest improvement)',
                  fontsize=25, fontweight='bold', family='serif')
    ax.set_ylabel('Verifier Accuracy',
                  fontsize=25, fontweight='bold', family='serif')

    ax.set_title(f'Evolution of Verifier Accuracy Across Training\n' +
                 f'Steps: {", ".join(map(str, training_steps))} (n={len(task_names)} tasks)',
                 fontsize=25, fontweight='bold', pad=20, family='serif')

    # Set y-axis tick labels to serif font
    for label in ax.get_yticklabels():
        label.set_family('serif')
        label.set_fontsize(19)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y', zorder=0)
    ax.legend(loc='upper right', fontsize=19, framealpha=0.9, prop={'family': 'serif'})

    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'figure3_verifier_accuracy_{args.model_size}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, f'figure3_verifier_accuracy_{args.model_size}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
