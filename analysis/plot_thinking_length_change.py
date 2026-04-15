#!/usr/bin/env python3
"""
Plot initial and final thinking length for each task as vertical lines with dots.

Usage:
    python plot_thinking_length_change.py [model_size]
    python plot_thinking_length_change.py 3b
    python plot_thinking_length_change.py 1.5b
    python plot_thinking_length_change.py --local  # for local testing
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


def load_step_data(base_dir, task_name, step):
    """Load thinking length for a specific step. Returns value or None."""
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

                # Calculate average thinking length
                thinking_lengths = []
                for item in data:
                    # Look for thinking text in the correct field
                    thinking_text = None
                    if 'teacher_thinking' in item:
                        thinking_text = item['teacher_thinking']
                    elif 'thinking_without_answer' in item:
                        thinking_text = item['thinking_without_answer']
                    elif 'teacher_response' in item:
                        thinking_text = item['teacher_response']
                    elif 'explanation' in item:
                        thinking_text = item['explanation']
                    elif 'response' in item:
                        thinking_text = item['response']

                    if thinking_text:
                        # Count number of characters
                        thinking_lengths.append(len(thinking_text))

                return np.mean(thinking_lengths) if thinking_lengths else None
            except:
                pass
    return None


def main():
    parser = argparse.ArgumentParser(description='Plot thinking length change across training steps')
    parser.add_argument('model_size', nargs='?', default='3b', help='Model size (default: 3b)')
    parser.add_argument('--local', action='store_true', help='Use local paths (for testing)')
    parser.add_argument('--model', type=str, default=None, help='Model name for rollout data (default: auto-detected from model_size)')
    parser.add_argument('--thinking_dir', type=str, default='grpo', help='Directory name for thinking variant')
    parser.add_argument('--variant', type=str, default='original', help='Variant folder name (default: original)')
    parser.add_argument('--batch', type=str, default='8', help='Batch folder name (default: 8)')

    args = parser.parse_args()

    # Auto-detect model name from model_size if not provided
    if args.model is None:
        if args.model_size == 'llama-3-3b':
            args.model = 'llama-3-3b-instruct'
        else:
            args.model = f'q{args.model_size}-instruct'

    # Define training steps (only before and after training)
    if args.model_size == 'llama-3-3b':
        training_steps = [3, 156]
    else:
        training_steps = [2, 156]
    step_labels = {training_steps[0]: "Before Training", training_steps[-1]: "After Training"}

    # Determine result directory name based on model
    if args.model_size == 'llama-3-3b':
        result_dir_name = 'grpo_llama-3-3b-instruct'
    else:
        result_dir_name = f'grpo_qwen-{args.model_size}-instruct'

    # Set base directories
    if args.local:
        base_dir = f'/Users/qinanyu/Desktop/rl-explanations/evaluate/results/{result_dir_name}/cot_importance'
        rollout_base = f'/Users/qinanyu/Desktop/rl-explanations/trainers/{args.thinking_dir}/{args.model}/rollout_data'
        output_dir = '/Users/qinanyu/Desktop/rl-explanations/analysis/graph'
    else:
        base_dir = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/{result_dir_name}/cot_importance'
        rollout_base = f'/nlp/scr/qinanyu/rl-explanations/trainers/{args.thinking_dir}/{args.model}/rollout_data'
        output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis/graph'

    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return

    # Collect data
    task_data = {}
    task_folders = [item.replace('_cot_importance', '')
                    for item in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, item))]

    for task_name in sorted(task_folders):
        # Load thinking length for all steps
        step_values = {}
        for step in training_steps:
            thinking_length = load_step_data(base_dir, task_name, step)
            if thinking_length is not None:
                step_values[step] = thinking_length

        # Load final thinking accuracy for sorting (use last step)
        thinking_acc = load_step_accuracy(rollout_base, task_name, training_steps[-1], args.variant, args.batch)

        # Only include tasks that have data for at least first and last step
        if training_steps[0] in step_values or training_steps[-1] in step_values:
            task_data[task_name] = {
                'steps': step_values,
                'thinking_end': thinking_acc if thinking_acc is not None else 0.0
            }

    if not task_data:
        print("No task data found")
        return

    # Sort tasks by change in thinking length (highest drop to highest improvement)
    # Change = last_step_value - first_step_value
    def get_length_change(item):
        task_name, data = item
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            return steps[available_steps[-1]] - steps[available_steps[0]]
        return 0

    sorted_tasks = sorted(task_data.items(), key=get_length_change)
    task_names = [task for task, _ in sorted_tasks]

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot lines and dots for each task
    x_positions = np.arange(len(task_names))

    # Color scheme for training steps
    step_colors = {
        training_steps[0]: (0.678, 0.847, 0.902),    # Light blue for "Before Training"
        training_steps[-1]: (0.098, 0.275, 0.471),  # Dark blue for "After Training"
    }

    # Add legend entries for both steps (dummy points outside plot)
    for step in training_steps:
        ax.scatter([], [], s=80, color=step_colors[step],
                  edgecolors='black', linewidth=1.5,
                  label=step_labels[step])

    for i, (task, data) in enumerate(zip(task_names, [d for _, d in sorted_tasks])):
        step_values = data['steps']

        # Get all available step values for this task
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

        # Plot dots at each available step
        for step, y_val, color in zip(available_steps, y_values, colors):
            ax.scatter(i, y_val, s=80, color=color,
                       edgecolors='black', linewidth=1.5, zorder=3)

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(task_names, rotation=70, ha='right', fontsize=19, family='serif')
    ax.set_xlabel('Tasks (sorted by thinking length change: highest drop → highest improvement)',
                  fontsize=25, fontweight='bold', family='serif')
    ax.set_ylabel('Thinking Length (characters)',
                  fontsize=25, fontweight='bold', family='serif')

    ax.set_title('Evolution of Thinking Length Across Training',
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
    output_path = os.path.join(output_dir, f'thinking_length_{args.model_size}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, f'thinking_length_{args.model_size}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"Successfully saved: {output_path}")
    print(f"Successfully saved: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()
