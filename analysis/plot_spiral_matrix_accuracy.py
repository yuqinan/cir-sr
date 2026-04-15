#!/usr/bin/env python3
"""
Plot accuracy across training steps for spiral_matrix_original experiment.

This script visualizes accuracy metrics across different training settings (8, 16, 32, 64)
and training steps.

Usage:
  python analysis/plot_spiral_matrix_accuracy.py
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Set font to serif for consistency
plt.rcParams['font.family'] = 'serif'


def get_available_steps(base_path, setting):
    """Get all available training steps for a given setting."""
    teacher_dir = os.path.join(base_path, str(setting), "0.0", "teacher")
    if not os.path.exists(teacher_dir):
        return []

    steps = []
    for item in os.listdir(teacher_dir):
        if item.startswith('step_'):
            match = re.search(r'step_(\d+)', item)
            if match:
                steps.append(int(match.group(1)))
    return sorted(steps)


def load_accuracy_for_step(base_path, setting, step, task_filter=None):
    """
    Load accuracy data for a specific setting and step.

    Args:
        base_path: Base path to the experiment results
        setting: Training setting (8, 16, 32, 64)
        step: Training step number
        task_filter: Optional list of task names to include (e.g., ['spiral_matrix_easy', 'spiral_matrix_medium'])

    Returns:
        Dictionary with task names as keys and accuracy values
    """
    step_dir = os.path.join(base_path, str(setting), "0.0", "teacher", f"step_{step}", "generalization")

    if not os.path.exists(step_dir):
        print(f"Warning: Path not found: {step_dir}")
        return {}

    task_accuracies = {}

    for json_file in os.listdir(step_dir):
        if not json_file.endswith('.json'):
            continue

        # Extract task name from filename
        task_name = json_file.replace(f'_step_{step}.json', '')

        # Apply task filter if provided
        if task_filter and task_name not in task_filter:
            continue

        json_path = os.path.join(step_dir, json_file)

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Calculate accuracy
            scores = [item.get('score', 0.0) for item in data if 'score' in item]
            if scores:
                accuracy = np.mean(scores)
                task_accuracies[task_name] = accuracy
        except Exception as e:
            print(f"Error loading {json_path}: {e}")

    return task_accuracies


def plot_accuracy_across_steps(base_path, settings, task_filter=None, output_dir=None):
    """
    Plot accuracy across training steps for different settings.

    Args:
        base_path: Base path to the experiment results
        settings: List of training settings (e.g., [8, 16, 32, 64])
        task_filter: Optional list of task names to include
        output_dir: Directory to save the plot (default: same as base_path)
    """
    if output_dir is None:
        output_dir = base_path

    # Collect data for each setting
    setting_data = {}

    for setting in settings:
        print(f"\n=== Processing setting {setting} ===")
        steps = get_available_steps(base_path, setting)

        if not steps:
            print(f"  Warning: No steps found for setting {setting}")
            continue

        print(f"  Found steps: {steps}")

        step_accuracies = {}
        for step in steps:
            task_accs = load_accuracy_for_step(base_path, setting, step, task_filter)

            if task_accs:
                # Calculate average accuracy across all tasks
                avg_accuracy = np.mean(list(task_accs.values()))
                step_accuracies[step] = {
                    'average': avg_accuracy,
                    'tasks': task_accs
                }
                print(f"  Step {step}: Avg accuracy = {avg_accuracy:.3f}, Tasks: {len(task_accs)}")

        if step_accuracies:
            setting_data[setting] = step_accuracies

    if not setting_data:
        print("\nError: No data found to plot")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palette
    colors = ["#2E2E2E", "#1E88E5", "#FFA726", "#66BB6A", "#EF5350", "#AB47BC"]
    markers = ["o", "s", "^", "D", "v", "p"]

    for idx, (setting, step_data) in enumerate(sorted(setting_data.items())):
        steps = sorted(step_data.keys())
        accuracies = [step_data[step]['average'] for step in steps]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        ax.plot(steps, accuracies, linewidth=3.5, color=color, marker=marker,
                markersize=10, label=f'Setting {setting}', alpha=0.85,
                markeredgewidth=1.5, markeredgecolor='white')

    ax.set_xlabel("Training Step", fontsize=20, fontweight="bold", family="serif")
    ax.set_ylabel("Average Accuracy", fontsize=20, fontweight="bold", family="serif")
    ax.set_title("Accuracy Across Training Steps (Spiral Matrix Original)",
                 fontsize=24, fontweight="bold", pad=15, family="serif")
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.5)
    ax.tick_params(axis="both", which="minor", labelsize=14, width=1)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family("serif")

    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    ax.minorticks_on()
    ax.legend(fontsize=16, loc="best", framealpha=0.95, prop={"family": "serif"},
              edgecolor='gray', fancybox=True, shadow=True, markerscale=1.2)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    # Save the plot
    png_path = os.path.join(output_dir, "spiral_matrix_accuracy.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {png_path}")

    pdf_path = os.path.join(output_dir, "spiral_matrix_accuracy.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    plt.close()

    return png_path, pdf_path


def plot_accuracy_by_task(base_path, settings, tasks, output_dir=None):
    """
    Plot accuracy across training steps separately for each task.

    Args:
        base_path: Base path to the experiment results
        settings: List of training settings (e.g., [8, 16, 32, 64])
        tasks: List of task names to plot
        output_dir: Directory to save the plot (default: same as base_path)
    """
    if output_dir is None:
        output_dir = base_path

    num_tasks = len(tasks)
    fig, axes = plt.subplots(num_tasks, 1, figsize=(12, 6 * num_tasks))

    if num_tasks == 1:
        axes = [axes]

    # Color palette
    colors = ["#2E2E2E", "#1E88E5", "#FFA726", "#66BB6A", "#EF5350", "#AB47BC"]
    markers = ["o", "s", "^", "D", "v", "p"]

    for task_idx, task_name in enumerate(tasks):
        ax = axes[task_idx]

        for setting_idx, setting in enumerate(settings):
            steps = get_available_steps(base_path, setting)

            if not steps:
                continue

            task_accuracies = []
            valid_steps = []

            for step in steps:
                task_accs = load_accuracy_for_step(base_path, setting, step, [task_name])

                if task_name in task_accs:
                    task_accuracies.append(task_accs[task_name])
                    valid_steps.append(step)

            if task_accuracies:
                color = colors[setting_idx % len(colors)]
                marker = markers[setting_idx % len(markers)]

                ax.plot(valid_steps, task_accuracies, linewidth=3.5, color=color,
                       marker=marker, markersize=10, label=f'Setting {setting}',
                       alpha=0.85, markeredgewidth=1.5, markeredgecolor='white')

        ax.set_ylabel("Accuracy", fontsize=18, fontweight="bold", family="serif")
        ax.set_title(f"{task_name.replace('_', ' ').title()}",
                    fontsize=20, fontweight="bold", pad=15, family="serif")
        ax.tick_params(axis="both", which="major", labelsize=14, width=1.5)
        ax.tick_params(axis="both", which="minor", labelsize=12, width=1)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_family("serif")

        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
        ax.minorticks_on()
        ax.legend(fontsize=14, loc="best", framealpha=0.95, prop={"family": "serif"},
                 edgecolor='gray', fancybox=True, shadow=True, markerscale=1.2)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        if task_idx == num_tasks - 1:
            ax.set_xlabel("Training Step", fontsize=18, fontweight="bold", family="serif")

    plt.tight_layout()

    # Save the plot
    png_path = os.path.join(output_dir, "spiral_matrix_accuracy_by_task.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {png_path}")

    pdf_path = os.path.join(output_dir, "spiral_matrix_accuracy_by_task.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    plt.close()

    return png_path, pdf_path


def main():
    # Configuration
    base_path = "/Users/qinanyu/Desktop/rl-explanations/evaluate/results/grpo_qwen-3b-instruct_original/spiral_matrix_original"
    settings = [8, 16, 32, 64]

    # Output directory
    output_dir = "/Users/qinanyu/Desktop/rl-explanations/analysis/graph"
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Average accuracy across all tasks
    print("=== Generating average accuracy plot ===")
    plot_accuracy_across_steps(base_path, settings, output_dir=output_dir)

    # Plot 2: Accuracy by specific tasks (spiral matrix tasks only)
    print("\n=== Generating per-task accuracy plot ===")
    spiral_tasks = ['spiral_matrix_easy', 'spiral_matrix_medium', 'spiral_matrix_hard']
    plot_accuracy_by_task(base_path, settings, spiral_tasks, output_dir=output_dir)

    # Plot 3: Average accuracy for spiral matrix tasks only
    print("\n=== Generating spiral matrix average accuracy plot ===")
    plot_accuracy_across_steps(base_path, settings, task_filter=spiral_tasks,
                               output_dir=output_dir)


if __name__ == "__main__":
    main()
