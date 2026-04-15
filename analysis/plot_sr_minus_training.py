#!/usr/bin/env python3
"""
Plot verifier accuracy and accuracy for SR- training.

This script loads data from:
- /nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-3b-instruct/cot_verifier_acc_sr_minus_trained_sr_minus_1.0/{task}/-1.0/teacher

Usage:
  python analysis/plot_sr_minus_training.py [model_size]
  python analysis/plot_sr_minus_training.py 3b
  python analysis/plot_sr_minus_training.py 1.5b
"""

import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Set font to serif for consistency
plt.rcParams['font.family'] = 'serif'


def get_available_steps(teacher_dir):
    """Automatically detect available training steps from a teacher directory."""
    if not os.path.exists(teacher_dir):
        return []
    steps = []
    for item in os.listdir(teacher_dir):
        if item.startswith('step_'):
            match = re.search(r'step_(\d+)', item)
            if match:
                steps.append(int(match.group(1)))
    return sorted(steps)


def load_metrics_at_step(teacher_dir, step):
    """Load verifier accuracy and accuracy from a teacher_responses json."""
    json_path = os.path.join(teacher_dir, f"step_{step}", f"teacher_responses_step_{step}.json")
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return None

    verifier_accuracies = []
    accuracies = []

    for item in data:
        # Teacher accuracy (reward_score averaged across k responses)
        if "k_responses" in item:
            k_scores = [kr.get("reward_score") for kr in item["k_responses"] if "reward_score" in kr]
            if k_scores:
                accuracies.append(float(np.mean(k_scores)))

        # SR: verifier accuracy (answers_match with "no answer found" special-case)
        if "verifier_comparison" in item:
            verifier_comp = item["verifier_comparison"]
            answers_match = verifier_comp.get("answers_match", False)

            with_q = verifier_comp.get("with_question_answer", "").strip().lower()
            without_q = verifier_comp.get("without_question_answer", "").strip().lower()

            if answers_match and with_q == "no answer found" and without_q == "no answer found":
                verifier_accuracies.append(0.0)
            else:
                verifier_accuracies.append(1.0 if answers_match else 0.0)

    if not verifier_accuracies and not accuracies:
        return None

    return {
        "verifier_accuracy": float(np.mean(verifier_accuracies)) if verifier_accuracies else None,
        "accuracy": float(np.mean(accuracies)) if accuracies else None,
    }


def load_sr_minus_data(base_path, tasks, steps_to_keep, collect_individual=False):
    """Load SR- training data.

    Args:
        base_path: Base path to the evaluation results
        tasks: List of task names
        steps_to_keep: List of training steps to keep
        collect_individual: If True, also collect per-task data
    """
    step_aggregated = {}
    task_step_data = {task: {} for task in tasks} if collect_individual else None

    print(f"\n=== Processing SR- training ===")

    for task in tasks:
        task_dir = f"cot_verifier_acc_sr_minus_trained_sr_minus_1.0/{task}/-1.0/teacher"
        full_path = os.path.join(base_path, task_dir)

        if not os.path.exists(full_path):
            print(f"  Warning: Path not found for {task}: {full_path}")
            continue

        training_steps = get_available_steps(full_path)
        if not training_steps:
            print(f"  Warning: No training steps found for {task}")
            continue

        task_metrics = {}
        for step in training_steps:
            if step not in steps_to_keep:
                continue
            metrics = load_metrics_at_step(full_path, step)
            if metrics:
                step_aggregated.setdefault(step, []).append(metrics)
                if collect_individual:
                    task_metrics[step] = metrics

        if collect_individual and task_metrics:
            task_step_data[task] = task_metrics

        print(f"  Loaded {len(task_metrics)} steps for {task}")

    # Aggregate across tasks
    averaged_step_data = {}
    for step, metrics_list in step_aggregated.items():
        ver_vals = [m["verifier_accuracy"] for m in metrics_list if m["verifier_accuracy"] is not None]
        acc_vals = [m["accuracy"] for m in metrics_list if m["accuracy"] is not None]
        averaged_step_data[step] = {
            "verifier_accuracy": float(np.mean(ver_vals)) if ver_vals else None,
            "verifier_accuracy_std": float(np.std(ver_vals) / np.sqrt(len(ver_vals))) if len(ver_vals) > 0 else None,
            "accuracy": float(np.mean(acc_vals)) if acc_vals else None,
            "accuracy_std": float(np.std(acc_vals) / np.sqrt(len(acc_vals))) if len(acc_vals) > 0 else None
        }

    print(f"  Successfully averaged {len(averaged_step_data)} steps across {len(tasks)} tasks")

    if collect_individual:
        return averaged_step_data, task_step_data
    return averaged_step_data


def plot_sr_minus_aggregated(step_data, output_dir, model_size):
    """Plot aggregated SR- training metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # Extract data
    steps = sorted(step_data.keys())
    verifier_values = [step_data[s]["verifier_accuracy"] for s in steps if step_data[s]["verifier_accuracy"] is not None]
    accuracy_values = [step_data[s]["accuracy"] for s in steps if step_data[s]["accuracy"] is not None]
    steps_ver = [s for s in steps if step_data[s]["verifier_accuracy"] is not None]
    steps_acc = [s for s in steps if step_data[s]["accuracy"] is not None]

    # Color and marker
    color = "#1E88E5"
    marker = "o"

    # Plot 1: Verifier Accuracy (SR)
    ax1.plot(steps_ver, verifier_values, linewidth=4.0, color=color, marker=marker,
             markersize=10, label='SR- Training', alpha=0.85, markeredgewidth=1.5,
             markeredgecolor='white')

    ax1.set_xlabel("RL Training Step", fontsize=35, fontweight="bold", family="serif")
    ax1.set_ylabel("SR", fontsize=35, fontweight="bold", family="serif")
    ax1.set_title("SR- Training → SR",
                  fontsize=35, fontweight="bold", pad=15, family="serif")
    ax1.tick_params(axis="both", which="major", labelsize=18, width=1.5)
    ax1.tick_params(axis="both", which="minor", labelsize=16, width=1)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_family("serif")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
    ax1.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    ax1.minorticks_on()
    ax1.legend(fontsize=56, loc="best", framealpha=0.95, prop={"family": "serif"},
               edgecolor='gray', fancybox=True, shadow=True, markerscale=1)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    # Plot 2: Accuracy
    ax2.plot(steps_acc, accuracy_values, linewidth=4.0, color=color, marker=marker,
             markersize=10, label='SR- Training', alpha=0.85, markeredgewidth=1.5,
             markeredgecolor='white')

    ax2.set_xlabel("RL Training Step", fontsize=35, fontweight="bold", family="serif")
    ax2.set_ylabel("Accuracy", fontsize=35, fontweight="bold", family="serif")
    ax2.set_title("SR- Training → Accuracy",
                  fontsize=35, fontweight="bold", pad=15, family="serif")
    ax2.tick_params(axis="both", which="major", labelsize=18, width=1.5)
    ax2.tick_params(axis="both", which="minor", labelsize=16, width=1)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_family("serif")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
    ax2.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    ax2.minorticks_on()
    ax2.legend(fontsize=56, loc="best", framealpha=0.95, prop={"family": "serif"},
               edgecolor='gray', fancybox=True, shadow=True, markerscale=1)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout(pad=2.0)

    png_path = os.path.join(output_dir, f"sr_minus_training_{model_size}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    pdf_path = os.path.join(output_dir, f"sr_minus_training_{model_size}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()
    return png_path, pdf_path


def plot_sr_minus_individual(task_data, tasks, output_dir, model_size):
    """Plot individual task metrics for SR- training in horizontal layout."""
    num_tasks = len(tasks)
    fig_width = max(24, num_tasks * 6)

    # 2 rows (verifier accuracy and accuracy) x num_tasks columns
    fig, axes = plt.subplots(2, num_tasks, figsize=(fig_width, 8))

    # Color and marker
    color = "#1E88E5"
    marker = "o"

    for task_idx, task in enumerate(tasks):
        if task not in task_data or not task_data[task]:
            continue

        step_data = task_data[task]
        steps = sorted(step_data.keys())

        # Row 0: Verifier Accuracy (SR)
        ax_ver = axes[0, task_idx]
        verifier_values = [step_data[s]["verifier_accuracy"] for s in steps if step_data[s]["verifier_accuracy"] is not None]
        steps_ver = [s for s in steps if step_data[s]["verifier_accuracy"] is not None]

        if steps_ver and verifier_values:
            ax_ver.plot(steps_ver, verifier_values, linewidth=3.5, color=color, marker=marker,
                       markersize=8, alpha=0.85, markeredgewidth=1.2, markeredgecolor='white')

        title = f'{task.replace("_", " ").title()}'
        if task_idx == 0:
            ax_ver.set_ylabel('SR', fontsize=20, fontweight='bold', family='serif')
        ax_ver.set_title(title, fontsize=24, fontweight='bold', family='serif', pad=15)
        ax_ver.tick_params(axis='both', which='major', labelsize=14, width=1.2)
        ax_ver.tick_params(axis='both', which='minor', labelsize=12, width=0.8)
        for label in ax_ver.get_xticklabels() + ax_ver.get_yticklabels():
            label.set_family('serif')
        ax_ver.set_ylim(-0.02, 1.02)
        ax_ver.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_ver.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
        ax_ver.minorticks_on()
        ax_ver.set_axisbelow(True)
        for spine in ax_ver.spines.values():
            spine.set_linewidth(1.2)

        # Row 1: Accuracy
        ax_acc = axes[1, task_idx]
        accuracy_values = [step_data[s]["accuracy"] for s in steps if step_data[s]["accuracy"] is not None]
        steps_acc = [s for s in steps if step_data[s]["accuracy"] is not None]

        if steps_acc and accuracy_values:
            ax_acc.plot(steps_acc, accuracy_values, linewidth=3.5, color=color, marker=marker,
                       markersize=8, alpha=0.85, markeredgewidth=1.2, markeredgecolor='white')

        if task_idx == 0:
            ax_acc.set_ylabel('Accuracy', fontsize=20, fontweight='bold', family='serif')
        ax_acc.set_xlabel('RL Training Step', fontsize=18, fontweight='bold', family='serif')
        ax_acc.tick_params(axis='both', which='major', labelsize=14, width=1.2)
        ax_acc.tick_params(axis='both', which='minor', labelsize=12, width=0.8)
        for label in ax_acc.get_xticklabels() + ax_acc.get_yticklabels():
            label.set_family('serif')
        ax_acc.set_ylim(-0.02, 1.02)
        ax_acc.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_acc.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
        ax_acc.minorticks_on()
        ax_acc.set_axisbelow(True)
        for spine in ax_acc.spines.values():
            spine.set_linewidth(1.2)

    plt.tight_layout()

    png_path = os.path.join(output_dir, f'sr_minus_training_individual_{model_size}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, f'sr_minus_training_individual_{model_size}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.close()

    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description='Plot SR- training metrics (verifier accuracy and accuracy)')
    parser.add_argument('model_size', nargs='?', default='3b', help='Model size (default: 3b)')
    args = parser.parse_args()

    model_size = args.model_size

    # Tasks
    tasks = ["binary_alternation", "binary_matrix", "bitwise_arithmetic", "count_bits",
             "manipulate_matrix", "futoshiki", "mini_sudoku", "rotate_matrix",
             "string_manipulation", "tsumego"]
    steps_to_keep = [2, 10, 20, 30, 39]

    base_path = f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct"

    print(f"\nProcessing model: qwen-{model_size}-instruct")
    print(f"Loading SR- training data")

    # Load data
    step_data, task_data = load_sr_minus_data(base_path, tasks, steps_to_keep, collect_individual=True)

    if not step_data:
        print("\nError: No data found for SR- training")
        return

    # Generate plots
    output_dir = "/nlp/scr/qinanyu/rl-explanations/analysis/graph"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    print("\n=== Generating aggregated SR- training plot ===")
    png_path, pdf_path = plot_sr_minus_aggregated(step_data, output_dir, model_size)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    print("\n=== Generating individual task plots for SR- training ===")
    ind_png_path, ind_pdf_path = plot_sr_minus_individual(task_data, tasks, output_dir, model_size)
    print(f"Saved: {ind_png_path}")
    print(f"Saved: {ind_pdf_path}")


if __name__ == "__main__":
    main()
