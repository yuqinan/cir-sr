#!/usr/bin/env python3
"""
Compare verifier accuracy between new and old SR-reward training.

This script loads verifier accuracy from:
- cot_verifier_acc_trained_new_{scale}/{task}/-1.0/teacher (new training)
- cot_verifier_acc_trained_{scale}/{task}/-1.0/teacher (old training)

Usage:
  python analysis/compare_verifier_accuracy.py [model_size]
  python analysis/compare_verifier_accuracy.py 3b
  python analysis/compare_verifier_accuracy.py 1.5b
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


def load_verifier_accuracy_at_step(teacher_dir, step):
    """Load verifier accuracy from a teacher_responses json."""
    json_path = os.path.join(teacher_dir, f"step_{step}", f"teacher_responses_step_{step}.json")
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return None

    verifier_accuracies = []

    for item in data:
        # SR: verifier accuracy (answers_match with "no answer found" special-case)
        if "verifier_comparison" in item:
            verifier_comp = item["verifier_comparison"]
            answers_match = verifier_comp.get("answers_match", False)

            with_q = verifier_comp.get("with_question_answer", "").strip().lower()
            without_q = verifier_comp.get("without_question_answer", "").strip().lower()

            if answers_match and "answer" in with_q and "answer" in without_q:
                verifier_accuracies.append(0.0)
            if answers_match and with_q == "" and without_q == "":
                verifier_accuracies.append(0.0)
            else:
                verifier_accuracies.append(1.0 if answers_match else 0.0)

    if not verifier_accuracies:
        return None

    return {
        "verifier_accuracy": float(np.mean(verifier_accuracies)),
    }


def load_verifier_data(base_path, tasks, scales, steps_to_keep, scale0_base_path, prefix="cot_verifier_acc_trained", collect_individual=False):
    """Load verifier accuracy data for SR-reward training.

    Args:
        base_path: Base path to the evaluation results
        tasks: List of task names
        scales: List of verifier scales (beta values)
        steps_to_keep: List of training steps to keep
        scale0_base_path: Base path for scale=0 (baseline)
        prefix: Directory prefix (e.g., "cot_verifier_acc_trained" or "cot_verifier_acc_trained_new")
        collect_individual: If True, also collect per-task data
    """
    all_scale_data = []
    task_scale_data = {task: {} for task in tasks} if collect_individual else None

    for scale in scales:
        print(f"\n=== Processing {prefix} β={scale} ===")
        step_aggregated = {}

        for task in tasks:
            if scale == 0 or scale == 0.0:
                task_dir = f"{task}_cot_importance/-1.0/teacher"
                full_path = os.path.join(scale0_base_path, task_dir)
            else:
                task_dir = f"{prefix}_{scale}/{task}/-1.0/teacher"
                full_path = os.path.join(base_path, task_dir)

            if not os.path.exists(full_path):
                print(f"  Warning: Path not found for {task}: {full_path}")
                continue

            training_steps = get_available_steps(full_path)
            if not training_steps:
                print(f"  Warning: No training steps found for {task}")
                continue

            task_step_data = {}
            for step in training_steps:
                if step not in steps_to_keep:
                    continue
                metrics = load_verifier_accuracy_at_step(full_path, step)
                if metrics:
                    step_aggregated.setdefault(step, []).append(metrics)
                    if collect_individual:
                        task_step_data[step] = metrics

            if collect_individual and task_step_data:
                task_scale_data[task][scale] = task_step_data

        averaged_step_data = {}
        for step, metrics_list in step_aggregated.items():
            ver_vals = [m["verifier_accuracy"] for m in metrics_list if m["verifier_accuracy"] is not None]
            averaged_step_data[step] = {
                "verifier_accuracy": float(np.mean(ver_vals)) if ver_vals else None,
                "verifier_accuracy_std": float(np.std(ver_vals) / np.sqrt(len(ver_vals))) if len(ver_vals) > 0 else None
            }

        if averaged_step_data:
            all_scale_data.append((f"β={scale}", averaged_step_data))
            print(f"  Successfully averaged {len(averaged_step_data)} steps across tasks")
        else:
            print(f"  Warning: No valid data for β={scale}")

    if collect_individual:
        return all_scale_data, task_scale_data
    return all_scale_data


def plot_individual_tasks(old_task_data, new_task_data, tasks, scales, output_dir, model_size):
    """Plot individual task comparisons in horizontal layout.

    First row: SR+ (new training)
    Second row: R- (old training)
    """
    num_tasks = len(tasks)
    fig_width = max(24, num_tasks * 6)

    # 2 rows (new training on top, old training on bottom) x num_tasks columns
    fig, axes = plt.subplots(2, num_tasks, figsize=(fig_width, 8))

    # More visually distinct color palette (same as main plot)
    colors = ["#2E2E2E", "#1E88E5", "#FFA726", "#66BB6A", "#EF5350", "#AB47BC"]
    markers = ["o", "o", "o", "o", "o", "o"]

    for task_idx, task in enumerate(tasks):
        # Row 0: New training (SR+)
        ax_new = axes[0, task_idx]
        for scale_idx, scale in enumerate(scales):
            if task in new_task_data and scale in new_task_data[task]:
                step_data = new_task_data[task][scale]
                steps = []
                verifier_values = []
                for step in sorted(step_data.keys()):
                    if step_data[step]["verifier_accuracy"] is not None:
                        steps.append(step)
                        verifier_values.append(step_data[step]["verifier_accuracy"])

                if steps and verifier_values:
                    color = colors[scale_idx % len(colors)]
                    marker = markers[scale_idx % len(markers)]
                    ax_new.plot(steps, verifier_values, linewidth=3.5, color=color, marker=marker,
                               markersize=8, label=f'β={scale}', alpha=0.85, markeredgewidth=1.2,
                               markeredgecolor='white')

        # Set title with row label for first column
        title = f'{task.replace("_", " ").title()}'
        if task_idx == 0:
            ax_new.set_ylabel('SR+\n\nSR', fontsize=20, fontweight='bold', family='serif')
        ax_new.set_title(title, fontsize=24, fontweight='bold', family='serif', pad=15)
        ax_new.tick_params(axis='both', which='major', labelsize=14, width=1.2)
        ax_new.tick_params(axis='both', which='minor', labelsize=12, width=0.8)
        for label in ax_new.get_xticklabels() + ax_new.get_yticklabels():
            label.set_family('serif')
        ax_new.set_ylim(-0.02, 1.02)
        ax_new.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_new.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
        ax_new.minorticks_on()
        if task_idx == 0:
            ax_new.legend(fontsize=32, loc='best', framealpha=0.95, prop={'family': 'serif'},
                         edgecolor='gray', fancybox=True, shadow=True, markerscale=0.8)
        ax_new.set_axisbelow(True)
        for spine in ax_new.spines.values():
            spine.set_linewidth(1.2)

        # Row 1: Old training (R-)
        ax_old = axes[1, task_idx]
        for scale_idx, scale in enumerate(scales):
            if task in old_task_data and scale in old_task_data[task]:
                step_data = old_task_data[task][scale]
                steps = []
                verifier_values = []
                for step in sorted(step_data.keys()):
                    if step_data[step]["verifier_accuracy"] is not None:
                        steps.append(step)
                        verifier_values.append(step_data[step]["verifier_accuracy"])

                if steps and verifier_values:
                    color = colors[scale_idx % len(colors)]
                    marker = markers[scale_idx % len(markers)]
                    ax_old.plot(steps, verifier_values, linewidth=3.5, color=color, marker=marker,
                               markersize=8, label=f'β={scale}', alpha=0.85, markeredgewidth=1.2,
                               markeredgecolor='white')

        if task_idx == 0:
            ax_old.set_ylabel('R-\n\nSR', fontsize=20, fontweight='bold', family='serif')
        ax_old.set_xlabel('RL Training Step', fontsize=18, fontweight='bold', family='serif')
        ax_old.tick_params(axis='both', which='major', labelsize=14, width=1.2)
        ax_old.tick_params(axis='both', which='minor', labelsize=12, width=0.8)
        for label in ax_old.get_xticklabels() + ax_old.get_yticklabels():
            label.set_family('serif')
        ax_old.set_ylim(-0.02, 1.02)
        ax_old.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_old.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
        ax_old.minorticks_on()
        ax_old.set_axisbelow(True)
        for spine in ax_old.spines.values():
            spine.set_linewidth(1.2)

    plt.tight_layout()

    png_path = os.path.join(output_dir, f'verifier_acc_comparison_individual_{model_size}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, f'verifier_acc_comparison_individual_{model_size}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.close()

    return png_path, pdf_path


def plot_comparison(old_data, new_data, output_dir, model_size):
    """Plot comparison of verifier accuracy between old and new training runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # More visually distinct color palette
    colors = ["#2E2E2E", "#1E88E5", "#FFA726", "#66BB6A", "#EF5350", "#AB47BC"]
    markers = ["o", "o", "o", "o", "o", "o"]

    # Plot 1: Old training (cot_verifier_acc_trained)
    for idx, (scale_name, step_data) in enumerate(old_data):
        steps = []
        verifier_values = []
        for step in sorted(step_data.keys()):
            if step_data[step]["verifier_accuracy"] is not None:
                steps.append(step)
                verifier_values.append(step_data[step]["verifier_accuracy"])
        if steps and verifier_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax1.plot(steps, verifier_values, linewidth=4.0, color=color, marker=marker,
                     markersize=10, label=scale_name, alpha=0.85, markeredgewidth=1.5,
                     markeredgecolor='white')

    ax1.set_xlabel("RL Training Step", fontsize=35, fontweight="bold", family="serif")
    ax1.set_ylabel("Verifier Accuracy (SR+)", fontsize=35, fontweight="bold", family="serif")
    ax1.set_title("SR+ eval",
                  fontsize=35, fontweight="bold", pad=15, family="serif")
    ax1.tick_params(axis="both", which="major", labelsize=18, width=1.5)
    ax1.tick_params(axis="both", which="minor", labelsize=16, width=1)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_family("serif")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
    ax1.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    ax1.minorticks_on()
    ax1.legend(fontsize=56, loc="upper left", framealpha=0.95, prop={"family": "serif"},
               edgecolor='gray', fancybox=True, shadow=True, markerscale=1)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    # Plot 2: New training (cot_verifier_acc_trained_new)
    for idx, (scale_name, step_data) in enumerate(new_data):
        steps = []
        verifier_values = []
        for step in sorted(step_data.keys()):
            if step_data[step]["verifier_accuracy"] is not None:
                steps.append(step)
                verifier_values.append(step_data[step]["verifier_accuracy"])
        if steps and verifier_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax2.plot(steps, verifier_values, linewidth=4.0, color=color, marker=marker,
                     markersize=10, label=scale_name, alpha=0.85, markeredgewidth=1.5,
                     markeredgecolor='white')

    ax2.set_xlabel("RL Training Step", fontsize=35, fontweight="bold", family="serif")
    ax2.set_ylabel("Verifier Accuracy (SR-)", fontsize=35, fontweight="bold", family="serif")
    ax2.set_title("SR- eval",
                  fontsize=35, fontweight="bold", pad=15, family="serif")
    ax2.tick_params(axis="both", which="major", labelsize=18, width=1.5)
    ax2.tick_params(axis="both", which="minor", labelsize=16, width=1)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_family("serif")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
    ax2.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    ax2.minorticks_on()
    ax2.legend(fontsize=56, loc="upper left", framealpha=0.95, prop={"family": "serif"},
               edgecolor='gray', fancybox=True, shadow=True, markerscale=1)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout(pad=2.0)

    png_path = os.path.join(output_dir, f"verifier_acc_comparison_{model_size}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    pdf_path = os.path.join(output_dir, f"verifier_acc_comparison_{model_size}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description='Compare verifier accuracy between old and new SR-reward training')
    parser.add_argument('model_size', nargs='?', default='3b', help='Model size (default: 3b)')
    args = parser.parse_args()

    model_size = args.model_size

    # Tasks and scales
    tasks = ["binary_alternation", "binary_matrix", "bitwise_arithmetic", "count_bits",
             "manipulate_matrix", "futoshiki", "mini_sudoku", "rotate_matrix",
             "string_manipulation", "tsumego"]
    verifier_scales = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    steps_to_keep = [2, 30, 60, 90, 120, 150, 156]

    base_path = f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct"
    scale0_base_path = f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct/cot_importance"

    print(f"\nProcessing model: qwen-{model_size}-instruct")

    # Load old training data
    print("\n=== Loading OLD training data (cot_verifier_acc_trained) ===")
    old_data, old_task_data = load_verifier_data(base_path, tasks, verifier_scales, steps_to_keep,
                                                   scale0_base_path, prefix="cot_verifier_acc_trained_sr_plus",
                                                   collect_individual=True)

    # Load new training data
    print("\n=== Loading NEW training data (cot_verifier_acc_trained_new) ===")
    new_data, new_task_data = load_verifier_data(base_path, tasks, verifier_scales, steps_to_keep,
                                                   scale0_base_path, prefix="cot_verifier_acc_trained_sr_minus",
                                                   collect_individual=True)

    if not old_data or not new_data:
        print("\nError: Missing data for old or new training")
        return

    # Generate comparison plot
    output_dir = "/nlp/scr/qinanyu/rl-explanations/analysis/graph"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    print("\n=== Generating aggregated comparison plot ===")
    png_path, pdf_path = plot_comparison(old_data, new_data, output_dir, model_size)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    print("\n=== Generating individual task comparison plots ===")
    ind_png_path, ind_pdf_path = plot_individual_tasks(old_task_data, new_task_data, tasks,
                                                        verifier_scales, output_dir, model_size)
    print(f"Saved: {ind_png_path}")
    print(f"Saved: {ind_pdf_path}")


if __name__ == "__main__":
    main()
