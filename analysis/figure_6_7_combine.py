#!/usr/bin/env python3
"""
Combined plot showing CIR and SR changes for both SR-reward and CIR-reward training.

This combines figure_6 (SR-reward) and figure_7 (CIR-reward) into a single figure
with side-by-side subplots.

Usage:
  python analysis/figure_6_7_combine.py [model_size] [--mode MODE]
  python analysis/figure_6_7_combine.py 3b                    # Combined plot (default)
  python analysis/figure_6_7_combine.py 3b --mode cir         # CIR-reward only
  python analysis/figure_6_7_combine.py 3b --mode sr          # SR-reward only
  python analysis/figure_6_7_combine.py 1.5b --mode combined  # Combined plot
"""

import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Set font to sans-serif (uses DejaVu Sans as fallback)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']


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
    """Load CIR proxy (cot_importance), accuracy, and verifier accuracy from a teacher_responses json."""
    json_path = os.path.join(teacher_dir, f"step_{step}", f"teacher_responses_step_{step}.json")
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return None

    cot_importance_values = []
    accuracies = []
    verifier_accuracies = []

    for item in data:
        # CIR proxy: use cot_importance_evaluation js_divergences if present
        if "cot_importance_evaluation" in item:
            eval_data = item["cot_importance_evaluation"]
            js_divs = eval_data.get("js_divergences", [])
            if len(js_divs) >= 2:
                # sample at 0%,10%,...,100% and average
                percentages = [0, 10, 20, 30, 35, 50, 60, 70, 80, 90, 100]
                sampled = []
                for p in percentages:
                    if p == 0:
                        idx = 0
                    else:
                        idx = max(0, min(int((p / 100.0) * len(js_divs)) - 1, len(js_divs) - 1))
                    sampled.append(js_divs[idx])
                cot_importance_values.append(float(np.mean(sampled)))

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

            if answers_match and "answer" in with_q and "answer" in without_q:
                verifier_accuracies.append(0.0)
            if answers_match and with_q == "" and without_q == "":
                verifier_accuracies.append(0.0)
            else:
                verifier_accuracies.append(1.0 if answers_match else 0.0)

    if not cot_importance_values and not verifier_accuracies and not accuracies:
        return None

    return {
        "cot_importance": float(np.mean(cot_importance_values)) if cot_importance_values else None,
        "accuracy": float(np.mean(accuracies)) if accuracies else None,
        "verifier_accuracy": float(np.mean(verifier_accuracies)) if verifier_accuracies else None,
    }


def load_sr_reward_data(base_path, tasks, verifier_scales, steps_to_keep, scale0_base_path, collect_individual=False, variant="sr_plus"):
    """Load data for SR-reward training (figure 6).
    variant: 'sr_plus', 'sr_minus', or 'original'
    """
    all_scale_data = []
    task_scale_data = {task: {} for task in tasks} if collect_individual else None

    for scale in verifier_scales:
        print(f"\n=== Processing SR α={scale} ({variant}) ===")
        step_aggregated = {}

        for task in tasks:
            if scale == 0 or scale == 0.0:
                task_dir = f"{task}_cot_importance/-1.0/teacher"
                full_path = os.path.join(scale0_base_path, task_dir)
            elif variant == "sr_minus":
                task_dir = f"cot_verifier_acc_trained_sr_minus_{scale}/{task}/-1.0/teacher"
                full_path = os.path.join(base_path, task_dir)
            elif variant == "original":
                task_dir = f"cot_verifier_acc_trained_{scale}/{task}/-1.0/teacher"
                full_path = os.path.join(base_path, task_dir)
            else:
                task_dir = f"cot_verifier_acc_trained_sr_plus_{scale}/{task}/-1.0/teacher"
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
                metrics = load_metrics_at_step(full_path, step)
                if metrics:
                    step_aggregated.setdefault(step, []).append(metrics)
                    if collect_individual:
                        task_step_data[step] = metrics

            if collect_individual and task_step_data:
                task_scale_data[task][scale] = task_step_data

        averaged_step_data = {}
        for step, metrics_list in step_aggregated.items():
            cot_vals = [m["cot_importance"] for m in metrics_list if m["cot_importance"] is not None]
            acc_vals = [m["accuracy"] for m in metrics_list if m["accuracy"] is not None]
            ver_vals = [m["verifier_accuracy"] for m in metrics_list if m["verifier_accuracy"] is not None]
            averaged_step_data[step] = {
                "cot_importance": float(np.mean(cot_vals)) if cot_vals else None,
                "cot_importance_std": float(np.std(cot_vals) / np.sqrt(len(cot_vals))) if len(cot_vals) > 0 else None,
                "accuracy": float(np.mean(acc_vals)) if acc_vals else None,
                "accuracy_std": float(np.std(acc_vals) / np.sqrt(len(acc_vals))) if len(acc_vals) > 0 else None,
                "verifier_accuracy": float(np.mean(ver_vals)) if ver_vals else None,
                "verifier_accuracy_std": float(np.std(ver_vals) / np.sqrt(len(ver_vals))) if len(ver_vals) > 0 else None
            }

        if averaged_step_data:
            all_scale_data.append((f"α={scale}", averaged_step_data))
            print(f"  Successfully averaged {len(averaged_step_data)} steps across tasks")
        else:
            print(f"  Warning: No valid data for β={scale}")

    if collect_individual:
        return all_scale_data, task_scale_data
    return all_scale_data


def load_cir_reward_data(base_path, tasks, coefficients, steps_to_keep, coeff0_base_path, collect_individual=False, use_sr_minus=False):
    """Load data for CIR-reward training (figure 7)."""
    all_coeff_data = []
    task_coeff_data = {task: {} for task in tasks} if collect_individual else None
    variant_label = "sr_minus" if use_sr_minus else "sr_plus"

    for coeff in coefficients:
        print(f"\n=== Processing CIR β={coeff} ({variant_label}) ===")
        step_aggregated = {}

        for task in tasks:
            if coeff == 0.0:
                task_dir = f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-3b-instruct/4o_mini_cot_importance/{task}_cot_importance/-1.0/teacher"
                full_path = os.path.join(coeff0_base_path, task_dir)
            elif use_sr_minus:
                task_dir = f"cot_importance_trained_sr_minus_{coeff}/{task}/-1.0/teacher"
                full_path = os.path.join(base_path, task_dir)
            else:
                # Try both naming variants (some have underscore before number, some don't)
                task_dir = f"cot_importance_trained_sr_plus_{coeff}/{task}/-1.0/teacher"
                full_path = os.path.join(base_path, task_dir)
                if not os.path.exists(full_path):
                    task_dir = f"cot_importance_trained_sr_plus{coeff}/{task}/-1.0/teacher"
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
                metrics = load_metrics_at_step(full_path, step)
                if metrics:
                    step_aggregated.setdefault(step, []).append(metrics)
                    if collect_individual:
                        task_step_data[step] = metrics

            if collect_individual and task_step_data:
                task_coeff_data[task][coeff] = task_step_data

        averaged_step_data = {}
        for step, metrics_list in step_aggregated.items():
            cot_vals = [m["cot_importance"] for m in metrics_list if m["cot_importance"] is not None]
            acc_vals = [m["accuracy"] for m in metrics_list if m["accuracy"] is not None]
            ver_vals = [m["verifier_accuracy"] for m in metrics_list if m["verifier_accuracy"] is not None]
            averaged_step_data[step] = {
                "cot_importance": float(np.mean(cot_vals)) if cot_vals else None,
                "cot_importance_std": float(np.std(cot_vals) / np.sqrt(len(cot_vals))) if len(cot_vals) > 0 else None,
                "accuracy": float(np.mean(acc_vals)) if acc_vals else None,
                "accuracy_std": float(np.std(acc_vals) / np.sqrt(len(acc_vals))) if len(acc_vals) > 0 else None,
                "verifier_accuracy": float(np.mean(ver_vals)) if ver_vals else None,
                "verifier_accuracy_std": float(np.std(ver_vals) / np.sqrt(len(ver_vals))) if len(ver_vals) > 0 else None
            }

        if averaged_step_data:
            all_coeff_data.append((f"β={coeff}", averaged_step_data))
            print(f"  Successfully averaged {len(averaged_step_data)} steps across tasks")
        else:
            print(f"  Warning: No valid data for α={coeff}")

    if collect_individual:
        return all_coeff_data, task_coeff_data
    return all_coeff_data


def _format_ax(ax, ylabel, title, has_legend=True, show_xlabel=False, custom_legend=False):
    """Apply compact formatting to an axis matching the reference style."""
    ax.set_ylabel(ylabel, fontsize=25, fontweight="normal", family="sans-serif")
    ax.set_title(title, fontsize=25, fontweight="normal", pad=8, family="sans-serif")
    ax.tick_params(axis="both", which="major", labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family("serif")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    if show_xlabel:
        ax.set_xlabel("RL training step", fontsize=40, fontweight="normal", family="sans-serif")
    if has_legend and not custom_legend:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9, prop={"family": "serif"})


def _lighten_color(color, amount=0.4):
    """Lighten a hex color by blending with white."""
    import matplotlib.colors as mcolors
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1 - c) * amount)


def _plot_lines(ax, data, metric_key, colors, linestyle='-', label_suffix="", lighten=False, skip_zero=False):
    """Plot lines for each scale/coeff on the given axis."""
    for idx, (name, step_data) in enumerate(data):
        # Skip the α=0.0 / β=0.0 entry if requested (e.g. for dashed SR- lines)
        if skip_zero and idx == 0 and "=0.0" in name:
            continue
        steps = []
        values = []
        for step in sorted(step_data.keys()):
            if step_data[step][metric_key] is not None:
                steps.append(step)
                values.append(step_data[step][metric_key])
        if steps and values:
            color = colors[idx % len(colors)]
            if lighten:
                color = _lighten_color(color)
            label = f"{name}{label_suffix}" if label_suffix else name
            lw = 1.2 if linestyle == '--' else 2.0
            ax.plot(steps, values, linewidth=lw, color=color,
                    marker='o', markersize=4, label=label, alpha=0.9, linestyle=linestyle)


def plot_cir_only(cir_data, cir_tasks, output_dir, model_size):
    """Plot only CIR-reward training (3 subplots: CIR, SR, Accuracy)."""
    colors = ["#2E2E2E", "#1E88E5", "#FFA726", "#66BB6A", "#EF5350", "#AB47BC"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.5))

    _plot_lines(ax1, cir_data, "cot_importance", colors)
    _format_ax(ax1, "CIR", "CIR reward \u2192 CIR", show_xlabel=True)

    _plot_lines(ax2, cir_data, "verifier_accuracy", colors)
    _format_ax(ax2, "SR", "CIR reward \u2192 SR", show_xlabel=True)

    _plot_lines(ax3, cir_data, "accuracy", colors)
    _format_ax(ax3, "Accuracy", "CIR reward \u2192 Accuracy", show_xlabel=True)

    plt.tight_layout()

    png_path = os.path.join(output_dir, f"figure7_cir_reward_{model_size}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    pdf_path = os.path.join(output_dir, f"figure7_cir_reward_{model_size}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()
    return png_path, pdf_path


def plot_sr_only(sr_data, sr_tasks, output_dir, model_size):
    """Plot only SR-reward training (3 subplots: CIR, SR, Accuracy)."""
    colors = ["#2E2E2E", "#1E88E5", "#FFA726", "#66BB6A", "#EF5350", "#AB47BC"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.5))

    _plot_lines(ax1, sr_data, "cot_importance", colors)
    _format_ax(ax1, "CIR", "SR reward \u2192 CIR", show_xlabel=True)

    _plot_lines(ax2, sr_data, "verifier_accuracy", colors)
    _format_ax(ax2, "SR", "SR reward \u2192 SR", show_xlabel=True)

    _plot_lines(ax3, sr_data, "accuracy", colors)
    _format_ax(ax3, "Accuracy", "SR reward \u2192 Accuracy", show_xlabel=True)

    plt.tight_layout()

    png_path = os.path.join(output_dir, f"figure6_sr_reward_{model_size}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    pdf_path = os.path.join(output_dir, f"figure6_sr_reward_{model_size}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()
    return png_path, pdf_path


def _add_sr_legend(ax, data, colors):
    """Add a custom legend with one entry per alpha, plus SR+/SR- indicators."""
    from matplotlib.lines import Line2D
    handles = []
    # One entry per alpha value
    for idx, (name, _) in enumerate(data):
        color = colors[idx % len(colors)]
        handles.append(Line2D([0], [0], color=color, linewidth=1.5, marker='o',
                              markersize=3, linestyle='-', label=name))
    # Add SR+ and SR- indicators
    handles.append(Line2D([0], [0], color='black', linewidth=1.5, linestyle='-', label='SR'))
    handles.append(Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', label='SR\u2212'))
    ax.legend(handles=handles, fontsize=6, loc="upper left", framealpha=0.9,
              prop={"family": "serif"}, ncol=1)


def plot_combined(sr_data, cir_data, sr_tasks, cir_tasks, output_dir, model_size,
                  sr_minus_sr_data=None, sr_minus_cir_data=None,
                  sr_original_data=None):
    """Plot combined figure with SR-reward (top row) and CIR-reward (bottom row) as subplots."""
    colors = ["#2E2E2E", "#1E88E5", "#FFA726", "#66BB6A", "#EF5350", "#AB47BC"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    # Use original data for CIR/Accuracy columns if available, sr_plus for SR column
    sr_cir_acc_data = sr_original_data if sr_original_data else sr_data

    # TOP ROW: SR-reward training
    _plot_lines(axes[0, 0], sr_cir_acc_data, "cot_importance", colors)
    _format_ax(axes[0, 0], "CIR", "SR reward \u2192 CIR")

    _plot_lines(axes[0, 1], sr_data, "verifier_accuracy", colors)
    if sr_minus_sr_data:
        _plot_lines(axes[0, 1], sr_minus_sr_data, "verifier_accuracy", colors, linestyle='--', lighten=True, skip_zero=True)
    _format_ax(axes[0, 1], "SR", "SR reward \u2192 SR", custom_legend=True)
    _add_sr_legend(axes[0, 1], sr_data, colors)

    _plot_lines(axes[0, 2], sr_cir_acc_data, "accuracy", colors)
    _format_ax(axes[0, 2], "Accuracy", "SR reward \u2192 Accuracy")

    # BOTTOM ROW: CIR-reward training
    _plot_lines(axes[1, 0], cir_data, "cot_importance", colors)
    _format_ax(axes[1, 0], "CIR", "CIR reward \u2192 CIR")

    _plot_lines(axes[1, 1], cir_data, "verifier_accuracy", colors)
    if sr_minus_cir_data:
        _plot_lines(axes[1, 1], sr_minus_cir_data, "verifier_accuracy", colors, linestyle='--', lighten=True, skip_zero=True)
    _format_ax(axes[1, 1], "SR", "CIR reward \u2192 SR", custom_legend=True)
    _add_sr_legend(axes[1, 1], cir_data, colors)

    _plot_lines(axes[1, 2], cir_data, "accuracy", colors)
    _format_ax(axes[1, 2], "Accuracy", "CIR reward \u2192 Accuracy")

    # Single shared x-axis label at the bottom center
    fig.text(0.5, -0.02, "RL training step", ha="center", fontsize=20, fontweight="normal", family="sans-serif")

    plt.tight_layout()

    png_path = os.path.join(output_dir, f"figure6_7_combine_{model_size}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    pdf_path = os.path.join(output_dir, f"figure6_7_combine_{model_size}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()
    return png_path, pdf_path


def plot_individual_tasks_sr(task_scale_data, output_dir, tasks, scales, model_size):
    """Plot individual task metrics for SR-reward training in a grid layout."""
    num_tasks = len(tasks)
    fig_height = max(12, num_tasks * 4)

    fig, axes = plt.subplots(num_tasks, 3, figsize=(24, fig_height))

    # Handle case where there's only one task
    if num_tasks == 1:
        axes = axes.reshape(1, -1)

    # More visually distinct color palette (same as main plot)
    colors = ["#2E2E2E", "#1E88E5", "#FFA726", "#66BB6A", "#EF5350", "#AB47BC"]
    markers = ["o", "o", "o", "o", "o", "o"]

    for task_idx, task in enumerate(tasks):
        # Plot 1: CoT Importance (CIR)
        ax_cir = axes[task_idx, 0]
        for scale_idx, scale in enumerate(scales):
            if task in task_scale_data and scale in task_scale_data[task]:
                step_data = task_scale_data[task][scale]
                steps = []
                cot_values = []
                for step in sorted(step_data.keys()):
                    if step_data[step]["cot_importance"] is not None:
                        steps.append(step)
                        cot_values.append(step_data[step]["cot_importance"])

                if steps and cot_values:
                    color = colors[scale_idx % len(colors)]
                    marker = markers[scale_idx % len(markers)]
                    ax_cir.plot(steps, cot_values, linewidth=3.5, color=color, marker=marker,
                               markersize=8, label=f'α={scale}', alpha=0.85, markeredgewidth=1.2,
                               markeredgecolor='white')

        ax_cir.set_ylabel('CIR', fontsize=20, fontweight='normal', family='serif')
        ax_cir.set_title(f'{task.replace("_", " ").title()}', fontsize=35, fontweight='normal', family='serif', pad=15)
        ax_cir.tick_params(axis='both', which='major', labelsize=16, width=1.2)
        ax_cir.tick_params(axis='both', which='minor', labelsize=14, width=0.8)
        for label in ax_cir.get_xticklabels() + ax_cir.get_yticklabels():
            label.set_family('serif')
        ax_cir.set_ylim(-0.02, 1.02)
        ax_cir.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_cir.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
        ax_cir.minorticks_on()
        if task_idx == 0:
            ax_cir.legend(fontsize=52, loc='best', framealpha=0.95, prop={'family': 'Helvetica'},
                         edgecolor='gray', fancybox=True, shadow=True, markerscale=1)
        ax_cir.set_axisbelow(True)
        for spine in ax_cir.spines.values():
            spine.set_linewidth(1.2)
        if task_idx == len(tasks) - 1:
            ax_cir.set_xlabel('RL Training Step', fontsize=30, fontweight='normal', family='serif')

        # Plot 2: Verifier Accuracy (SR)
        ax_sr = axes[task_idx, 1]
        for scale_idx, scale in enumerate(scales):
            if task in task_scale_data and scale in task_scale_data[task]:
                step_data = task_scale_data[task][scale]
                steps = []
                verifier_values = []
                for step in sorted(step_data.keys()):
                    if step_data[step]["verifier_accuracy"] is not None:
                        steps.append(step)
                        verifier_values.append(step_data[step]["verifier_accuracy"])

                if steps and verifier_values:
                    color = colors[scale_idx % len(colors)]
                    marker = markers[scale_idx % len(markers)]
                    ax_sr.plot(steps, verifier_values, linewidth=3.5, color=color, marker=marker,
                             markersize=8, label=f'α={scale}', alpha=0.85, markeredgewidth=1.2,
                             markeredgecolor='white')

        ax_sr.set_ylabel('SR', fontsize=20, fontweight='normal', family='serif')
        ax_sr.tick_params(axis='both', which='major', labelsize=16, width=1.2)
        ax_sr.tick_params(axis='both', which='minor', labelsize=14, width=0.8)
        for label in ax_sr.get_xticklabels() + ax_sr.get_yticklabels():
            label.set_family('serif')
        ax_sr.set_ylim(-0.02, 1.02)
        ax_sr.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_sr.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
        ax_sr.minorticks_on()
        ax_sr.set_axisbelow(True)
        for spine in ax_sr.spines.values():
            spine.set_linewidth(1.2)
        if task_idx == len(tasks) - 1:
            ax_sr.set_xlabel('RL Training Step', fontsize=30, fontweight='normal', family='serif')

        # Plot 3: Accuracy
        ax_acc = axes[task_idx, 2]
        for scale_idx, scale in enumerate(scales):
            if task in task_scale_data and scale in task_scale_data[task]:
                step_data = task_scale_data[task][scale]
                steps = []
                accuracy_values = []
                for step in sorted(step_data.keys()):
                    if step_data[step]["accuracy"] is not None:
                        steps.append(step)
                        accuracy_values.append(step_data[step]["accuracy"])

                if steps and accuracy_values:
                    color = colors[scale_idx % len(colors)]
                    marker = markers[scale_idx % len(markers)]
                    ax_acc.plot(steps, accuracy_values, linewidth=3.5, color=color, marker=marker,
                              markersize=8, label=f'α={scale}', alpha=0.85, markeredgewidth=1.2,
                              markeredgecolor='white')

        ax_acc.set_ylabel('Accuracy', fontsize=20, fontweight='normal', family='serif')
        ax_acc.tick_params(axis='both', which='major', labelsize=16, width=1.2)
        ax_acc.tick_params(axis='both', which='minor', labelsize=14, width=0.8)
        for label in ax_acc.get_xticklabels() + ax_acc.get_yticklabels():
            label.set_family('serif')
        ax_acc.set_ylim(-0.02, 1.02)
        ax_acc.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_acc.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
        ax_acc.minorticks_on()
        ax_acc.set_axisbelow(True)
        for spine in ax_acc.spines.values():
            spine.set_linewidth(1.2)
        if task_idx == len(tasks) - 1:
            ax_acc.set_xlabel('RL Training Step', fontsize=20, fontweight='normal', family='serif')

    plt.tight_layout()

    png_path = os.path.join(output_dir, f'figure6_sr_reward_individual_{model_size}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, f'figure6_sr_reward_individual_{model_size}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.close()

    return png_path, pdf_path


def plot_individual_tasks_cir(task_coeff_data, output_dir, tasks, coefficients, model_size):
    """Plot individual task metrics for CIR-reward training in a grid layout."""
    num_tasks = len(tasks)
    fig_height = max(12, num_tasks * 4)

    fig, axes = plt.subplots(num_tasks, 3, figsize=(24, fig_height))

    # Handle case where there's only one task
    if num_tasks == 1:
        axes = axes.reshape(1, -1)

    # More visually distinct color palette (same as main plot)
    colors = ["#2E2E2E", "#1E88E5", "#FFA726", "#66BB6A", "#EF5350", "#AB47BC"]
    markers = ["o", "o", "o", "o", "o", "o"]

    for task_idx, task in enumerate(tasks):
        # Plot 1: CoT Importance (CIR)
        ax_cir = axes[task_idx, 0]
        for coeff_idx, coeff in enumerate(coefficients):
            if task in task_coeff_data and coeff in task_coeff_data[task]:
                step_data = task_coeff_data[task][coeff]
                steps = []
                cot_values = []
                for step in sorted(step_data.keys()):
                    if step_data[step]["cot_importance"] is not None:
                        steps.append(step)
                        cot_values.append(step_data[step]["cot_importance"])

                if steps and cot_values:
                    color = colors[coeff_idx % len(colors)]
                    marker = markers[coeff_idx % len(markers)]
                    ax_cir.plot(steps, cot_values, linewidth=3.5, color=color, marker=marker,
                               markersize=8, label=f'β={coeff}', alpha=0.85, markeredgewidth=1.2,
                               markeredgecolor='white')

        ax_cir.set_ylabel('CIR', fontsize=20, fontweight='normal', family='serif')
        ax_cir.set_title(f'{task.replace("_", " ").title()}', fontsize=35, fontweight='normal', family='serif', pad=15)
        ax_cir.tick_params(axis='both', which='major', labelsize=16, width=1.2)
        ax_cir.tick_params(axis='both', which='minor', labelsize=14, width=0.8)
        for label in ax_cir.get_xticklabels() + ax_cir.get_yticklabels():
            label.set_family('serif')
        ax_cir.set_ylim(-0.02, 1.02)
        ax_cir.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_cir.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
        ax_cir.minorticks_on()
        if task_idx == 0:
            ax_cir.legend(fontsize=52, loc='best', framealpha=0.95, prop={'family': 'Helvetica'},
                         edgecolor='gray', fancybox=True, shadow=True, markerscale=1)
        ax_cir.set_axisbelow(True)
        for spine in ax_cir.spines.values():
            spine.set_linewidth(1.2)
        if task_idx == len(tasks) - 1:
            ax_cir.set_xlabel('RL Training Step', fontsize=30, fontweight='normal', family='serif')

        # Plot 2: Verifier Accuracy (SR)
        ax_sr = axes[task_idx, 1]
        for coeff_idx, coeff in enumerate(coefficients):
            if task in task_coeff_data and coeff in task_coeff_data[task]:
                step_data = task_coeff_data[task][coeff]
                steps = []
                verifier_values = []
                for step in sorted(step_data.keys()):
                    if step_data[step]["verifier_accuracy"] is not None:
                        steps.append(step)
                        verifier_values.append(step_data[step]["verifier_accuracy"])

                if steps and verifier_values:
                    color = colors[coeff_idx % len(colors)]
                    marker = markers[coeff_idx % len(markers)]
                    ax_sr.plot(steps, verifier_values, linewidth=3.5, color=color, marker=marker,
                             markersize=8, label=f'β={coeff}', alpha=0.85, markeredgewidth=1.2,
                             markeredgecolor='white')

        ax_sr.set_ylabel('SR', fontsize=20, fontweight='normal', family='serif')
        ax_sr.tick_params(axis='both', which='major', labelsize=16, width=1.2)
        ax_sr.tick_params(axis='both', which='minor', labelsize=14, width=0.8)
        for label in ax_sr.get_xticklabels() + ax_sr.get_yticklabels():
            label.set_family('serif')
        ax_sr.set_ylim(-0.02, 1.02)
        ax_sr.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_sr.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
        ax_sr.minorticks_on()
        ax_sr.set_axisbelow(True)
        for spine in ax_sr.spines.values():
            spine.set_linewidth(1.2)
        if task_idx == len(tasks) - 1:
            ax_sr.set_xlabel('RL Training Step', fontsize=30, fontweight='normal', family='serif')

        # Plot 3: Accuracy
        ax_acc = axes[task_idx, 2]
        for coeff_idx, coeff in enumerate(coefficients):
            if task in task_coeff_data and coeff in task_coeff_data[task]:
                step_data = task_coeff_data[task][coeff]
                steps = []
                accuracy_values = []
                for step in sorted(step_data.keys()):
                    if step_data[step]["accuracy"] is not None:
                        steps.append(step)
                        accuracy_values.append(step_data[step]["accuracy"])

                if steps and accuracy_values:
                    color = colors[coeff_idx % len(colors)]
                    marker = markers[coeff_idx % len(markers)]
                    ax_acc.plot(steps, accuracy_values, linewidth=3.5, color=color, marker=marker,
                              markersize=8, label=f'β={coeff}', alpha=0.85, markeredgewidth=1.2,
                              markeredgecolor='white')

        ax_acc.set_ylabel('Accuracy', fontsize=20, fontweight='normal', family='serif')
        ax_acc.tick_params(axis='both', which='major', labelsize=16, width=1.2)
        ax_acc.tick_params(axis='both', which='minor', labelsize=14, width=0.8)
        for label in ax_acc.get_xticklabels() + ax_acc.get_yticklabels():
            label.set_family('serif')
        ax_acc.set_ylim(-0.02, 1.02)
        ax_acc.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_acc.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
        ax_acc.minorticks_on()
        ax_acc.set_axisbelow(True)
        for spine in ax_acc.spines.values():
            spine.set_linewidth(1.2)
        if task_idx == len(tasks) - 1:
            ax_acc.set_xlabel('RL Training Step', fontsize=30, fontweight='normal', family='serif')

    plt.tight_layout()

    png_path = os.path.join(output_dir, f'figure7_cir_reward_individual_{model_size}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, f'figure7_cir_reward_individual_{model_size}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.close()

    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description='Plot combined CIR and SR for both SR-reward and CIR-reward training')
    parser.add_argument('model_size', nargs='?', default='3b', help='Model size (default: 3b)')
    parser.add_argument('--mode', type=str, choices=['combined', 'cir', 'sr'], default='combined',
                       help='Plot mode: combined (both SR and CIR), cir (CIR-reward only), sr (SR-reward only)')
    args = parser.parse_args()

    model_size = args.model_size

    # SR-reward training data (Figure 6)
    sr_tasks = ["binary_matrix", "bitwise_arithmetic", "count_bits", "manipulate_matrix", "futoshiki", "mini_sudoku", "rotate_matrix", "tsumego"]
    verifier_scales = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # CIR-reward training data (Figure 7)
    cir_tasks = ["binary_matrix", "bitwise_arithmetic", "count_bits", "manipulate_matrix", "futoshiki", "mini_sudoku", "rotate_matrix", "tsumego"]
    coefficients = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    steps_to_keep = [2, 30, 60, 90, 120, 150, 156]

    base_path = f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct"
    scale0_base_path = f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct/cot_importance"
    coeff0_base_path = f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct/cot_importance"

    print(f"\nProcessing model: qwen-{model_size}-instruct")
    print(f"Mode: {args.mode}")

    output_dir = "/nlp/scr/qinanyu/rl-explanations/analysis/graph"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    if args.mode == 'cir':
        # Only load and plot CIR-reward data
        print("\n=== Loading CIR-reward data ===")
        cir_data, task_coeff_data = load_cir_reward_data(base_path, cir_tasks, coefficients, steps_to_keep, coeff0_base_path, collect_individual=True)

        if not cir_data:
            print("\nError: Missing data for CIR training")
            return

        # Generate CIR-reward plot only
        print("\n=== Generating CIR-reward plot ===")
        png_path, pdf_path = plot_cir_only(cir_data, cir_tasks, output_dir, model_size)
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

        # Generate individual task plot for CIR-reward
        print("\n=== Generating individual task plots for CIR-reward ===")
        cir_ind_png, cir_ind_pdf = plot_individual_tasks_cir(task_coeff_data, output_dir, cir_tasks, coefficients, model_size)
        print(f"Saved: {cir_ind_png}")
        print(f"Saved: {cir_ind_pdf}")

    elif args.mode == 'sr':
        # Only load and plot SR-reward data
        print("\n=== Loading SR-reward data ===")
        sr_data, task_scale_data = load_sr_reward_data(base_path, sr_tasks, verifier_scales, steps_to_keep, scale0_base_path, collect_individual=True)

        if not sr_data:
            print("\nError: Missing data for SR training")
            return

        # Generate SR-reward plot only
        print("\n=== Generating SR-reward plot ===")
        png_path, pdf_path = plot_sr_only(sr_data, sr_tasks, output_dir, model_size)
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

        # Generate individual task plot for SR-reward
        print("\n=== Generating individual task plots for SR-reward ===")
        sr_ind_png, sr_ind_pdf = plot_individual_tasks_sr(task_scale_data, output_dir, sr_tasks, verifier_scales, model_size)
        print(f"Saved: {sr_ind_png}")
        print(f"Saved: {sr_ind_pdf}")

    else:  # args.mode == 'combined'
        # Load SR-reward data: sr_plus for SR column, original for CIR/Accuracy columns
        print("\n=== Loading SR-reward data (sr_plus, for SR column) ===")
        sr_data, task_scale_data = load_sr_reward_data(base_path, sr_tasks, verifier_scales, steps_to_keep, scale0_base_path, collect_individual=True)

        print("\n=== Loading SR-reward data (original, for CIR/Accuracy columns) ===")
        sr_original_data = load_sr_reward_data(base_path, sr_tasks, verifier_scales, steps_to_keep, scale0_base_path, variant="original")

        print("\n=== Loading CIR-reward data (sr_plus) ===")
        cir_data, task_coeff_data = load_cir_reward_data(base_path, cir_tasks, coefficients, steps_to_keep, coeff0_base_path, collect_individual=True)

        # Load sr_minus data for dashed lines in SR column
        print("\n=== Loading SR-reward data (sr_minus) ===")
        sr_minus_sr_data = load_sr_reward_data(base_path, sr_tasks, verifier_scales, steps_to_keep, scale0_base_path, variant="sr_minus")

        print("\n=== Loading CIR-reward data (sr_minus) ===")
        sr_minus_cir_data = load_cir_reward_data(base_path, cir_tasks, coefficients, steps_to_keep, coeff0_base_path, use_sr_minus=True)

        if not sr_data or not cir_data:
            print("\nError: Missing data for SR or CIR training")
            return

        # Generate combined plot (Figure 6 + 7)
        print("\n=== Generating combined plot ===")
        png_path, pdf_path = plot_combined(sr_data, cir_data, sr_tasks, cir_tasks, output_dir, model_size,
                                           sr_minus_sr_data=sr_minus_sr_data, sr_minus_cir_data=sr_minus_cir_data,
                                           sr_original_data=sr_original_data)
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

        # Generate individual task plot for SR-reward (Figure 6 individual)
        print("\n=== Generating individual task plots for SR-reward ===")
        sr_ind_png, sr_ind_pdf = plot_individual_tasks_sr(task_scale_data, output_dir, sr_tasks, verifier_scales, model_size)
        print(f"Saved: {sr_ind_png}")
        print(f"Saved: {sr_ind_pdf}")

        # Generate individual task plot for CIR-reward (Figure 7 individual)
        print("\n=== Generating individual task plots for CIR-reward ===")
        cir_ind_png, cir_ind_pdf = plot_individual_tasks_cir(task_coeff_data, output_dir, cir_tasks, coefficients, model_size)
        print(f"Saved: {cir_ind_png}")
        print(f"Saved: {cir_ind_pdf}")


if __name__ == "__main__":
    main()
