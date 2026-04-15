#!/usr/bin/env python3
"""
Analyze CIR and SR changes across training steps for SR-reward training, sweeping verifier scales.

Data source (for β in {0.5, 1.0, ...}):
  /nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-3b-instruct/cot_verifier_acc_trained_{beta}

Usage:
  python analysis/figure_6_sr_reward_trained.py
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Set font to serif for consistency
plt.rcParams['font.family'] = 'serif'


def get_available_steps(teacher_dir: str):
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


def load_metrics_at_step(teacher_dir: str, step: int):
    """Load CIR proxy (cot_importance), accuracy, and verifier accuracy from a teacher_responses json."""
    json_path = os.path.join(teacher_dir, f"step_{step}", f"teacher_responses_step_{step}.json")
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r") as f:
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
                # sample at 0%,10%,...,100% and average (same as other scripts)
                percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
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

        # SR: verifier accuracy (answers_match with "no answer found" special-case, same as figure_6)
        if "verifier_comparison" in item:
            verifier_comp = item["verifier_comparison"]
            answers_match = verifier_comp.get("answers_match", False)

            with_q = verifier_comp.get("with_question_answer", "").strip().lower()
            without_q = verifier_comp.get("without_question_answer", "").strip().lower()

            if answers_match and with_q == "no answer found" and without_q == "no answer found":
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


def plot_metrics_progression(all_scale_data, output_dir: str, tasks: list[str]):
    """Plot CIR, SR, and Accuracy across training steps for multiple verifier scales."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D"]

    # Plot 1: CIR
    for idx, (scale_name, step_data) in enumerate(all_scale_data):
        steps = []
        cot_values = []
        for step in sorted(step_data.keys()):
            if step_data[step]["cot_importance"] is not None:
                steps.append(step)
                cot_values.append(step_data[step]["cot_importance"])

        if steps and cot_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax1.plot(steps, cot_values, linewidth=3.0, color=color, marker=marker,
                     markersize=8, label=scale_name, alpha=0.9)

    ax1.set_xlabel("Training Step", fontsize=25, fontweight="bold", family="serif")
    ax1.set_ylabel("CIR", fontsize=25, fontweight="bold", family="serif")
    ax1.set_title(f"CIR Across Training (SR as reward; n={len(tasks)} tasks)",
                  fontsize=25, fontweight="bold", pad=20, family="serif")
    ax1.tick_params(axis="both", which="major", labelsize=19)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_family("serif")
    ax1.set_ylim(0, 1.0)
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    ax1.legend(fontsize=19, loc="best", framealpha=0.9, prop={"family": "serif"})
    ax1.set_axisbelow(True)

    # Plot 2: SR
    for idx, (scale_name, step_data) in enumerate(all_scale_data):
        steps = []
        verifier_values = []
        for step in sorted(step_data.keys()):
            if step_data[step]["verifier_accuracy"] is not None:
                steps.append(step)
                verifier_values.append(step_data[step]["verifier_accuracy"])

        if steps and verifier_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax2.plot(steps, verifier_values, linewidth=3.0, color=color, marker=marker,
                     markersize=8, label=scale_name, alpha=0.9)

    ax2.set_xlabel("Training Step", fontsize=25, fontweight="bold", family="serif")
    ax2.set_ylabel("SR", fontsize=25, fontweight="bold", family="serif")
    ax2.set_title("SR Across Training (SR as reward)",
                  fontsize=25, fontweight="bold", pad=20, family="serif")
    ax2.tick_params(axis="both", which="major", labelsize=19)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_family("serif")
    ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    ax2.legend(fontsize=19, loc="best", framealpha=0.9, prop={"family": "serif"})
    ax2.set_axisbelow(True)

    # Plot 3: Accuracy
    for idx, (scale_name, step_data) in enumerate(all_scale_data):
        steps = []
        accuracy_values = []
        for step in sorted(step_data.keys()):
            if step_data[step]["accuracy"] is not None:
                steps.append(step)
                accuracy_values.append(step_data[step]["accuracy"])

        if steps and accuracy_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax3.plot(steps, accuracy_values, linewidth=3.0, color=color, marker=marker,
                     markersize=8, label=scale_name, alpha=0.9)

    ax3.set_xlabel("Training Step", fontsize=25, fontweight="bold", family="serif")
    ax3.set_ylabel("Accuracy", fontsize=25, fontweight="bold", family="serif")
    ax3.set_title("Accuracy Across Training (SR as reward)",
                  fontsize=25, fontweight="bold", pad=20, family="serif")
    ax3.tick_params(axis="both", which="major", labelsize=19)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_family("serif")
    ax3.set_ylim(0, 1.0)
    ax3.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    ax3.legend(fontsize=19, loc="best", framealpha=0.9, prop={"family": "serif"})
    ax3.set_axisbelow(True)

    plt.tight_layout()

    png_path = os.path.join(output_dir, "figure6_sr_reward_trained.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    pdf_path = os.path.join(output_dir, "figure6_sr_reward_trained.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()
    return png_path, pdf_path


def main():
    tasks = ["count_bits", "manipulate_matrix"]
    verifier_scales = [0, 1.0]
    steps_to_keep = [2, 30, 60, 90, 120, 150, 156]

    base_path = "/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-3b-instruct"
    scale0_base_path = "/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-3b-instruct/cot_importance"

    all_scale_data = []

    for scale in verifier_scales:
        print(f"\n=== Processing β={scale} ===")
        step_aggregated = {}

        for task in tasks:
            if scale == 0 or scale == 0.0:
                task_dir = f"{task}_cot_importance/-1.0/teacher"
                full_path = os.path.join(scale0_base_path, task_dir)
            else:
                task_dir = f"cot_verifier_acc_trained_{scale}/{task}/-1.0/teacher"
                full_path = os.path.join(base_path, task_dir)

            if not os.path.exists(full_path):
                print(f"  Warning: Path not found for {task}: {full_path}")
                continue

            training_steps = get_available_steps(full_path)
            if not training_steps:
                print(f"  Warning: No training steps found for {task}")
                continue

            for step in training_steps:
                if step not in steps_to_keep:
                    continue
                metrics = load_metrics_at_step(full_path, step)
                if metrics:
                    step_aggregated.setdefault(step, []).append(metrics)

        averaged_step_data = {}
        for step, metrics_list in step_aggregated.items():
            cot_vals = [m["cot_importance"] for m in metrics_list if m["cot_importance"] is not None]
            acc_vals = [m["accuracy"] for m in metrics_list if m["accuracy"] is not None]
            ver_vals = [m["verifier_accuracy"] for m in metrics_list if m["verifier_accuracy"] is not None]
            averaged_step_data[step] = {
                "cot_importance": float(np.mean(cot_vals)) if cot_vals else None,
                "accuracy": float(np.mean(acc_vals)) if acc_vals else None,
                "verifier_accuracy": float(np.mean(ver_vals)) if ver_vals else None
            }

        if averaged_step_data:
            all_scale_data.append((f"β={scale}", averaged_step_data))
            print(f"  Successfully averaged {len(averaged_step_data)} steps across tasks")
        else:
            print(f"  Warning: No valid data for β={scale}")

    if not all_scale_data:
        print("\nError: No valid data loaded")
        return

    output_dir = "/nlp/scr/qinanyu/rl-explanations/analysis/graph"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    png_path, pdf_path = plot_metrics_progression(all_scale_data, output_dir, tasks)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

