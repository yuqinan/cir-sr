#!/usr/bin/env python3
"""
Rebuttal figure: CIR-reward α=0.8 across multiple seeds vs original (α=0).

Shows CIR, SR, and Accuracy over training steps for:
  - Original (no augmentation): 4o_mini_cot_importance
  - Seed 0 (α=0.8): cot_importance_trained_0.8_old_sr
  - Seed 1 (α=0.8): cot_importance_trained_seed1_0.8

Mean ± std across the same 10 tasks used in figure_6_7_combine.py.

Usage: python analysis/figure_rebuttal_seeds.py [model_size]
"""

import json
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

TASKS = [
    "binary_matrix", "bitwise_arithmetic", "count_bits", "futoshiki"
    "manipulate_matrix",  "mini_sudoku", "rotate_matrix","tsumego"
]

STEPS_TO_KEEP = {2, 30, 60, 90, 120, 156}
PERCENTAGES = [0, 10, 20, 30, 35, 50, 60, 70, 80, 90, 100]


def get_available_steps(teacher_dir):
    if not os.path.exists(teacher_dir):
        return []
    steps = []
    for item in os.listdir(teacher_dir):
        if item.startswith('step_'):
            m = re.search(r'step_(\d+)', item)
            if m:
                steps.append(int(m.group(1)))
    return sorted(steps)


def load_metrics_at_step(teacher_dir, step):
    """Same logic as figure_6_7_combine.py."""
    json_path = os.path.join(teacher_dir, f"step_{step}", f"teacher_responses_step_{step}.json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return None

    cot_vals, acc_vals, ver_vals = [], [], []

    for item in data:
        if "cot_importance_evaluation" in item:
            js_divs = item["cot_importance_evaluation"].get("js_divergences", [])
            if len(js_divs) >= 2:
                sampled = []
                for p in PERCENTAGES:
                    idx = 0 if p == 0 else max(0, min(int((p / 100.0) * len(js_divs)) - 1, len(js_divs) - 1))
                    sampled.append(js_divs[idx])
                cot_vals.append(float(np.mean(sampled)))

        if "k_responses" in item:
            k_scores = [kr.get("reward_score") for kr in item["k_responses"] if "reward_score" in kr]
            if k_scores:
                acc_vals.append(float(np.mean(k_scores)))

        if "verifier_comparison" in item:
            vc = item["verifier_comparison"]
            answers_match = vc.get("answers_match", False)
            with_q = vc.get("with_question_answer", "").strip().lower()
            without_q = vc.get("without_question_answer", "").strip().lower()
            if answers_match and "answer" in with_q and "answer" in without_q:
                ver_vals.append(0.0)
            elif answers_match and with_q == "" and without_q == "":
                ver_vals.append(0.0)
            else:
                ver_vals.append(1.0 if answers_match else 0.0)

    if not cot_vals and not acc_vals and not ver_vals:
        return None

    return {
        "cot_importance": float(np.mean(cot_vals)) if cot_vals else None,
        "accuracy":       float(np.mean(acc_vals)) if acc_vals else None,
        "verifier_accuracy": float(np.mean(ver_vals)) if ver_vals else None,
    }


def load_series(teacher_dir_fn, label):
    """
    Load mean ± std across tasks for each step.
    teacher_dir_fn(task) -> path to teacher dir for that task.
    Returns (label, {step: {mean, std, n} for each metric}).
    """
    step_data = {}  # step -> list of per-task metrics dicts

    for task in TASKS:
        teacher_dir = teacher_dir_fn(task)
        steps = get_available_steps(teacher_dir)
        for step in steps:
            if step not in STEPS_TO_KEEP:
                continue
            metrics = load_metrics_at_step(teacher_dir, step)
            if metrics:
                step_data.setdefault(step, []).append(metrics)

    aggregated = {}
    for step in sorted(step_data.keys()):
        mlist = step_data[step]
        for key in ["cot_importance", "accuracy", "verifier_accuracy"]:
            vals = [m[key] for m in mlist if m[key] is not None]
            aggregated.setdefault(step, {})[key] = {
                "mean": float(np.mean(vals)) if vals else None,
                "std":  float(np.std(vals))  if vals else None,
                "n":    len(vals),
            }

    print(f"  [{label}] loaded steps: {sorted(aggregated.keys())}")
    return label, aggregated


def plot_metric(ax, series_list, metric_key, ylabel, colors):
    for idx, (label, step_agg) in enumerate(series_list):
        steps, means, stds = [], [], []
        for step in sorted(step_agg.keys()):
            entry = step_agg[step].get(metric_key, {})
            m = entry.get("mean")
            s = entry.get("std")
            if m is not None:
                steps.append(step)
                means.append(m)
                stds.append(s if s is not None else 0.0)

        if not steps:
            continue

        steps_arr = np.array(steps)
        means_arr = np.array(means)
        stds_arr  = np.array(stds)

        color = colors[idx % len(colors)]
        ax.plot(steps_arr, means_arr, linewidth=4.0, color=color, marker='o',
                markersize=10, label=label, alpha=0.85,
                markeredgewidth=1.5, markeredgecolor='white')

    ax.set_xlabel("RL Training Step", fontsize=28, fontweight="bold", family="serif")
    ax.set_ylabel(ylabel, fontsize=28, fontweight="bold", family="serif")
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_family("serif")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    ax.minorticks_on()
    ax.legend(fontsize=18, loc="best", framealpha=0.95,
              prop={"family": "serif"}, edgecolor='gray')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def main():
    model_size = sys.argv[1] if len(sys.argv) > 1 else "3b"
    base = f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct"

    series_specs = [
        (
            "Original (α=0)",
            lambda task: f"{base}/cot_importance_trained_original_k8_0.8/{task}/-1.0/teacher"
        ),
        (
            "CIR-reward α=0.8, seed 0",
            lambda task: f"{base}/cot_importance_trained_seed0_k8_0.8/{task}/-1.0/teacher"
        ),
        (
            "CIR-reward α=0.8, seed 1",
            lambda task: f"{base}/cot_importance_trained_seed1_k8_0.8/{task}/-1.0/teacher"
        ),
        (
            "CIR-reward α=0.8, seed 2",
            lambda task: f"{base}/cot_importance_trained_seed2_k8_0.8/{task}/-1.0/teacher"
        ),
        (
            "CIR-reward α=0.8, seed 3",
            lambda task: f"{base}/cot_importance_trained_seed3_k8_0.8/{task}/-1.0/teacher"
        ),
    ]

    colors = ["#2E2E2E", "#1E88E5", "#FFA726", "#228B22", "#FF0000"]

    print(f"\nLoading data for qwen-{model_size}-instruct ...")
    series_list = []
    for label, dir_fn in series_specs:
        print(f"\n=== {label} ===")
        series_list.append(load_series(dir_fn, label))

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))

    plot_metric(ax1, series_list, "accuracy", "Accuracy", colors)
    ax1.set_title("CIR-reward → Accuracy", fontsize=30, fontweight="bold", pad=15, family="serif")

    fig.suptitle(
        f"CIR-reward α=0.8: Seed Variation vs Original sample 8 times",
        fontsize=22, fontweight="bold", family="serif", y=1.02
    )
    plt.tight_layout(pad=2.0)

    output_dir = "/nlp/scr/qinanyu/rl-explanations/analysis/graph"
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = f"{output_dir}/figure_rebuttal_seeds_{model_size}.pdf"
    png_path = f"{output_dir}/figure_rebuttal_seeds_{model_size}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
    plt.close()

    print(f"\nSaved:\n  {pdf_path}\n  {png_path}")


if __name__ == "__main__":
    main()
