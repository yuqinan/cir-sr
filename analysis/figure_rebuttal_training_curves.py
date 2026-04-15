#!/usr/bin/env python3
"""
Rebuttal figure: scatter plots of ΔCIR vs ΔAcc and ΔSR vs ΔAcc at each training step.

Each column = one training step (delta from step 2 to that step).
Row 0: ΔCIR vs ΔAcc  |  Row 1: ΔSR vs ΔAcc
Same visual style as Figure 4 (green reference lines, red dashed regression).

Usage: python figure_rebuttal_training_curves.py [model_size]
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


BASELINE_STEP = 2
STEPS = [30, 60, 90, 120, 150, 156]
PERCENTAGES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def load_step_data(base_dir, task_name, step):
    json_path = os.path.join(
        base_dir, f"{task_name}_cot_importance", "-1.0", "teacher",
        f"step_{step}", f"teacher_responses_step_{step}.json"
    )
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def compute_accuracy(items):
    scores = [item.get('best_reward_score', item.get('reward_score'))
              for item in items
              if 'best_reward_score' in item or 'reward_score' in item]
    return np.mean(scores) if scores else None


def compute_cir(items):
    instance_vals = []
    for item in items:
        if 'cot_importance_evaluation' not in item:
            continue
        js_divs = item['cot_importance_evaluation'].get('js_divergences', [])
        if len(js_divs) < 2:
            continue
        sampled = [
            js_divs[max(0, min(int((p / 100.0) * len(js_divs)) - 1, len(js_divs) - 1))]
            for p in PERCENTAGES
        ]
        instance_vals.append(np.mean(sampled))
    return np.mean(instance_vals) if instance_vals else None


def compute_sr(items):
    scores = []
    for item in items:
        if 'verifier_comparison' not in item:
            continue
        vc = item['verifier_comparison']
        answers_match = vc.get('answers_match', False)
        with_q = vc.get('with_question_answer', '').strip().lower()
        without_q = vc.get('without_question_answer', '').strip().lower()
        if answers_match and with_q == 'no answer found' and without_q == 'no answer found':
            scores.append(0)
        else:
            scores.append(1 if answers_match else 0)
    return np.mean(scores) if scores else None


def collect_all_metrics(base_dir):
    """Returns dict: task_name -> {step -> {'accuracy', 'cir', 'sr'}}"""
    task_dirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.endswith('_cot_importance')
    ])
    task_names = [d.replace('_cot_importance', '') for d in task_dirs]

    all_metrics = {}
    for task_name in task_names:
        task_metrics = {}
        for step in [BASELINE_STEP] + STEPS:
            items = load_step_data(base_dir, task_name, step)
            if items is None:
                continue
            acc = compute_accuracy(items)
            cir = compute_cir(items)
            sr  = compute_sr(items)
            if acc is not None or cir is not None or sr is not None:
                task_metrics[step] = {'accuracy': acc, 'cir': cir, 'sr': sr}
        if task_metrics:
            all_metrics[task_name] = task_metrics

    return all_metrics


def draw_scatter(ax, x_vals, y_vals, x_label, y_label, first_col):
    if len(x_vals) < 3:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=13, family='serif')
        return

    ax.scatter(x_vals, y_vals, alpha=0.6, s=100, edgecolors='black',
               linewidth=1.5, color='#1E88E5', zorder=3)

    # Regression line
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(x_vals), max(x_vals), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5)

    # Reference lines (same as figure 4)
    ax.axhline(y=0.5, color='green', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
    ax.axvline(x=0.0, color='green', linestyle='-', linewidth=2, alpha=0.7, zorder=1)

    if first_col:
        ax.set_ylabel(y_label, fontsize=18, fontweight='bold', family='serif')
    ax.set_xlabel(x_label, fontsize=16, fontweight='bold', family='serif')

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_family('serif')
        lbl.set_fontsize(12)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)


def main():
    model_size = sys.argv[1] if len(sys.argv) > 1 else "3b"
    base_dir = (f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/"
                f"grpo_qwen-{model_size}-instruct/4o_mini_cot_importance")

    if not os.path.exists(base_dir):
        print(f"Base dir not found: {base_dir}")
        return

    print(f"Loading metrics from {base_dir} ...")
    all_metrics = collect_all_metrics(base_dir)
    print(f"Loaded data for {len(all_metrics)} tasks.")

    n_steps = len(STEPS)
    fig, axes = plt.subplots(2, n_steps, figsize=(4.5 * n_steps, 9))

    row_specs = [
        ('cir', '$\\Delta$ CIR', '$\\Delta$ Acc.'),
        ('sr',  '$\\Delta$ SR',  '$\\Delta$ Acc.'),
    ]

    for col_idx, step in enumerate(STEPS):
        axes[0, col_idx].set_title(f'Step {step}', fontsize=17,
                                   fontweight='bold', family='serif', pad=8)

        for row_idx, (metric_key, x_label, y_label) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            x_vals, y_vals = [], []

            for task_name, task_metrics in all_metrics.items():
                base = task_metrics.get(BASELINE_STEP, {})
                curr = task_metrics.get(step, {})
                x_base = base.get(metric_key)
                x_curr = curr.get(metric_key)
                y_base = base.get('accuracy')
                y_curr = curr.get('accuracy')

                if None not in (x_base, x_curr, y_base, y_curr):
                    x_vals.append(x_curr - x_base)
                    y_vals.append(y_curr - y_base)

            draw_scatter(ax, x_vals, y_vals, x_label, y_label,
                         first_col=(col_idx == 0))

            if len(x_vals) >= 3:
                r, pval = stats.spearmanr(x_vals, y_vals)
                print(f"Step {step:4d} | {metric_key} vs acc: ρ={r:.4f}, p={pval:.4f}, n={len(x_vals)}")

    fig.suptitle(
        f'Correlation between $\\Delta$Acc. and $\\Delta$CIR, $\\Delta$SR across Training Steps — Qwen2.5-{model_size.upper()}',
        fontsize=20, fontweight='bold', family='serif', y=1.01
    )
    plt.tight_layout()

    output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis/graph'
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = f'{output_dir}/figure_rebuttal_training_curves_{model_size}.pdf'
    png_path = f'{output_dir}/figure_rebuttal_training_curves_{model_size}.png'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()

    print(f"\nSaved:\n  {pdf_path}\n  {png_path}")


if __name__ == "__main__":
    main()
