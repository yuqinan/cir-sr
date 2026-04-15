#!/usr/bin/env python3
"""
Create scatter plots showing correlation between initial and final CIR/SR values.

Usage:
    python plot_initial_final_correlation.py
    python plot_initial_final_correlation.py --model_size 3b
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Set font to serif for consistency
plt.rcParams['font.family'] = 'serif'


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


def load_verifier_accuracy_at_step(base_dir, task_name, step):
    """
    Load average verifier accuracy from verifier_comparison.answers_match for a task at a specific training step.

    Special handling: If answers_match is True but both with_question_answer and without_question_answer
    are "no answer found", count as 0 (incorrect).
    """
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
    model_size = sys.argv[1] if len(sys.argv) > 1 else "3b"
    base_dir = f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct/cot_importance"

    print(f"Loading data for model: {model_size}")
    print(f"Base directory: {base_dir}")

    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return

    # Collect data
    task_folders = [item.replace('_cot_importance', '')
                    for item in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, item))]

    start_step = 2
    end_step = 156

    cir_data = []
    sr_data = []
    task_names_cir = []
    task_names_sr = []

    for task_name in sorted(task_folders):
        # Load CIR values
        initial_cir = load_cot_importance_at_step(base_dir, task_name, start_step)
        final_cir = load_cot_importance_at_step(base_dir, task_name, end_step)

        if initial_cir is not None and final_cir is not None:
            cir_data.append((initial_cir, final_cir))
            task_names_cir.append(task_name)

        # Load SR values
        initial_sr = load_verifier_accuracy_at_step(base_dir, task_name, start_step)
        final_sr = load_verifier_accuracy_at_step(base_dir, task_name, end_step)

        if initial_sr is not None and final_sr is not None:
            sr_data.append((initial_sr, final_sr))
            task_names_sr.append(task_name)

    print(f"\nTasks with CIR data: {len(cir_data)}")
    print(f"Tasks with SR data: {len(sr_data)}")

    if not cir_data and not sr_data:
        print("No data found!")
        return

    print("\n" + "=" * 80)
    print("INITIAL vs FINAL CORRELATION ANALYSIS")
    print("=" * 80)

    # Create side-by-side plots
    fig, (ax_cir, ax_sr) = plt.subplots(1, 2, figsize=(14, 7))

    # LEFT: CIR correlation
    if cir_data:
        initial_cir_vals = [x[0] for x in cir_data]
        final_cir_vals = [x[1] for x in cir_data]

        # Calculate Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(initial_cir_vals, final_cir_vals)

        # Scatter plot
        ax_cir.scatter(initial_cir_vals, final_cir_vals, s=100, color='#1E88E5',
                      edgecolors='black', linewidth=1.5, alpha=0.7, zorder=3)

        # Add regression line
        z = np.polyfit(initial_cir_vals, final_cir_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(initial_cir_vals), max(initial_cir_vals), 100)
        ax_cir.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, zorder=2, label='Regression line')

        # Add diagonal line (y=x)
        min_val = min(min(initial_cir_vals), min(final_cir_vals))
        max_val = max(max(initial_cir_vals), max(final_cir_vals))
        ax_cir.plot([min_val, max_val], [min_val, max_val], 'k:', linewidth=2, alpha=0.5, zorder=1, label='y=x')

        # Formatting
        ax_cir.set_xlabel('Initial CIR (Step 2)', fontsize=20, fontweight='bold', family='serif')
        ax_cir.set_ylabel('Final CIR (Step 156)', fontsize=20, fontweight='bold', family='serif')
        ax_cir.set_title(f'Initial vs Final CIR\nSpearman ρ = {spearman_r:.3f} (p = {spearman_p:.4f})',
                        fontsize=22, fontweight='bold', pad=20, family='serif')
        ax_cir.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=0)
        ax_cir.legend(fontsize=16, loc='upper left', framealpha=0.9, prop={'family': 'serif'})

        for label in ax_cir.get_xticklabels() + ax_cir.get_yticklabels():
            label.set_family('serif')
            label.set_fontsize(16)

        print(f"\nCIR Correlation:")
        print(f"  Spearman ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")
        print(f"  n = {len(cir_data)} tasks")

    # RIGHT: SR correlation
    if sr_data:
        initial_sr_vals = [x[0] for x in sr_data]
        final_sr_vals = [x[1] for x in sr_data]

        # Calculate Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(initial_sr_vals, final_sr_vals)

        # Scatter plot
        ax_sr.scatter(initial_sr_vals, final_sr_vals, s=100, color='#E91E63',
                     edgecolors='black', linewidth=1.5, alpha=0.7, zorder=3)

        # Add regression line
        z = np.polyfit(initial_sr_vals, final_sr_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(initial_sr_vals), max(initial_sr_vals), 100)
        ax_sr.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, zorder=2, label='Regression line')

        # Add diagonal line (y=x)
        min_val = min(min(initial_sr_vals), min(final_sr_vals))
        max_val = max(max(initial_sr_vals), max(final_sr_vals))
        ax_sr.plot([min_val, max_val], [min_val, max_val], 'k:', linewidth=2, alpha=0.5, zorder=1, label='y=x')

        # Formatting
        ax_sr.set_xlabel('Initial SR (Step 2)', fontsize=20, fontweight='bold', family='serif')
        ax_sr.set_ylabel('Final SR (Step 156)', fontsize=20, fontweight='bold', family='serif')
        ax_sr.set_title(f'Initial vs Final SR\nSpearman ρ = {spearman_r:.3f} (p = {spearman_p:.4f})',
                       fontsize=22, fontweight='bold', pad=20, family='serif')
        ax_sr.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=0)
        ax_sr.legend(fontsize=16, loc='upper left', framealpha=0.9, prop={'family': 'serif'})

        for label in ax_sr.get_xticklabels() + ax_sr.get_yticklabels():
            label.set_family('serif')
            label.set_fontsize(16)

        print(f"\nSR Correlation:")
        print(f"  Spearman ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")
        print(f"  n = {len(sr_data)} tasks")

    print("\n" + "=" * 80)

    plt.tight_layout()

    # Save figure
    output_path = f'/nlp/scr/qinanyu/rl-explanations/analysis/graph/initial_final_correlation_{model_size}.pdf'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\nSaved plot to: {output_path}")

    plt.close()
    print("\nPlot created successfully!")


if __name__ == "__main__":
    main()
