#!/usr/bin/env python3
"""
Create scatter plots for correlations between CoT importance, verifier accuracy, GRPO-Direct difference, and generalization.

Usage: python scatter_plots_correlations.py [model_size]
Examples:
    python scatter_plots_correlations.py 3b
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
from pathlib import Path


def load_task_categories(json_path):
    """Load task categories from JSON file."""
    with open(json_path, 'r') as f:
        categories = json.load(f)

    # Create a mapping from task name to category
    task_to_category = {}
    for category, tasks in categories.items():
        for task in tasks:
            task_to_category[task] = category

    return task_to_category


def is_math_task(task_name, task_to_category):
    """Check if a task is a math task (Algebra, Arithmetic, or Geometry)."""
    category = task_to_category.get(task_name, "")
    return category in ["Algebra", "Arithmetic", "Geometry"]


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


def load_step_accuracy_from_evaluate(base_dir, task_name, step):
    """Load accuracy for a specific step from evaluate results."""
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

                # Extract accuracy scores
                scores = []
                for item in data:
                    if 'correct' in item:
                        scores.append(1 if item['correct'] else 0)
                    elif 'accuracy' in item:
                        scores.append(item['accuracy'])
                    elif 'score' in item:
                        scores.append(item['score'])
                    elif 'reward_score' in item:
                        scores.append(item['reward_score'])

                if scores:
                    return np.mean(scores)
            except:
                pass
    return None


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

                percentages = [10, 30, 30, 40, 50, 60, 70, 80, 90, 100]
                instance_vals = []
                for item in data:
                    if 'cot_importance_evaluation' in item:
                        js_divs = item['cot_importance_evaluation'].get('js_divergences', [])
                        if len(js_divs) >= 2:
                            sampled = [js_divs[max(0, min(int((p/100.0)*len(js_divs))-1, len(js_divs)-1))] for p in percentages]
                            instance_vals.append(np.mean(sampled))
                return np.mean(instance_vals) if instance_vals else None
            except:
                pass
    return None


def load_verifier_accuracy_at_step(base_dir, task_name, step, debug=False):
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
        if debug:
            print(f"    [{task_name} step {step}] No file found at any path")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        if debug:
            print(f"    [{task_name} step {step}] Failed to load JSON: {e}")
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
        if debug:
            print(f"    [{task_name} step {step}] File exists but no 'verifier_comparison' field found (checked {len(data)} items)")
        return None


def load_generalization_step(base_dir, task_name, step):
    """Load generalization accuracy for a step."""
    folder_name = f"{task_name}_cot_importance"
    gen_path = os.path.join(base_dir, folder_name, "-1.0", "teacher", f"step_{step}", "generalization", f"{task_name}_step_{step}.json")

    if os.path.exists(gen_path):
        try:
            with open(gen_path, 'r') as f:
                data = json.load(f)
            scores = [item.get('best_reward_score', item.get('reward_score')) for item in data
                     if 'best_reward_score' in item or 'reward_score' in item]
            return np.mean(scores) if scores else None
        except: pass
    return None


def load_task_metrics(base_dir, task_name, start_step, end_step):
    """Load all metrics for a task."""
    metrics = {}

    # CoT importance
    initial_cot = load_cot_importance_at_step(base_dir, task_name, start_step)
    final_cot = load_cot_importance_at_step(base_dir, task_name, end_step)
    metrics['initial_cot_importance'] = initial_cot
    metrics['final_cot_importance'] = final_cot
    metrics['cot_change'] = (final_cot - initial_cot) if (initial_cot is not None and final_cot is not None) else None

    # Accuracy
    initial_grpo = load_step_accuracy_from_evaluate(base_dir, task_name, start_step)
    final_grpo = load_step_accuracy_from_evaluate(base_dir, task_name, end_step)

    metrics['initial_accuracy'] = initial_grpo
    metrics['final_accuracy'] = final_grpo
    metrics['accuracy_change'] = (final_grpo - initial_grpo) if (initial_grpo is not None and final_grpo is not None) else None

    # Verifier accuracy
    initial_verif = load_verifier_accuracy_at_step(base_dir, task_name, start_step)
    final_verif = load_verifier_accuracy_at_step(base_dir, task_name, end_step)
    metrics['initial_verifier_acc'] = initial_verif
    metrics['final_verifier_acc'] = final_verif
    metrics['verifier_delta'] = (final_verif - initial_verif) if (initial_verif is not None and final_verif is not None) else None

    # Generalization
    initial_gen = load_generalization_step(base_dir, task_name, start_step)
    final_gen = load_generalization_step(base_dir, task_name, end_step)
    metrics['initial_gen_acc'] = initial_gen
    metrics['final_gen_acc'] = final_gen
    metrics['gen_delta'] = (final_gen - initial_gen) if (initial_gen is not None and final_gen is not None) else None

    return metrics


def create_scatter_plot(ax, x_vals, y_vals, task_names, x_label, y_label, title, task_to_category=None):
    """Create a scatter plot with regression line and statistics, separated by math/non-math tasks."""
    if len(x_vals) < 3:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
               transform=ax.transAxes, fontsize=19, family='serif')
        return

    # Plot all tasks with the same color (no separation by math/non-math)
    ax.scatter(x_vals, y_vals, alpha=0.6, s=130, edgecolors='black', linewidth=1.5,
              color='#1E88E5', zorder=3)

    # Regression line for all data (no label)
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(x_vals), max(x_vals), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5)

    # Add reference lines
    # Horizontal line at y=0.5 for performance delta
    ax.axhline(y=0.5, color='green', linestyle='-', linewidth=2, alpha=0.7, zorder=1)
    # Vertical line at x=0.0
    ax.axvline(x=0.0, color='green', linestyle='-', linewidth=2, alpha=0.7, zorder=1)

    ax.set_xlabel(x_label, fontsize=25, fontweight='bold', family='serif')
    ax.set_ylabel(y_label, fontsize=25, fontweight='bold', family='serif')
    # ax.set_title(title, fontsize=25, fontweight='bold', pad=30, family='serif')

    # Set tick label fonts
    for label in ax.get_xticklabels():
        label.set_family('serif')
        label.set_fontsize(19)
    for label in ax.get_yticklabels():
        label.set_family('serif')
        label.set_fontsize(19)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='both', zorder=0)


def main():
    model_size = sys.argv[1] if len(sys.argv) > 1 else "3b"
    base_dir = f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct/cot_importance"

    # Load task categories
    task_category_path = '/nlp/scr/qinanyu/rl-explanations/analysis/task_category.json'
    if not os.path.exists(task_category_path):
        task_to_category = None
    else:
        task_to_category = load_task_categories(task_category_path)

    if not os.path.exists(base_dir):
        return

    # Collect data - iterate through all task folders
    task_data = {}
    task_folders = [item.replace('_cot_importance', '')
                    for item in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, item))]

    start_step = 2
    end_step = 156

    for task_name in sorted(task_folders):

        # Load initial and final metrics
        initial_cot = load_cot_importance_at_step(base_dir, task_name, start_step)
        final_cot = load_cot_importance_at_step(base_dir, task_name, end_step)

        initial_verif = load_verifier_accuracy_at_step(base_dir, task_name, start_step)
        final_verif = load_verifier_accuracy_at_step(base_dir, task_name, end_step)

        initial_grpo = load_step_accuracy_from_evaluate(base_dir, task_name, start_step)
        final_grpo = load_step_accuracy_from_evaluate(base_dir, task_name, end_step)

        # Calculate changes
        cot_change = (final_cot - initial_cot) if (initial_cot is not None and final_cot is not None) else None
        verifier_change = (final_verif - initial_verif) if (initial_verif is not None and final_verif is not None) else None
        accuracy_change = (final_grpo - initial_grpo) if (initial_grpo is not None and final_grpo is not None) else None

        metrics = {
            'cot_change': cot_change,
            'verifier_change': verifier_change,
            'accuracy_change': accuracy_change,
            'initial_verifier': initial_verif,
            'initial_cot': initial_cot
        }

        # Only include tasks that have at least some data
        if any(metrics[k] is not None for k in ['cot_change', 'verifier_change', 'accuracy_change']):
            task_data[task_name] = metrics

    if not task_data:
        return

    # Define correlation pairs to plot - 2 plots (1x2 grid)
    correlations = [
        ('cot_change', 'accuracy_change', '$\\Delta$ CIR', '$\\Delta$ Acc', 'CIR vs Performance'),
        ('verifier_change', 'accuracy_change', '$\\Delta$ SR', '$\\Delta$ Acc', 'SR vs Performance'),
    ]

    # Create figure with subplots (1x2 grid)
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)

    for idx, (x_key, y_key, x_label, y_label, title) in enumerate(correlations):
        ax = fig.add_subplot(gs[0, idx])

        # Get paired values
        task_names, x_vals, y_vals = [], [], []
        for task in sorted(task_data.keys()):
            x_val = task_data[task].get(x_key)
            y_val = task_data[task].get(y_key)
            if x_val is not None and y_val is not None:
                task_names.append(task)
                x_vals.append(x_val)
                y_vals.append(y_val)

        # Calculate Spearman correlation
        if len(x_vals) >= 3:
            spearman_r, spearman_p = stats.spearmanr(x_vals, y_vals)
            print(f"{title}: Spearman ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")

        create_scatter_plot(ax, x_vals, y_vals, task_names, x_label, y_label, title, task_to_category)

    fig.suptitle(f'Correlation between $\\Delta$ CIR, $\\Delta$ SR and $\\Delta$ Acc Qwen2.5-{model_size.upper()}', fontsize=28, fontweight='bold', family='serif', y=1.02)

    plt.tight_layout()

    # Save figure
    output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis/graph'
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = f'{output_dir}/figure4_scatter_correlations_{model_size}.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')

    png_path = f'{output_dir}/figure4_scatter_correlations_{model_size}.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')

    plt.close()

    # Additional analysis: CIR & SR Δ vs Performance Δ categorization

    # Get data for CIR Δ vs Performance Δ
    cir_changes = []
    cir_perf_changes = []
    for task in sorted(task_data.keys()):
        cir_val = task_data[task].get('cot_change')
        perf_val = task_data[task].get('accuracy_change')
        if cir_val is not None and perf_val is not None:
            cir_changes.append(cir_val)
            cir_perf_changes.append(perf_val)

    # Get data for SR Δ vs Performance Δ
    sr_changes = []
    sr_perf_changes = []
    for task in sorted(task_data.keys()):
        sr_val = task_data[task].get('verifier_change')
        perf_val = task_data[task].get('accuracy_change')
        if sr_val is not None and perf_val is not None:
            sr_changes.append(sr_val)
            sr_perf_changes.append(perf_val)

    if len(cir_changes) >= 1 and len(sr_changes) >= 1:
        # CIR categorization
        low_perf_cir_negative = sum(1 for cir, perf in zip(cir_changes, cir_perf_changes) if perf < 0.5 and cir < 0)
        low_perf_cir_positive = sum(1 for cir, perf in zip(cir_changes, cir_perf_changes) if perf < 0.5 and cir > 0)
        low_perf_cir_total = sum(1 for perf in cir_perf_changes if perf < 0.5)

        high_perf_cir_negative = sum(1 for cir, perf in zip(cir_changes, cir_perf_changes) if perf >= 0.5 and cir < 0)
        high_perf_cir_positive = sum(1 for cir, perf in zip(cir_changes, cir_perf_changes) if perf >= 0.5 and cir > 0)
        high_perf_cir_total = sum(1 for perf in cir_perf_changes if perf >= 0.5)

        # SR categorization
        low_perf_sr_negative = sum(1 for sr, perf in zip(sr_changes, sr_perf_changes) if perf < 0.5 and sr < 0)
        low_perf_sr_positive = sum(1 for sr, perf in zip(sr_changes, sr_perf_changes) if perf < 0.5 and sr > 0)
        low_perf_sr_total = sum(1 for perf in sr_perf_changes if perf < 0.5)

        high_perf_sr_negative = sum(1 for sr, perf in zip(sr_changes, sr_perf_changes) if perf >= 0.5 and sr < 0)
        high_perf_sr_positive = sum(1 for sr, perf in zip(sr_changes, sr_perf_changes) if perf >= 0.5 and sr > 0)
        high_perf_sr_total = sum(1 for perf in sr_perf_changes if perf >= 0.5)

        # Create side-by-side bar graphs
        fig_bar, (ax_cir, ax_sr) = plt.subplots(1, 2, figsize=(12, 7))

        categories = ['Low Performance\n(Δ < 0.5)', 'High Performance\n(Δ >= 0.5)']
        x = np.arange(len(categories))
        width = 0.35

        # LEFT: CIR bar plot
        cir_low_perf_prop_negative = low_perf_cir_negative / low_perf_cir_total if low_perf_cir_total > 0 else 0
        cir_low_perf_prop_positive = low_perf_cir_positive / low_perf_cir_total if low_perf_cir_total > 0 else 0
        cir_high_perf_prop_negative = high_perf_cir_negative / high_perf_cir_total if high_perf_cir_total > 0 else 0
        cir_high_perf_prop_positive = high_perf_cir_positive / high_perf_cir_total if high_perf_cir_total > 0 else 0

        ax_cir.bar(x - width/2, [cir_low_perf_prop_negative, cir_high_perf_prop_negative],
                   width, label='CIR Δ < 0', color='#E57373', edgecolor='black', linewidth=1.5)
        ax_cir.bar(x + width/2, [cir_low_perf_prop_positive, cir_high_perf_prop_positive],
                   width, label='CIR Δ > 0', color='#89CFF0', edgecolor='black', linewidth=1.5)

        ax_cir.set_ylabel('Proportion', fontsize=30, fontweight='bold', family='serif')
        ax_cir.set_xlabel('Performance Gain', fontsize=30, fontweight='bold', family='serif')
        ax_cir.set_title('CIR Change Distribution\nby Performance Level',
                        fontsize=22, fontweight='bold', pad=30, family='serif')
        ax_cir.set_xticks(x)
        ax_cir.set_xticklabels(categories, fontsize=18, family='serif')
        ax_cir.set_ylim(0, 1.0)
        ax_cir.legend(fontsize=24, loc='upper right', framealpha=0.9, prop={'family': 'serif'})
        ax_cir.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y', zorder=0)

        for label in ax_cir.get_yticklabels():
            label.set_family('serif')
            label.set_fontsize(16)

        # RIGHT: SR bar plot
        sr_low_perf_prop_negative = low_perf_sr_negative / low_perf_sr_total if low_perf_sr_total > 0 else 0
        sr_low_perf_prop_positive = low_perf_sr_positive / low_perf_sr_total if low_perf_sr_total > 0 else 0
        sr_high_perf_prop_negative = high_perf_sr_negative / high_perf_sr_total if high_perf_sr_total > 0 else 0
        sr_high_perf_prop_positive = high_perf_sr_positive / high_perf_sr_total if high_perf_sr_total > 0 else 0

        ax_sr.bar(x - width/2, [sr_low_perf_prop_negative, sr_high_perf_prop_negative],
                  width, label='SR Δ < 0', color='#E57373', edgecolor='black', linewidth=1.5)
        ax_sr.bar(x + width/2, [sr_low_perf_prop_positive, sr_high_perf_prop_positive],
                  width, label='SR Δ > 0', color='#89CFF0', edgecolor='black', linewidth=1.5)

        ax_sr.set_ylabel('Proportion', fontsize=30, fontweight='bold', family='serif')
        ax_sr.set_xlabel('Performance Gain', fontsize=30, fontweight='bold', family='serif')
        ax_sr.set_title('SR Change Distribution\nby Performance Level',
                       fontsize=22, fontweight='bold', pad=30, family='serif')
        ax_sr.set_xticks(x)
        ax_sr.set_xticklabels(categories, fontsize=18, family='serif')
        ax_sr.set_ylim(0, 1.0)
        ax_sr.legend(fontsize=24, loc='upper right', framealpha=0.9, prop={'family': 'serif'})
        ax_sr.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y', zorder=0)

        for label in ax_sr.get_yticklabels():
            label.set_family('serif')
            label.set_fontsize(16)

        plt.tight_layout()

        # Save bar graph
        bar_pdf_path = f'/nlp/scr/qinanyu/rl-explanations/analysis/graph/figure4_bar_cir_sr_by_performance_{model_size}.pdf'
        plt.savefig(bar_pdf_path, dpi=300, bbox_inches='tight', format='pdf')

        bar_png_path = f'/nlp/scr/qinanyu/rl-explanations/analysis/graph/figure4_bar_cir_sr_by_performance_{model_size}.png'
        plt.savefig(bar_png_path, dpi=300, bbox_inches='tight', format='png')

        plt.close()


if __name__ == "__main__":
    main()
