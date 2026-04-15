#!/usr/bin/env python3
"""
Compare Verifier Accuracy evolution between 4.1-mini and 4o-mini across training.

Usage:
    python plot_verifier_comparison.py [model_size]
    python plot_verifier_comparison.py 3b
    python plot_verifier_comparison.py 7b
    python plot_verifier_comparison.py 1.5b
    python plot_verifier_comparison.py 3b --local  # for local testing
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Set font to serif for consistency
plt.rcParams['font.family'] = 'serif'


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


def load_verifier_accuracy_at_step(base_dir, task_name, step):
    """
    Load average verifier accuracy from verifier_comparison.answers_match for a task at a specific training step.

    Special handling: If answers_match is True but both with_question_answer and without_question_answer
    are "no answer found", count as 0 (incorrect).

    Args:
        base_dir: Base directory containing task results
        task_name: Name of the task folder
        step: Training step number

    Returns:
        Average verifier accuracy or None if data not found
    """
    # Try different possible paths
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


def load_verifier_data(base_dir, training_steps):
    """Load verifier accuracy data for all tasks at specified training steps."""
    task_folders = [item.replace('_cot_importance', '')
                    for item in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, item))]

    print(f"  Found {len(task_folders)} task folders")

    verifier_task_data = {}
    tasks_with_verifier_data = 0

    for task_name in sorted(task_folders):
        # Load verifier accuracy for all steps
        step_values = {}
        for step in training_steps:
            verifier_acc = load_verifier_accuracy_at_step(base_dir, task_name, step)
            if verifier_acc is not None:
                step_values[step] = verifier_acc

        if step_values:
            tasks_with_verifier_data += 1

        # Only include tasks that have data for at least first and last step
        if (2 in step_values or 156 in step_values or 150 in step_values):
            verifier_task_data[task_name] = {
                'steps': step_values
            }

    print(f"  Tasks with verifier data: {tasks_with_verifier_data}")
    print(f"  Tasks included in dataset: {len(verifier_task_data)}")

    return verifier_task_data


def plot_verifier_subplot(ax, verifier_task_data, training_steps, step_colors, title, task_to_category):
    """Plot verifier accuracy evolution for a single subplot."""
    # Sort tasks by initial verifier accuracy (lowest to highest)
    def get_initial_verifier(item):
        task_name, data = item
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 1:
            return steps[available_steps[0]]
        return 0

    sorted_verifier_tasks = sorted(verifier_task_data.items(), key=get_initial_verifier)
    verifier_task_names = [task for task, _ in sorted_verifier_tasks]

    x_positions_verifier = np.arange(len(verifier_task_names))

    # Add legend entries for both steps (dummy points outside plot)
    ax.scatter([], [], s=80, color=step_colors[2], edgecolors='black', linewidth=1.5, label='Before Training')
    ax.scatter([], [], s=80, color=step_colors[156], edgecolors='black', linewidth=1.5, label='After Training')

    for i, (task, data) in enumerate(zip(verifier_task_names, [d for _, d in sorted_verifier_tasks])):
        step_values = data['steps']

        # Get all available step values for this task (only steps 2 and 156)
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
            line_color = '#2E86AB' if is_increasing else '#D32F2F'

            # Draw line with multiple chevron arrows
            num_chevrons = 5
            for idx in range(num_chevrons):
                start_frac = idx / num_chevrons
                end_frac = (idx + 1) / num_chevrons

                y_start = y_values[0] + start_frac * (y_values[-1] - y_values[0])
                y_end = y_values[0] + end_frac * (y_values[-1] - y_values[0])

                ax.annotate('', xy=(i, y_end), xytext=(i, y_start),
                           arrowprops=dict(arrowstyle='->', color=line_color, lw=2.5, alpha=0.7,
                                         mutation_scale=15), zorder=2)

        # Plot dots at each available step with step colors
        for step, y_val, color in zip(available_steps, y_values, colors):
            ax.scatter(i, y_val, s=80, color=color,
                       edgecolors='black', linewidth=1.5, zorder=3)

    # Formatting
    ax.set_xticks(x_positions_verifier)
    ax.set_xticklabels(verifier_task_names, rotation=70, ha='right', fontsize=19, family='serif')

    # Make math task names bold
    for i, (label, task_name) in enumerate(zip(ax.get_xticklabels(), verifier_task_names)):
        if is_math_task(task_name, task_to_category):
            label.set_fontweight('bold')

    ax.set_xlabel('Tasks (sorted by initial SR: lowest → highest)',
                  fontsize=35, fontweight='bold', family='serif')
    ax.set_ylabel('SR',
                  fontsize=35, fontweight='bold', family='serif')
    ax.set_title(title,
                 fontsize=35, fontweight='bold', pad=20, family='serif')

    for label in ax.get_yticklabels():
        label.set_family('serif')
        label.set_fontsize(19)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y', zorder=0)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), framealpha=0.9, prop={'family': 'serif', 'size': 24}, markerscale=2)


def analyze_verifier_agreement(verifier_data_1, verifier_data_2, name_1="4.1-mini", name_2="4o-mini"):
    """
    Analyze whether the two verifiers produce similar results using statistical tests.

    Tests:
    1. Paired t-test for initial accuracies (step 2)
    2. Paired t-test for final accuracies (step 156)
    3. Pearson correlation for initial and final accuracies
    4. Mean absolute difference
    """
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS: Comparing {name_1} vs {name_2}")
    print(f"{'='*80}")

    # Find common tasks
    tasks_1 = set(verifier_data_1.keys())
    tasks_2 = set(verifier_data_2.keys())
    common_tasks = tasks_1.intersection(tasks_2)

    print(f"\nDataset overview:")
    print(f"  Tasks in {name_1}: {len(tasks_1)}")
    print(f"  Tasks in {name_2}: {len(tasks_2)}")
    print(f"  Common tasks: {len(common_tasks)}")

    if len(common_tasks) == 0:
        print(f"\nERROR: No common tasks found between verifiers")
        return

    # Extract initial and final accuracies for common tasks
    initial_1 = []
    initial_2 = []
    final_1 = []
    final_2 = []
    tasks_with_both_steps = []

    for task in sorted(common_tasks):
        data_1 = verifier_data_1[task]['steps']
        data_2 = verifier_data_2[task]['steps']

        # Check if both have initial step (2)
        if 2 in data_1 and 2 in data_2:
            initial_1.append(data_1[2])
            initial_2.append(data_2[2])

        # Check if both have final step (156)
        if 156 in data_1 and 156 in data_2:
            final_1.append(data_1[156])
            final_2.append(data_2[156])

        # Track tasks with both steps in both verifiers
        if (2 in data_1 and 2 in data_2 and 156 in data_1 and 156 in data_2):
            tasks_with_both_steps.append(task)

    initial_1 = np.array(initial_1)
    initial_2 = np.array(initial_2)
    final_1 = np.array(final_1)
    final_2 = np.array(final_2)

    print(f"\n  Tasks with initial step (2) in both: {len(initial_1)}")
    print(f"  Tasks with final step (156) in both: {len(final_1)}")
    print(f"  Tasks with both steps in both verifiers: {len(tasks_with_both_steps)}")

    # ============================================================================
    # INITIAL ACCURACY COMPARISON (Step 2)
    # ============================================================================
    if len(initial_1) >= 2:
        print(f"\n{'-'*80}")
        print(f"INITIAL ACCURACY COMPARISON (Step 2 - Before Training)")
        print(f"{'-'*80}")

        # Descriptive statistics
        print(f"\nDescriptive statistics:")
        print(f"  {name_1}: mean={np.mean(initial_1):.4f}, std={np.std(initial_1):.4f}, median={np.median(initial_1):.4f}")
        print(f"  {name_2}: mean={np.mean(initial_2):.4f}, std={np.std(initial_2):.4f}, median={np.median(initial_2):.4f}")

        # Mean difference
        mean_diff_initial = np.mean(initial_1 - initial_2)
        print(f"\n  Mean difference ({name_1} - {name_2}): {mean_diff_initial:.4f}")
        print(f"  Mean absolute difference: {np.mean(np.abs(initial_1 - initial_2)):.4f}")

        # Paired t-test
        t_stat_initial, p_value_initial = stats.ttest_rel(initial_1, initial_2)
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat_initial:.4f}")
        print(f"  p-value: {p_value_initial:.4f}")
        if p_value_initial < 0.05:
            print(f"  Result: SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"  Result: NO significant difference (p >= 0.05)")

        # Pearson correlation
        corr_initial, corr_p_initial = stats.pearsonr(initial_1, initial_2)
        print(f"\nPearson correlation:")
        print(f"  r = {corr_initial:.4f}")
        print(f"  p-value: {corr_p_initial:.4f}")
        if corr_initial > 0.7:
            print(f"  Result: STRONG positive correlation")
        elif corr_initial > 0.4:
            print(f"  Result: MODERATE positive correlation")
        else:
            print(f"  Result: WEAK correlation")

    # ============================================================================
    # FINAL ACCURACY COMPARISON (Step 156)
    # ============================================================================
    if len(final_1) >= 2:
        print(f"\n{'-'*80}")
        print(f"FINAL ACCURACY COMPARISON (Step 156 - After Training)")
        print(f"{'-'*80}")

        # Descriptive statistics
        print(f"\nDescriptive statistics:")
        print(f"  {name_1}: mean={np.mean(final_1):.4f}, std={np.std(final_1):.4f}, median={np.median(final_1):.4f}")
        print(f"  {name_2}: mean={np.mean(final_2):.4f}, std={np.std(final_2):.4f}, median={np.median(final_2):.4f}")

        # Mean difference
        mean_diff_final = np.mean(final_1 - final_2)
        print(f"\n  Mean difference ({name_1} - {name_2}): {mean_diff_final:.4f}")
        print(f"  Mean absolute difference: {np.mean(np.abs(final_1 - final_2)):.4f}")

        # Paired t-test
        t_stat_final, p_value_final = stats.ttest_rel(final_1, final_2)
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat_final:.4f}")
        print(f"  p-value: {p_value_final:.4f}")
        if p_value_final < 0.05:
            print(f"  Result: SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"  Result: NO significant difference (p >= 0.05)")

        # Pearson correlation
        corr_final, corr_p_final = stats.pearsonr(final_1, final_2)
        print(f"\nPearson correlation:")
        print(f"  r = {corr_final:.4f}")
        print(f"  p-value: {corr_p_final:.4f}")
        if corr_final > 0.7:
            print(f"  Result: STRONG positive correlation")
        elif corr_final > 0.4:
            print(f"  Result: MODERATE positive correlation")
        else:
            print(f"  Result: WEAK correlation")

    # ============================================================================
    # CHANGE ANALYSIS (Δ SR = final - initial)
    # ============================================================================
    if len(tasks_with_both_steps) >= 2:
        print(f"\n{'-'*80}")
        print(f"CHANGE ANALYSIS (Δ SR = Final - Initial)")
        print(f"{'-'*80}")

        # Calculate changes for tasks with both steps
        changes_1 = []
        changes_2 = []
        for task in tasks_with_both_steps:
            data_1 = verifier_data_1[task]['steps']
            data_2 = verifier_data_2[task]['steps']
            changes_1.append(data_1[156] - data_1[2])
            changes_2.append(data_2[156] - data_2[2])

        changes_1 = np.array(changes_1)
        changes_2 = np.array(changes_2)

        print(f"\nDescriptive statistics for Δ SR:")
        print(f"  {name_1}: mean={np.mean(changes_1):.4f}, std={np.std(changes_1):.4f}, median={np.median(changes_1):.4f}")
        print(f"  {name_2}: mean={np.mean(changes_2):.4f}, std={np.std(changes_2):.4f}, median={np.median(changes_2):.4f}")

        # Mean difference in changes
        mean_diff_change = np.mean(changes_1 - changes_2)
        print(f"\n  Mean difference in Δ SR ({name_1} - {name_2}): {mean_diff_change:.4f}")
        print(f"  Mean absolute difference: {np.mean(np.abs(changes_1 - changes_2)):.4f}")

        # Paired t-test
        t_stat_change, p_value_change = stats.ttest_rel(changes_1, changes_2)
        print(f"\nPaired t-test for Δ SR:")
        print(f"  t-statistic: {t_stat_change:.4f}")
        print(f"  p-value: {p_value_change:.4f}")
        if p_value_change < 0.05:
            print(f"  Result: SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"  Result: NO significant difference (p >= 0.05)")

        # Pearson correlation
        corr_change, corr_p_change = stats.pearsonr(changes_1, changes_2)
        print(f"\nPearson correlation for Δ SR:")
        print(f"  r = {corr_change:.4f}")
        print(f"  p-value: {corr_p_change:.4f}")
        if corr_change > 0.7:
            print(f"  Result: STRONG positive correlation")
        elif corr_change > 0.4:
            print(f"  Result: MODERATE positive correlation")
        else:
            print(f"  Result: WEAK correlation")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nThe analysis compares whether {name_1} and {name_2} produce similar results.")
    print(f"\nKey findings:")

    if len(initial_1) >= 2:
        if p_value_initial >= 0.05:
            print(f"  ✓ Initial accuracies are NOT significantly different (p={p_value_initial:.4f})")
        else:
            print(f"  ✗ Initial accuracies are significantly different (p={p_value_initial:.4f})")

    if len(final_1) >= 2:
        if p_value_final >= 0.05:
            print(f"  ✓ Final accuracies are NOT significantly different (p={p_value_final:.4f})")
        else:
            print(f"  ✗ Final accuracies are significantly different (p={p_value_final:.4f})")

    if len(tasks_with_both_steps) >= 2:
        if p_value_change >= 0.05:
            print(f"  ✓ Changes (Δ SR) are NOT significantly different (p={p_value_change:.4f})")
        else:
            print(f"  ✗ Changes (Δ SR) are significantly different (p={p_value_change:.4f})")

    print(f"\nInterpretation:")
    print(f"  - Paired t-test checks if the mean difference between verifiers is significantly different from 0")
    print(f"  - p-value >= 0.05 means we CANNOT reject the null hypothesis that they are the same")
    print(f"  - p-value < 0.05 means there IS a significant difference between verifiers")
    print(f"  - High correlation (r > 0.7) indicates the verifiers rank tasks similarly")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Compare Verifier Accuracy between 4.1-mini and 4o-mini')
    parser.add_argument('model_size', nargs='?', default='3b', help='Model size: 1.5b, 3b, or 7b (default: 3b)')
    parser.add_argument('--local', action='store_true', help='Use local paths (for testing)')

    args = parser.parse_args()

    # Define training steps (only before and after training)
    training_steps = [2, 156]
    step_labels = {2: "Before Training", 156: "After Training"}

    # Load task categories
    task_category_path = os.path.join(os.path.dirname(__file__), 'task_category.json')
    if not os.path.exists(task_category_path):
        print(f"Warning: task_category.json not found at {task_category_path}")
        task_to_category = {}
    else:
        task_to_category = load_task_categories(task_category_path)
        print(f"Loaded task categories from {task_category_path}")

    # Set base directories
    if args.local:
        base_dir_41_mini = f'/Users/qinanyu/Desktop/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'
        base_dir_4o_mini = f'/Users/qinanyu/Desktop/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/4o_mini_cot_importance'
        output_dir = '/Users/qinanyu/Desktop/rl-explanations/analysis/graph'
    else:
        base_dir_41_mini = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'
        base_dir_4o_mini = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/4o_mini_cot_importance'
        output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis/graph'

    print(f"\n{'='*80}")
    print(f"Configuration")
    print(f"{'='*80}")
    print(f"Model size: {args.model_size}")
    print(f"4.1-mini base directory: {base_dir_41_mini}")
    print(f"4o-mini base directory: {base_dir_4o_mini}")
    print(f"4.1-mini dir exists: {os.path.exists(base_dir_41_mini)}")
    print(f"4o-mini dir exists: {os.path.exists(base_dir_4o_mini)}")

    if not os.path.exists(base_dir_41_mini):
        print(f"ERROR: 4.1-mini directory does not exist: {base_dir_41_mini}")
        return

    if not os.path.exists(base_dir_4o_mini):
        print(f"ERROR: 4o-mini directory does not exist: {base_dir_4o_mini}")
        return

    # Load verifier data from both directories
    print(f"\n{'='*80}")
    print(f"Loading 4.1-mini verifier data")
    print(f"{'='*80}")
    verifier_data_41_mini = load_verifier_data(base_dir_41_mini, training_steps)

    print(f"\n{'='*80}")
    print(f"Loading 4o-mini verifier data")
    print(f"{'='*80}")
    verifier_data_4o_mini = load_verifier_data(base_dir_4o_mini, training_steps)

    if not verifier_data_41_mini and not verifier_data_4o_mini:
        print(f"\nERROR: No data found for either verifier")
        return

    # Perform statistical analysis comparing the two verifiers
    analyze_verifier_agreement(verifier_data_41_mini, verifier_data_4o_mini, "4.1-mini", "4o-mini")

    # Colors for before and after training
    step_colors = {
        2: (0.678, 0.847, 0.902),    # Light blue for "Before Training"
        156: (0.098, 0.275, 0.471),  # Dark blue for "After Training"
    }

    # Create the combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 10))

    # LEFT SUBPLOT: 4.1-mini
    print(f"\n{'='*80}")
    print(f"Plotting 4.1-mini verifier accuracy")
    print(f"{'='*80}")
    plot_verifier_subplot(ax1, verifier_data_41_mini, training_steps, step_colors,
                          'Evolution of SR: Before vs. After Training (4.1-mini)',
                          task_to_category)

    # RIGHT SUBPLOT: 4o-mini
    print(f"\n{'='*80}")
    print(f"Plotting 4o-mini verifier accuracy")
    print(f"{'='*80}")
    plot_verifier_subplot(ax2, verifier_data_4o_mini, training_steps, step_colors,
                          'Evolution of SR: Before vs. After Training (4o-mini)',
                          task_to_category)

    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f'verifier_comparison_{args.model_size}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved successfully")
    print(f"{'='*80}")
    print(f"PDF saved to: {pdf_path}")

    png_path = os.path.join(output_dir, f'verifier_comparison_{args.model_size}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"PNG saved to: {png_path}")
    print(f"{'='*80}\n")

    plt.show()


if __name__ == "__main__":
    main()
