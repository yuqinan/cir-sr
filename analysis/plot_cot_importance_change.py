#!/usr/bin/env python3
"""
Plot initial and final CoT importance for each task as vertical lines with dots.

Usage:
    python plot_cot_importance_change.py [model_size]
    python plot_cot_importance_change.py 3b
    python plot_cot_importance_change.py 1.5b
    python plot_cot_importance_change.py --local  # for local testing
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


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


def load_step_accuracy(base_dir, task_name, step, variant="original", batch="8"):
    """Load accuracy for a specific step."""
    rollout_dir = Path(base_dir) / task_name / variant / variant / batch
    if not rollout_dir.exists():
        return None
    step_file = rollout_dir / f"{step}.jsonl"
    if not step_file.exists():
        return None
    try:
        entries = load_jsonl_file(step_file)
        return calculate_average_accuracy(entries)
    except:
        return None


def load_step_data(base_dir, task_name, step):
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

                # Check if number of examples is less than 95
                if len(data) < 95:
                    return None

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


def main():
    parser = argparse.ArgumentParser(description='Plot CoT importance change across training steps')
    parser.add_argument('model_size', nargs='?', default='3b', help='Model size (default: 3b)')
    parser.add_argument('--local', action='store_true', help='Use local paths (for testing)')
    parser.add_argument('--model', type=str, default='q3b-instruct', help='Model name for rollout data (default: q3b-instruct)')
    parser.add_argument('--thinking_dir', type=str, default='grpo', help='Directory name for thinking variant')
    parser.add_argument('--variant', type=str, default='original', help='Variant folder name (default: original)')
    parser.add_argument('--batch', type=str, default='8', help='Batch folder name (default: 8)')

    args = parser.parse_args()

    # Define training steps
    training_steps = [2, 30, 60, 90, 120, 156]

    # Set base directories
    if args.local:
        base_dir = f'/Users/qinanyu/Desktop/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'
        rollout_base = f'/Users/qinanyu/Desktop/rl-explanations/trainers/{args.thinking_dir}/{args.model}/rollout_data'
        output_dir = '/Users/qinanyu/Desktop/rl-explanations/analysis/graph'
    else:
        base_dir = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'
        rollout_base = f'/nlp/scr/qinanyu/rl-explanations/trainers/{args.thinking_dir}/{args.model}/rollout_data'
        output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis/graph'

    if not os.path.exists(base_dir):
        return

    # Collect data
    task_data = {}
    task_folders = [item.replace('_cot_importance', '')
                    for item in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, item))]

    for task_name in sorted(task_folders):
        # Load CoT importance for all steps
        # Special case: simple_geometry uses steps 40 and 100 instead of 30 and 90
        if task_name == 'simple_geometry':
            task_training_steps = [2, 40, 60, 100, 120, 156]
        else:
            task_training_steps = training_steps

        step_values = {}
        for step in task_training_steps:
            cot_importance = load_step_data(base_dir, task_name, step)
            if cot_importance is not None:
                step_values[step] = cot_importance

        # Load final thinking accuracy for sorting (use step 156)
        thinking_acc = load_step_accuracy(rollout_base, task_name, 156, args.variant, args.batch)

        # Check for missing steps and print them
        if len(step_values) > 0:
            missing_steps = [step for step in task_training_steps if step not in step_values]
            if missing_steps:
                print(f"Task '{task_name}': Missing steps {missing_steps}")

        # Only include tasks that have data for at least first and last step
        if (2 in step_values or 156 in step_values or 150 in step_values) and thinking_acc is not None:
            task_data[task_name] = {
                'steps': step_values,
                'thinking_end': thinking_acc,
                'training_steps': task_training_steps
            }

    if not task_data:
        return

    # Sort tasks by initial CoT importance (lowest to highest)
    def get_initial_cot(item):
        task_name, data = item
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 1:
            return steps[available_steps[0]]
        return 0

    sorted_tasks = sorted(task_data.items(), key=get_initial_cot)
    task_names = [task for task, _ in sorted_tasks]

    # Calculate Pearson correlation between initial and final CIR
    initial_cir_values = []
    final_cir_values = []
    for task_name, data in task_data.items():
        steps = data['steps']
        available_steps = sorted(steps.keys())
        if len(available_steps) >= 2:
            initial_cir_values.append(steps[available_steps[0]])
            final_cir_values.append(steps[available_steps[-1]])

    if len(initial_cir_values) >= 2:
        pearson_r, pearson_p = stats.pearsonr(initial_cir_values, final_cir_values)
        spearman_r, spearman_p = stats.spearmanr(initial_cir_values, final_cir_values)

        # Calculate percentage of tasks that increased vs decreased
        tasks_increased = sum(1 for init, final in zip(initial_cir_values, final_cir_values) if final > init)
        tasks_decreased = sum(1 for init, final in zip(initial_cir_values, final_cir_values) if final < init)
        tasks_unchanged = sum(1 for init, final in zip(initial_cir_values, final_cir_values) if final == init)
        total_tasks = len(initial_cir_values)

        print(f"\n{'='*80}")
        print(f"CIR CORRELATION ANALYSIS")
        print(f"{'='*80}")
        print(f"Total tasks: {total_tasks}")
        print(f"Tasks with CIR increase: {tasks_increased}/{total_tasks} ({100*tasks_increased/total_tasks:.1f}%)")
        print(f"Tasks with CIR decrease: {tasks_decreased}/{total_tasks} ({100*tasks_decreased/total_tasks:.1f}%)")
        if tasks_unchanged > 0:
            print(f"Tasks with CIR unchanged: {tasks_unchanged}/{total_tasks} ({100*tasks_unchanged/total_tasks:.1f}%)")

        print(f"\nPearson correlation (initial CIR vs final CIR):")
        print(f"  r = {pearson_r:.4f}")
        print(f"  p-value = {pearson_p:.4e}")
        print(f"  n = {len(initial_cir_values)} tasks")

        # Determine significance for Pearson
        if pearson_p < 0.001:
            pearson_sig = "***"
        elif pearson_p < 0.01:
            pearson_sig = "**"
        elif pearson_p < 0.05:
            pearson_sig = "*"
        else:
            pearson_sig = "n.s."
        print(f"  Significance: {pearson_sig}")

        print(f"\nSpearman correlation (initial CIR vs final CIR):")
        print(f"  ρ = {spearman_r:.4f}")
        print(f"  p-value = {spearman_p:.4e}")

        # Determine significance for Spearman
        if spearman_p < 0.001:
            spearman_sig = "***"
        elif spearman_p < 0.01:
            spearman_sig = "**"
        elif spearman_p < 0.05:
            spearman_sig = "*"
        else:
            spearman_sig = "n.s."
        print(f"  Significance: {spearman_sig}")

        # Count tasks that start with CIR <= 0.15
        low_start_tasks_015 = []
        low_start_low_end_tasks_015 = []
        for i, (init_cir, final_cir) in enumerate(zip(initial_cir_values, final_cir_values)):
            if init_cir <= 0.15:
                low_start_tasks_015.append((list(task_data.keys())[i], init_cir, final_cir))
                if final_cir < 0.15:
                    low_start_low_end_tasks_015.append((list(task_data.keys())[i], init_cir, final_cir))

        print(f"\nTasks starting with CIR ≤ 0.15:")
        print(f"  Total: {len(low_start_tasks_015)} tasks")
        print(f"  Tasks remaining under 0.15 at end: {len(low_start_low_end_tasks_015)}/{len(low_start_tasks_015)}")
        if low_start_tasks_015:
            print(f"\n  Task details:")
            for task_name, init_cir, final_cir in low_start_tasks_015:
                status = "✓" if final_cir < 0.15 else "✗"
                print(f"    {status} {task_name}: {init_cir:.4f} → {final_cir:.4f}")

        # Count tasks that start with CIR <= 0.2
        low_start_tasks_020 = []
        low_start_low_end_tasks_020 = []
        for i, (init_cir, final_cir) in enumerate(zip(initial_cir_values, final_cir_values)):
            if init_cir <= 0.2:
                low_start_tasks_020.append((list(task_data.keys())[i], init_cir, final_cir))
                if final_cir < 0.2:
                    low_start_low_end_tasks_020.append((list(task_data.keys())[i], init_cir, final_cir))

        print(f"\nTasks starting with CIR ≤ 0.20:")
        print(f"  Total: {len(low_start_tasks_020)} tasks")
        print(f"  Tasks remaining under 0.20 at end: {len(low_start_low_end_tasks_020)}/{len(low_start_tasks_020)}")
        if low_start_tasks_020:
            print(f"\n  Task details:")
            for task_name, init_cir, final_cir in low_start_tasks_020:
                status = "✓" if final_cir < 0.2 else "✗"
                print(f"    {status} {task_name}: {init_cir:.4f} → {final_cir:.4f}")

        print(f"{'='*80}\n")

        low_start_count_015 = len(low_start_tasks_015)
        low_start_low_end_count_015 = len(low_start_low_end_tasks_015)
        low_start_count_020 = len(low_start_tasks_020)
        low_start_low_end_count_020 = len(low_start_low_end_tasks_020)
    else:
        pearson_r = None
        pearson_p = None
        spearman_r = None
        spearman_p = None
        low_start_count_015 = 0
        low_start_low_end_count_015 = 0
        low_start_count_020 = 0
        low_start_low_end_count_020 = 0

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot lines and dots for each task
    x_positions = np.arange(len(task_names))

    # Create blue gradient (light to dark)
    num_steps = len(training_steps)
    blue_gradient = [
        (0.678, 0.847, 0.902),  # Light blue for step 2
        (0.529, 0.808, 0.922),  # Step 30
        (0.416, 0.678, 0.839),  # Step 60
        (0.282, 0.525, 0.671),  # Step 90
        (0.176, 0.396, 0.573),  # Step 120
        (0.098, 0.275, 0.471),  # Dark blue for step 156
    ]

    for i, (task, data) in enumerate(zip(task_names, [d for _, d in sorted_tasks])):
        step_values = data['steps']
        task_training_steps = data['training_steps']

        # Get all available step values for this task
        available_steps = []
        y_values = []
        colors = []
        for step_idx, step in enumerate(task_training_steps):
            if step in step_values:
                available_steps.append(step)
                y_values.append(step_values[step])
                colors.append(blue_gradient[step_idx])

        # Determine if increasing or decreasing
        if len(available_steps) > 1:
            first_val = y_values[0]
            last_val = y_values[-1]
            is_increasing = last_val > first_val
            line_color = '#2E86AB' if is_increasing else '#D32F2F'  # Blue for increase, red for decrease

            # Draw line with multiple chevron arrows
            num_chevrons = 5  # Number of chevron arrows along the line
            for idx in range(num_chevrons):
                # Calculate start and end positions for each chevron segment
                start_frac = idx / num_chevrons
                end_frac = (idx + 1) / num_chevrons

                y_start = y_values[0] + start_frac * (y_values[-1] - y_values[0])
                y_end = y_values[0] + end_frac * (y_values[-1] - y_values[0])

                # Draw chevron arrow segment
                ax.annotate('', xy=(i, y_end), xytext=(i, y_start),
                           arrowprops=dict(arrowstyle='->', color=line_color, lw=2.5, alpha=0.7,
                                         mutation_scale=15), zorder=2)

        # Plot dots at each available step with gradient colors
        for step_idx, (step, y_val, color) in enumerate(zip(available_steps, y_values, colors)):
            # Only add label once per step (for first task)
            ax.scatter(i, y_val, s=80, color=color,
                       edgecolors='black', linewidth=1.5, zorder=3,
                       label=f'Step {step}' if i == 0 else '')

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(task_names, rotation=70, ha='right', fontsize=19, family='serif')
    ax.set_xlabel('Tasks (sorted by initial CoT importance: lowest → highest)',
                  fontsize=25, fontweight='bold', family='serif')
    ax.set_ylabel('CoT Importance',
                  fontsize=25, fontweight='bold', family='serif')

    ax.set_title(f'Evolution of CoT Importance Across Training\n' +
                 f'Steps: {", ".join(map(str, training_steps))} (n={len(task_names)} tasks)',
                 fontsize=25, fontweight='bold', pad=20, family='serif')

    # Set y-axis tick labels to serif font
    for label in ax.get_yticklabels():
        label.set_family('serif')
        label.set_fontsize(19)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y', zorder=0)
    ax.legend(loc='upper right', fontsize=19, framealpha=0.9, prop={'family': 'serif'})

    # Add correlation text box if correlation was calculated
    if pearson_r is not None:
        # Determine significance for Pearson
        if pearson_p < 0.001:
            pearson_sig = "***"
        elif pearson_p < 0.01:
            pearson_sig = "**"
        elif pearson_p < 0.05:
            pearson_sig = "*"
        else:
            pearson_sig = "n.s."

        # Determine significance for Spearman
        if spearman_p < 0.001:
            spearman_sig = "***"
        elif spearman_p < 0.01:
            spearman_sig = "**"
        elif spearman_p < 0.05:
            spearman_sig = "*"
        else:
            spearman_sig = "n.s."

        corr_text = f'Correlation (start vs end):\n'
        corr_text += f'Pearson: r = {pearson_r:.3f} {pearson_sig}\n'
        corr_text += f'Spearman: ρ = {spearman_r:.3f} {spearman_sig}\n\n'
        corr_text += f'CIR ≤ 0.15 at start: {low_start_count_015}\n'
        corr_text += f'  Remaining < 0.15: {low_start_low_end_count_015}/{low_start_count_015}\n\n'
        corr_text += f'CIR ≤ 0.20 at start: {low_start_count_020}\n'
        corr_text += f'  Remaining < 0.20: {low_start_low_end_count_020}/{low_start_count_020}'
        ax.text(0.02, 0.98, corr_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                fontsize=16, family='serif')

    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'figure2_cot_importance_{args.model_size}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, f'figure2_cot_importance_{args.model_size}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
