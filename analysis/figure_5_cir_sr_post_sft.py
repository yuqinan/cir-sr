#!/usr/bin/env python3
"""
Analyze CoT importance, accuracy, and verifier accuracy changes across training steps.
Averages metrics across multiple tasks for different k values.

Usage:
    python figure_5_cir_sr_post_sft.py [model_size]
    python figure_5_cir_sr_post_sft.py 3b
    python figure_5_cir_sr_post_sft.py 1.5b
"""

import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Set font to serif for consistency
plt.rcParams['font.family'] = 'serif'


def get_available_steps(base_dir, skip_step_2=True):
    """Automatically detect available training steps from directory.

    Args:
        base_dir: Directory to search for step folders
        skip_step_2: If True, skip step 2 (default). Set to False for k=0.
    """
    if not os.path.exists(base_dir):
        return []

    steps = []
    for item in os.listdir(base_dir):
        if item.startswith('step_'):
            match = re.search(r'step_(\d+)', item)
            if match:
                step_num = int(match.group(1))
                # Skip step 2 only if skip_step_2 is True
                if skip_step_2 and step_num == 2:
                    continue
                steps.append(step_num)

    return sorted(steps)


def load_metrics_at_step(base_dir, step):
    """Load CoT importance, accuracy, and verifier accuracy for a specific training step."""
    json_path = os.path.join(base_dir, f"step_{step}", f"teacher_responses_step_{step}.json")

    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except:
        return None

    cot_importance_values = []
    accuracies = []
    verifier_accuracies = []

    for item in data:
        # Extract CoT importance - sample at 0%, 10%, 20%, ..., 100% positions
        if 'cot_importance_evaluation' in item:
            eval_data = item['cot_importance_evaluation']
            js_divs = eval_data.get('js_divergences', [])
            if len(js_divs) >= 2:
                # Sample at 11 positions: 0%, 10%, 20%, ..., 100%
                percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                sampled_values = []
                for p in percentages:
                    idx = max(0, min(int((p/100.0)*len(js_divs))-1, len(js_divs)-1))
                    # Special handling for 0% - use index 0
                    if p == 0:
                        idx = 0
                    sampled_values.append(js_divs[idx])
                # Average the 11 sampled values
                cot_importance_values.append(np.mean(sampled_values))

        # Extract accuracy from k_responses
        if 'k_responses' in item:
            k_correct = []
            for k_response in item['k_responses']:
                if 'reward_score' in k_response:
                    k_correct.append(k_response['reward_score'])
            if k_correct:
                accuracies.append(np.mean(k_correct))

        # Extract verifier accuracy
        if 'verifier_comparison' in item:
            verifier_accuracies.append(item['verifier_comparison']['answers_match'] and item['verifier_comparison']['answers_match_original'])

    if not cot_importance_values:
        return None

    return {
        'cot_importance': np.mean(cot_importance_values) if cot_importance_values else None,
        'accuracy': np.mean(accuracies) if accuracies else None,
        'verifier_accuracy': np.mean(verifier_accuracies) if verifier_accuracies else None,
        'count': len(cot_importance_values)
    }


def plot_metrics_progression(all_k_data, output_dir, tasks, model_size):
    """Plot CoT importance, accuracy, and verifier accuracy across training steps."""

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

    # More visually distinct color palette
    colors = ['#2E2E2E', '#1E88E5', '#FFA726', '#66BB6A', '#EF5300', '#AB47BC', '#8D6E63']
    markers = ['X', 'o', 's', '^', 'D', 'v', 'p']  # X marker for k=0

    # Plot 1: CoT Importance
    for idx, (k_name, step_data) in enumerate(all_k_data):
        steps = []
        cot_values = []
        for step in sorted(step_data.keys()):
            if step_data[step]['cot_importance'] is not None:
                steps.append(step)
                cot_values.append(step_data[step]['cot_importance'])

        if steps and cot_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            # Change k= to N= in label
            label = k_name.replace('k=', 'N=')
            ax1.plot(steps, cot_values, linewidth=5, color=color, marker=marker,
                     markersize=6, label=label, alpha=0.85, markeredgewidth=1.5,
                     markeredgecolor='white')

    ax1.set_xlabel('RL Training Step', fontsize=30, fontweight='bold', family='serif')
    ax1.set_ylabel('CIR', fontsize=30, fontweight='bold', family='serif')
    ax1.set_title('CIR Across Training',
                  fontsize=30, fontweight='bold', pad=15, family='serif')
    ax1.tick_params(axis='both', which='major', labelsize=18, width=1.5)
    ax1.tick_params(axis='both', which='minor', labelsize=16, width=1)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_family('serif')
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
    ax1.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    ax1.minorticks_on()
    ax1.legend(fontsize=40, loc='best', framealpha=0.95, prop={'family': 'serif'},
               edgecolor='gray', fancybox=True, shadow=True, markerscale=2)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    # Plot 2: Verifier Accuracy
    for idx, (k_name, step_data) in enumerate(all_k_data):
        steps = []
        verifier_values = []
        for step in sorted(step_data.keys()):
            if step_data[step]['verifier_accuracy'] is not None:
                steps.append(step)
                verifier_values.append(step_data[step]['verifier_accuracy'])

        if steps and verifier_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            # Change k= to N= in label
            label = k_name.replace('k=', 'N=')
            ax2.plot(steps, verifier_values, linewidth=5, color=color, marker=marker,
                     markersize=6, label=label, alpha=0.85, markeredgewidth=1.5,
                     markeredgecolor='white')

    ax2.set_xlabel('RL Training Step', fontsize=30, fontweight='bold', family='serif')
    ax2.set_ylabel('SR', fontsize=30, fontweight='bold', family='serif')
    ax2.set_title('SR Across Training',
                  fontsize=30, fontweight='bold', pad=15, family='serif')
    ax2.tick_params(axis='both', which='major', labelsize=18, width=1.5)
    ax2.tick_params(axis='both', which='minor', labelsize=16, width=1)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_family('serif')
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
    ax2.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    ax2.minorticks_on()
    ax2.legend(fontsize=36, loc='best', framealpha=0.95, prop={'family': 'serif'},
               edgecolor='gray', fancybox=True, shadow=True, markerscale=2)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    # Plot 3: Accuracy
    for idx, (k_name, step_data) in enumerate(all_k_data):
        steps = []
        accuracy_values = []
        for step in sorted(step_data.keys()):
            if step_data[step]['accuracy'] is not None:
                steps.append(step)
                accuracy_values.append(step_data[step]['accuracy'])

        if steps and accuracy_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            # Change k= to N= in label
            label = k_name.replace('k=', 'N=')
            ax3.plot(steps, accuracy_values, linewidth=5, color=color, marker=marker,
                     markersize=6, label=label, alpha=0.85, markeredgewidth=1.5,
                     markeredgecolor='white')

    ax3.set_xlabel('RL Training Step', fontsize=30, fontweight='bold', family='serif')
    ax3.set_ylabel('Accuracy', fontsize=30, fontweight='bold', family='serif')
    ax3.set_title('Accuracy Across Training',
                  fontsize=30, fontweight='bold', pad=15, family='serif')
    ax3.tick_params(axis='both', which='major', labelsize=18, width=1.5)
    ax3.tick_params(axis='both', which='minor', labelsize=16, width=1)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_family('serif')
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
    ax3.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    ax3.minorticks_on()
    ax3.legend(fontsize=36, loc='best', framealpha=0.95, prop={'family': 'serif'},
               edgecolor='gray', fancybox=True, shadow=True, markerscale=2)
    ax3.set_axisbelow(True)
    for spine in ax3.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout(pad=2.0)

    png_path = os.path.join(output_dir, f'figure5_cir_sr_post_sft_{model_size}_new.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, f'figure5_cir_sr_post_sft_{model_size}_new.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.close()

    return png_path, pdf_path


def plot_individual_tasks(task_k_data, output_dir, tasks, k_values, model_size):
    """Plot individual task metrics in a grid layout (dynamic based on number of tasks)."""

    # Dynamically determine grid size based on number of tasks
    num_tasks = len(tasks)
    # Each row is 4 inches tall, minimum height of 12
    fig_height = max(12, num_tasks * 4)

    fig, axes = plt.subplots(num_tasks, 3, figsize=(30, fig_height))

    # Handle case where there's only one task (axes would be 1D instead of 2D)
    if num_tasks == 1:
        axes = axes.reshape(1, -1)

    # More visually distinct color palette (same as main plot)
    colors = ['#2E2E2E', '#1E88E5', '#FFA726', '#66BB6A', '#EF5300', '#AB47BC', '#8D6E63']
    markers = ['X', 'o', 's', '^', 'D', 'v', 'p']  # X marker for k=0

    for task_idx, task in enumerate(tasks):
        # Plot 1: CoT Importance (CIR)
        ax_cir = axes[task_idx, 0]
        for k_idx, k in enumerate(k_values):
            if task in task_k_data and k in task_k_data[task]:
                step_data = task_k_data[task][k]
                steps = []
                cot_values = []
                for step in sorted(step_data.keys()):
                    if step_data[step]['cot_importance'] is not None:
                        steps.append(step)
                        cot_values.append(step_data[step]['cot_importance'])

                if steps and cot_values:
                    color = colors[k_idx % len(colors)]
                    marker = markers[k_idx % len(markers)]
                    ax_cir.plot(steps, cot_values, linewidth=2.5, color=color, marker=marker,
                               markersize=7, label=f'N={k}', alpha=0.85, markeredgewidth=1.2,
                               markeredgecolor='white')

        ax_cir.set_ylabel('CIR', fontsize=20, fontweight='bold', family='serif')
        ax_cir.set_title(f'{task.replace("_", " ").title()}', fontsize=22, fontweight='bold', family='serif', pad=15)
        ax_cir.tick_params(axis='both', which='major', labelsize=16, width=1.2)
        for label in ax_cir.get_xticklabels() + ax_cir.get_yticklabels():
            label.set_family('serif')
        ax_cir.set_ylim(-0.02, 1.02)
        ax_cir.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_cir.minorticks_on()
        if task_idx == 0:
            ax_cir.legend(fontsize=30, loc='best', framealpha=0.95, prop={'family': 'serif'},
                         edgecolor='gray', fancybox=True, shadow=True, markerscale=2)
        ax_cir.set_axisbelow(True)
        for spine in ax_cir.spines.values():
            spine.set_linewidth(1.2)
        if task_idx == len(tasks) - 1:
            ax_cir.set_xlabel('RL Training Step', fontsize=20, fontweight='bold', family='serif')

        # Plot 2: Verifier Accuracy (SR)
        ax_sr = axes[task_idx, 1]
        for k_idx, k in enumerate(k_values):
            if task in task_k_data and k in task_k_data[task]:
                step_data = task_k_data[task][k]
                steps = []
                verifier_values = []
                for step in sorted(step_data.keys()):
                    if step_data[step]['verifier_accuracy'] is not None:
                        steps.append(step)
                        verifier_values.append(step_data[step]['verifier_accuracy'])

                if steps and verifier_values:
                    color = colors[k_idx % len(colors)]
                    marker = markers[k_idx % len(markers)]
                    ax_sr.plot(steps, verifier_values, linewidth=2.5, color=color, marker=marker,
                             markersize=7, label=f'N={k}', alpha=0.85, markeredgewidth=1.2,
                             markeredgecolor='white')

        ax_sr.set_ylabel('SR', fontsize=20, fontweight='bold', family='serif')
        ax_sr.tick_params(axis='both', which='major', labelsize=16, width=1.2)
        for label in ax_sr.get_xticklabels() + ax_sr.get_yticklabels():
            label.set_family('serif')
        ax_sr.set_ylim(-0.02, 1.02)
        ax_sr.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_sr.minorticks_on()
        ax_sr.set_axisbelow(True)
        for spine in ax_sr.spines.values():
            spine.set_linewidth(1.2)
        if task_idx == len(tasks) - 1:
            ax_sr.set_xlabel('RL Training Step', fontsize=20, fontweight='bold', family='serif')

        # Plot 3: Accuracy
        ax_acc = axes[task_idx, 2]
        for k_idx, k in enumerate(k_values):
            if task in task_k_data and k in task_k_data[task]:
                step_data = task_k_data[task][k]
                steps = []
                accuracy_values = []
                for step in sorted(step_data.keys()):
                    if step_data[step]['accuracy'] is not None:
                        steps.append(step)
                        accuracy_values.append(step_data[step]['accuracy'])

                if steps and accuracy_values:
                    color = colors[k_idx % len(colors)]
                    marker = markers[k_idx % len(markers)]
                    ax_acc.plot(steps, accuracy_values, linewidth=2.5, color=color, marker=marker,
                              markersize=7, label=f'N={k}', alpha=0.85, markeredgewidth=1.2,
                              markeredgecolor='white')

        ax_acc.set_ylabel('Accuracy', fontsize=20, fontweight='bold', family='serif')
        ax_acc.tick_params(axis='both', which='major', labelsize=16, width=1.2)
        for label in ax_acc.get_xticklabels() + ax_acc.get_yticklabels():
            label.set_family('serif')
        ax_acc.set_ylim(-0.02, 1.02)
        ax_acc.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, which='major')
        ax_acc.minorticks_on()
        ax_acc.set_axisbelow(True)
        for spine in ax_acc.spines.values():
            spine.set_linewidth(1.2)
        if task_idx == len(tasks) - 1:
            ax_acc.set_xlabel('RL Training Step', fontsize=20, fontweight='bold', family='serif')

    plt.tight_layout()

    png_path = os.path.join(output_dir, f'figure5_extra_{model_size}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, f'figure5_extra_{model_size}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.close()

    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description='Plot CoT importance and verifier accuracy for post-SFT training')
    parser.add_argument('model_size', nargs='?', default='3b', help='Model size (default: 3b)')
    args = parser.parse_args()

    model_size = args.model_size

    # Define tasks to average over - add/remove tasks here and subgraphs will adjust automatically
    tasks = ['count_bits', 'bitwise_arithmetic', 'tsumego', 'mini_sudoku', "binary_matrix", "rotate_matrix", "manipulate_matrix"]

    # Define k values to compare (k=0 uses different path structure)
    k_values = [0, 2, 8, 64, 512]

    # Only keep specific training steps
    steps_to_keep = [0, 30, 60, 90, 120, 156]

    base_path = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_post_sft_qwen-{model_size}-instruct'
    # For k=0, use the same base as plot_cot_importance_change.py
    k0_base_path = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct/cot_importance'

    print(f"\nProcessing model: qwen-{model_size}-instruct")
    print(f"Processing {len(tasks)} tasks: {tasks}")

    # Collect data for each k value, averaged across tasks
    all_k_data = []
    # Also collect individual task data for figure5_extra
    task_k_data = {task: {} for task in tasks}

    for k in k_values:
        print(f"\n=== Processing k={k} ===")

        # Dictionary to store aggregated data: step -> list of metric dicts
        step_aggregated = {}

        for task in tasks:
            # Special handling for k=0: use different path structure
            if k == 0:
                task_dir = f"{task}_cot_importance/-1.0/teacher"
                full_path = os.path.join(k0_base_path, task_dir)
            else:
                task_dir = f"cot_importance_{k}/{task}_cot_importance_{k}/-1.0/teacher"
                full_path = os.path.join(base_path, task_dir)

            if not os.path.exists(full_path):
                print(f"  Warning: Path not found for {task}: {full_path}")
                continue

            # For k=0, don't skip step 2 since we need it to map to step 0
            training_steps = get_available_steps(full_path, skip_step_2=(k != 0))
            if not training_steps:
                print(f"  Warning: No training steps found for {task}")
                continue

            print(f"  Processing {task}: found {len(training_steps)} steps")

            # Collect individual task data
            task_step_data = {}
            for step in training_steps:
                # For k=0, treat step_2 as step_0
                actual_step = 0 if (k == 0 and step == 2) else step

                # Only keep specific training steps
                if actual_step not in steps_to_keep:
                    continue

                metrics = load_metrics_at_step(full_path, step)
                if metrics:
                    if actual_step not in step_aggregated:
                        step_aggregated[actual_step] = []
                    step_aggregated[actual_step].append(metrics)

                    # Store for individual task plotting
                    task_step_data[actual_step] = metrics

            # Store individual task data
            if task_step_data:
                task_k_data[task][k] = task_step_data

        # Average metrics across tasks for each step
        averaged_step_data = {}
        for step, metrics_list in step_aggregated.items():
            cot_vals = [m['cot_importance'] for m in metrics_list if m['cot_importance'] is not None]
            acc_vals = [m['accuracy'] for m in metrics_list if m['accuracy'] is not None]
            ver_vals = [m['verifier_accuracy'] for m in metrics_list if m['verifier_accuracy'] is not None]

            averaged_step_data[step] = {
                'cot_importance': np.mean(cot_vals) if cot_vals else None,
                'accuracy': np.mean(acc_vals) if acc_vals else None,
                'verifier_accuracy': np.mean(ver_vals) if ver_vals else None
            }

        if averaged_step_data:
            all_k_data.append((f'k={k}', averaged_step_data))
            print(f"  Successfully averaged {len(averaged_step_data)} steps across tasks")
        else:
            print(f"  Warning: No valid data for k={k}")

    if not all_k_data:
        print("\nError: No valid data loaded")
        return

    output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis/graph'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    png_path, pdf_path = plot_metrics_progression(all_k_data, output_dir, tasks, model_size)

    if os.path.exists(png_path):
        print(f"Successfully saved: {png_path}")
    else:
        print(f"Error: Failed to save {png_path}")

    if os.path.exists(pdf_path):
        print(f"Successfully saved: {pdf_path}")
    else:
        print(f"Error: Failed to save {pdf_path}")

    # NOTE: Your local path (e.g. /Users/...) is not accessible from the cluster.
    # Instead, we print a helper command you can run locally to fetch the PDF.
    local_output_dir = "/Users/qinanyu/Desktop/690d3788eb576bbb6d979e66/graph"
    if os.path.exists(pdf_path):
        local_fetch_hint = (
            f"# Run this on your *local machine* to copy the PDF into your local graph folder:\n"
            f"mkdir -p \"{local_output_dir}\" && \\\n"
            f"scp qinanyu@scdt.stanford.edu:{pdf_path} \"{local_output_dir}/\"\n"
        )
        print("\n[LOCAL DOWNLOAD COMMAND]\n" + local_fetch_hint)

    # Generate individual task plots (figure5_extra)
    print(f"\n=== Generating individual task plots ===")
    extra_png_path, extra_pdf_path = plot_individual_tasks(task_k_data, output_dir, tasks, k_values, model_size)

    if os.path.exists(extra_png_path):
        print(f"Successfully saved: {extra_png_path}")
    else:
        print(f"Error: Failed to save {extra_png_path}")

    if os.path.exists(extra_pdf_path):
        print(f"Successfully saved: {extra_pdf_path}")
    else:
        print(f"Error: Failed to save {extra_pdf_path}")


if __name__ == "__main__":
    main()
