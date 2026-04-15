#!/usr/bin/env python3
"""
Analyze CIR and SR changes across training steps for CIR-reward training, sweeping coefficients.

Data source (for α in {0.2,0.4,0.8}):
  /nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-3b-instruct/cot_importance_trained_{alpha}

Usage:
  python analysis/figure_7_cir_reward_trained.py
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Set font to serif for consistency
plt.rcParams['font.family'] = 'serif'


def get_available_steps(base_dir):
    """Automatically detect available training steps from directory."""
    if not os.path.exists(base_dir):
        return []

    steps = []
    for item in os.listdir(base_dir):
        if item.startswith('step_'):
            match = re.search(r'step_(\d+)', item)
            if match:
                steps.append(int(match.group(1)))

    return sorted(steps)


def load_metrics_at_step(base_dir, step):
    """Load CIR proxy (cot_importance), accuracy, and SR (verifier accuracy) for a specific training step."""
    json_path = os.path.join(base_dir, f"step_{step}", f"teacher_responses_step_{step}.json")

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
        # CIR proxy from JS divergences
        if 'cot_importance_evaluation' in item:
            eval_data = item['cot_importance_evaluation']
            js_divs = eval_data.get('js_divergences', [])
            if len(js_divs) >= 2:
                percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                sampled_values = []
                for p in percentages:
                    if p == 0:
                        idx = 0
                    else:
                        idx = max(0, min(int((p/100.0)*len(js_divs))-1, len(js_divs)-1))
                    sampled_values.append(js_divs[idx])
                cot_importance_values.append(np.mean(sampled_values))

        # Teacher accuracy
        if 'k_responses' in item:
            k_correct = []
            for k_response in item['k_responses']:
                if 'reward_score' in k_response:
                    k_correct.append(k_response['reward_score'])
            if k_correct:
                accuracies.append(np.mean(k_correct))

        # SR (verifier accuracy)
        if 'verifier_comparison' in item:
            verifier_comp = item['verifier_comparison']
            answers_match = verifier_comp.get('answers_match', False)

            with_q_answer = verifier_comp.get('with_question_answer', '').strip().lower()
            without_q_answer = verifier_comp.get('without_question_answer', '').strip().lower()

            if answers_match and with_q_answer == 'no answer found' and without_q_answer == 'no answer found':
                verifier_accuracies.append(0)
            else:
                verifier_accuracies.append(1 if answers_match else 0)

    if not cot_importance_values:
        return None

    return {
        'cot_importance': np.mean(cot_importance_values) if cot_importance_values else None,
        'accuracy': np.mean(accuracies) if accuracies else None,
        'verifier_accuracy': np.mean(verifier_accuracies) if verifier_accuracies else None,
        'count': len(cot_importance_values)
    }


def plot_metrics_progression(all_coeff_data, output_dir, tasks):
    """Plot CIR, SR, and Accuracy across training steps."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    colors = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c']  # Black for 0.0, then 0.2/0.4/0.8
    markers = ['X', 'o', 's', '^']

    # Plot 1: CIR
    for idx, (coeff_name, step_data) in enumerate(all_coeff_data):
        steps = []
        cot_values = []
        for step in sorted(step_data.keys()):
            if step_data[step]['cot_importance'] is not None:
                steps.append(step)
                cot_values.append(step_data[step]['cot_importance'])

        if steps and cot_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax1.plot(steps, cot_values, linewidth=3.0, color=color, marker=marker,
                     markersize=8, label=coeff_name, alpha=0.9)

    ax1.set_xlabel('Training Step', fontsize=25, fontweight='bold', family='serif')
    ax1.set_ylabel('CIR', fontsize=25, fontweight='bold', family='serif')
    ax1.set_title(f'CIR Across Training (CIR as reward; n={len(tasks)} tasks)',
                  fontsize=25, fontweight='bold', pad=20, family='serif')
    ax1.tick_params(axis='both', which='major', labelsize=19)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_family('serif')
    ax1.set_ylim(0, 1.0)
    ax1.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.legend(fontsize=19, loc='best', framealpha=0.9, prop={'family': 'serif'})
    ax1.set_axisbelow(True)

    # Plot 2: SR
    for idx, (coeff_name, step_data) in enumerate(all_coeff_data):
        steps = []
        verifier_values = []
        for step in sorted(step_data.keys()):
            if step_data[step]['verifier_accuracy'] is not None:
                steps.append(step)
                verifier_values.append(step_data[step]['verifier_accuracy'])

        if steps and verifier_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax2.plot(steps, verifier_values, linewidth=3.0, color=color, marker=marker,
                     markersize=8, label=coeff_name, alpha=0.9)

    ax2.set_xlabel('Training Step', fontsize=25, fontweight='bold', family='serif')
    ax2.set_ylabel('SR', fontsize=25, fontweight='bold', family='serif')
    ax2.set_title('SR Across Training (CIR as reward)',
                  fontsize=25, fontweight='bold', pad=20, family='serif')
    ax2.tick_params(axis='both', which='major', labelsize=19)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_family('serif')
    ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.legend(fontsize=19, loc='best', framealpha=0.9, prop={'family': 'serif'})
    ax2.set_axisbelow(True)

    # Plot 3: Accuracy
    for idx, (coeff_name, step_data) in enumerate(all_coeff_data):
        steps = []
        accuracy_values = []
        for step in sorted(step_data.keys()):
            if step_data[step]['accuracy'] is not None:
                steps.append(step)
                accuracy_values.append(step_data[step]['accuracy'])

        if steps and accuracy_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax3.plot(steps, accuracy_values, linewidth=3.0, color=color, marker=marker,
                     markersize=8, label=coeff_name, alpha=0.9)

    ax3.set_xlabel('Training Step', fontsize=25, fontweight='bold', family='serif')
    ax3.set_ylabel('Accuracy', fontsize=25, fontweight='bold', family='serif')
    ax3.set_title('Accuracy Across Training (CIR as reward)',
                  fontsize=25, fontweight='bold', pad=20, family='serif')
    ax3.tick_params(axis='both', which='major', labelsize=19)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_family('serif')
    ax3.set_ylim(0, 1.0)
    ax3.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.legend(fontsize=19, loc='best', framealpha=0.9, prop={'family': 'serif'})
    ax3.set_axisbelow(True)

    plt.tight_layout()

    png_path = os.path.join(output_dir, 'figure7_cir_reward_trained.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, 'figure7_cir_reward_trained.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.close()
    return png_path, pdf_path


def main():
    tasks = ['bitwise_arithmetic', "mini_sudoku", "futoshiki"]
    coefficients = [0.0, 0.2, 0.4, 0.8]
    steps_to_keep = [2, 30, 60, 90, 120, 150, 156]

    base_path = '/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-3b-instruct'
    coeff0_base_path = '/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-3b-instruct/cot_importance'

    all_coeff_data = []

    for coeff in coefficients:
        print(f"\n=== Processing α={coeff} ===")
        step_aggregated = {}

        for task in tasks:
            if coeff == 0.0:
                task_dir = f"{task}_cot_importance/-1.0/teacher"
                full_path = os.path.join(coeff0_base_path, task_dir)
            else:
                task_dir = f"cot_importance_trained_{coeff}/{task}/-1.0/teacher"
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
            cot_vals = [m['cot_importance'] for m in metrics_list if m['cot_importance'] is not None]
            acc_vals = [m['accuracy'] for m in metrics_list if m['accuracy'] is not None]
            ver_vals = [m['verifier_accuracy'] for m in metrics_list if m['verifier_accuracy'] is not None]
            averaged_step_data[step] = {
                'cot_importance': np.mean(cot_vals) if cot_vals else None,
                'accuracy': np.mean(acc_vals) if acc_vals else None,
                'verifier_accuracy': np.mean(ver_vals) if ver_vals else None
            }

        if averaged_step_data:
            all_coeff_data.append((f'α={coeff}', averaged_step_data))
            print(f"  Successfully averaged {len(averaged_step_data)} steps across tasks")
        else:
            print(f"  Warning: No valid data for α={coeff}")

    if not all_coeff_data:
        print("\nError: No valid data loaded")
        return

    output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis/graph'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    png_path, pdf_path = plot_metrics_progression(all_coeff_data, output_dir, tasks)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

