#!/usr/bin/env python3
"""
Plot the density distribution of CIR change (final - initial) across tasks.

Usage:
    python plot_cir_change_density.py [model_size]
    python plot_cir_change_density.py 3b
    python plot_cir_change_density.py 1.5b
    python plot_cir_change_density.py --local  # for local testing
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
    parser = argparse.ArgumentParser(description='Plot density of CIR change (final - initial)')
    parser.add_argument('model_size', nargs='?', default='3b', help='Model size (default: 3b)')
    parser.add_argument('--local', action='store_true', help='Use local paths (for testing)')

    args = parser.parse_args()

    # Define training steps
    initial_step = 2
    final_step = 156

    # Set base directories
    if args.local:
        base_dir = f'/Users/qinanyu/Desktop/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'
        output_dir = '/Users/qinanyu/Desktop/rl-explanations/analysis/graph'
    else:
        base_dir = f'/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{args.model_size}-instruct/cot_importance'
        output_dir = '/nlp/scr/qinanyu/rl-explanations/analysis/graph'

    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return

    # Collect data
    task_folders = [item.replace('_cot_importance', '')
                    for item in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, item))]

    cir_changes = []
    task_names = []

    for task_name in sorted(task_folders):
        # Special case: simple_geometry might use different steps
        initial_cir = load_cot_importance_at_step(base_dir, task_name, initial_step)
        final_cir = load_cot_importance_at_step(base_dir, task_name, final_step)

        if initial_cir is not None and final_cir is not None:
            change = final_cir - initial_cir
            cir_changes.append(change)
            task_names.append(task_name)
            print(f"{task_name}: {initial_cir:.4f} → {final_cir:.4f} (Δ={change:.4f})")

    if not cir_changes:
        print("No data found!")
        return

    print(f"\nTotal tasks with data: {len(cir_changes)}")
    print(f"Mean CIR change: {np.mean(cir_changes):.4f}")
    print(f"Median CIR change: {np.median(cir_changes):.4f}")
    print(f"Std CIR change: {np.std(cir_changes):.4f}")

    # Create the density plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create histogram with density
    n, bins, patches = ax.hist(cir_changes, bins=20, density=True, alpha=0.6,
                                color='#1f77b4', edgecolor='black', linewidth=1.5,
                                label='Histogram')

    # Overlay kernel density estimate (KDE)
    kde = stats.gaussian_kde(cir_changes)
    x_range = np.linspace(min(cir_changes), max(cir_changes), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=3, label='KDE', alpha=0.8)

    # Add vertical line at mean
    ax.axvline(np.mean(cir_changes), color='green', linestyle='--', linewidth=2.5,
               label=f'Mean = {np.mean(cir_changes):.3f}')

    # Add vertical line at median
    ax.axvline(np.median(cir_changes), color='orange', linestyle='--', linewidth=2.5,
               label=f'Median = {np.median(cir_changes):.3f}')

    # Add vertical line at zero
    ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.5,
               label='No change')

    # Formatting
    ax.set_xlabel('CIR Change (Final - Initial)', fontsize=25, fontweight='bold', family='serif')
    ax.set_ylabel('Density', fontsize=25, fontweight='bold', family='serif')
    ax.set_title(f'Distribution of CIR Change Across Training\n'
                 f'Steps: {initial_step} → {final_step} (n={len(cir_changes)} tasks)',
                 fontsize=25, fontweight='bold', pad=20, family='serif')

    # Set tick label fonts
    for label in ax.get_xticklabels():
        label.set_family('serif')
        label.set_fontsize(19)
    for label in ax.get_yticklabels():
        label.set_family('serif')
        label.set_fontsize(19)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='both', zorder=0)
    ax.legend(loc='best', fontsize=19, framealpha=0.9, prop={'family': 'serif'})

    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'cir_change_density_{args.model_size}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    pdf_path = os.path.join(output_dir, f'cir_change_density_{args.model_size}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"\nSaved plots:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()
