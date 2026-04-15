#!/usr/bin/env python3
"""
Plot average accuracy increase for tasks categorized by CIR and SR changes.

Categories:
1. CIR ↑ & SR ↑
2. CIR ↑ & SR ↓
3. CIR ↓ & SR ↑
4. CIR ↓ & SR ↓

Usage:
    python plot_cir_sr_categories.py
    python plot_cir_sr_categories.py --model_size 3b
"""

import json
import os
import sys
import numpy as np
from pathlib import Path


def load_task_categories():
    """Load task categories from task_category.json."""
    category_path = Path(__file__).parent / 'task_category.json'
    if not category_path.exists():
        print(f"Warning: task_category.json not found at {category_path}")
        return {}

    with open(category_path, 'r') as f:
        return json.load(f)


def is_math_task(task_name, task_categories):
    """Check if a task is math-related (Algebra, Arithmetic, or Geometry)."""
    math_categories = ['Algebra', 'Arithmetic', 'Geometry']
    for category in math_categories:
        if task_name in task_categories.get(category, []):
            return True, category
    return False, None


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
    """Load average verifier accuracy from verifier_comparison.answers_match."""
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


def main():
    model_size = sys.argv[1] if len(sys.argv) > 1 else "3b"
    base_dir = f"/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-{model_size}-instruct/cot_importance"

    print(f"Loading data for model: {model_size}")
    print(f"Base directory: {base_dir}")

    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return

    # Load task categories
    task_categories = load_task_categories()

    # Collect data
    task_folders = [item.replace('_cot_importance', '')
                    for item in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, item))]

    start_step = 2
    end_step = 156

    # Categorize tasks - store tuples of (acc_change, task_name, is_math, math_category)
    categories = {
        'CIR↑ & SR↑': [],
        'CIR↑ & SR↓': [],
        'CIR↓ & SR↑': [],
        'CIR↓ & SR↓': []
    }

    for task_name in sorted(task_folders):
        # Load initial and final values
        initial_cir = load_cot_importance_at_step(base_dir, task_name, start_step)
        final_cir = load_cot_importance_at_step(base_dir, task_name, end_step)

        initial_sr = load_verifier_accuracy_at_step(base_dir, task_name, start_step)
        final_sr = load_verifier_accuracy_at_step(base_dir, task_name, end_step)

        initial_acc = load_step_accuracy_from_evaluate(base_dir, task_name, start_step)
        final_acc = load_step_accuracy_from_evaluate(base_dir, task_name, end_step)

        # Check if all data is available
        if (initial_cir is not None and final_cir is not None and
            initial_sr is not None and final_sr is not None and
            initial_acc is not None and final_acc is not None):

            cir_change = final_cir - initial_cir
            sr_change = final_sr - initial_sr
            acc_change = final_acc - initial_acc

            # Check if task is math-related
            is_math, math_category = is_math_task(task_name, task_categories)

            # Categorize - store (acc_change, task_name, is_math, math_category)
            if cir_change > 0 and sr_change > 0:
                categories['CIR↑ & SR↑'].append((acc_change, task_name, is_math, math_category))
            elif cir_change > 0 and sr_change <= 0:
                categories['CIR↑ & SR↓'].append((acc_change, task_name, is_math, math_category))
            elif cir_change <= 0 and sr_change > 0:
                categories['CIR↓ & SR↑'].append((acc_change, task_name, is_math, math_category))
            else:
                categories['CIR↓ & SR↓'].append((acc_change, task_name, is_math, math_category))

    # Print statistics
    print("\n" + "=" * 80)
    print("TASK CATEGORIZATION BY CIR & SR CHANGES")
    print("=" * 80)

    total_tasks = sum(len(tasks) for tasks in categories.values())
    print(f"\nTotal tasks: {total_tasks}")

    for cat_name, task_data in categories.items():
        if task_data:
            acc_changes = [acc for acc, _, _, _ in task_data]
            avg_acc_change = np.mean(acc_changes)

            # Count math tasks
            math_tasks = [(task_name, math_cat) for _, task_name, is_math, math_cat in task_data if is_math]

            print(f"\n{cat_name}:")
            print(f"  Tasks: {len(task_data)}")
            print(f"  Math tasks: {len(math_tasks)}")
            print(f"  Avg accuracy change: {avg_acc_change:.4f}")

            if math_tasks:
                print(f"  Math task details:")
                for task_name, math_cat in sorted(math_tasks):
                    print(f"    - {task_name} ({math_cat})")
        else:
            print(f"\n{cat_name}:")
            print(f"  Tasks: 0")

    print("\n" + "=" * 80)

    # Generate LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    print()

    latex_table = r"""\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
Category & Number of Tasks & Avg Accuracy Change \\
\hline
"""

    cat_labels = list(categories.keys())
    for cat_name in cat_labels:
        task_data = categories[cat_name]
        if task_data:
            acc_changes = [acc for acc, _, _, _ in task_data]
            avg_acc_change = np.mean(acc_changes)
            num_tasks = len(task_data)
        else:
            avg_acc_change = 0.0
            num_tasks = 0

        # Escape special characters for LaTeX
        cat_label = cat_name.replace('↑', r'$\uparrow$').replace('↓', r'$\downarrow$')
        latex_table += f"{cat_label} & {num_tasks} & {avg_acc_change:.4f} \\\\\n"

    latex_table += r"""\hline
\end{tabular}
\caption{Task categorization by CIR and SR changes}
\label{tab:cir_sr_categories}
\end{table}"""

    print(latex_table)
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
