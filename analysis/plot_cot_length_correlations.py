"""
Analysis: CoT length is positively correlated with CIR and SR.

Uses grpo_qwen-3b-instruct, step 156, all tasks.
CoT length = num_words in cot_importance_evaluation (words in thinking).
CIR        = final_normalized_js
SR         = best_reward_score
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

ROOT = Path("/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-3b-instruct/cot_importance")
STEP = "step_156"


def verifier_score(item: dict) -> float | None:
    """Compute verifier accuracy from verifier_comparison, matching plot_cot_verifier_combined.py logic."""
    vc = item.get("verifier_comparison")
    if not vc:
        return None
    answers_match = vc.get("answers_match", False)
    answers_match_original = vc.get("answers_match_original", False)
    with_q = vc.get("with_question_answer", "").strip().lower()
    without_q = vc.get("without_question_answer", "").strip().lower()
    if answers_match and with_q == "no answer found" and without_q == "no answer found":
        return 0.0
    return 1.0 if (answers_match and answers_match_original) else 0.0


def load_task(task_dir: Path):
    """Returns list of (cot_len, cir, sr) per response."""
    ckpt_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
    records = []
    for ckpt_dir in ckpt_dirs:
        step_dir = ckpt_dir / "teacher" / STEP
        if not step_dir.exists():
            continue
        json_files = list(step_dir.glob("teacher_responses_*.json"))
        if not json_files:
            continue
        try:
            responses = json.load(open(json_files[0]))
        except json.JSONDecodeError:
            print(f"  [SKIP] {json_files[0]} — malformed JSON")
            continue
        for r in responses:
            cie = r.get("cot_importance_evaluation")
            if not cie:
                continue
            cir = cie.get("final_normalized_js")
            if cir is None:
                continue
            cot_len = cie.get("num_words", 0)
            sr = verifier_score(r)
            if sr is None:
                continue
            records.append((cot_len, cir, sr))
    return records


def main():
    all_len, all_cir, all_sr = [], [], []
    task_means = []  # (task_name, mean_len, mean_cir, mean_sr)

    task_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(task_dirs)} task directories\n")

    for task_dir in task_dirs:
        task_name = task_dir.name.replace("_cot_importance_", "").replace("_cot_importance", "")
        records = load_task(task_dir)
        if not records:
            print(f"  [SKIP] {task_name} — no step_156 data")
            continue

        len_vals = [r[0] for r in records]
        cir_vals = [r[1] for r in records]
        sr_vals  = [r[2] for r in records]

        all_len.extend(len_vals)
        all_cir.extend(cir_vals)
        all_sr.extend(sr_vals)

        task_means.append((task_name, np.mean(len_vals), np.mean(cir_vals), np.mean(sr_vals)))
        print(f"  {task_name:40s}  len={np.mean(len_vals):6.1f}  CIR={np.mean(cir_vals):.4f}  SR={np.mean(sr_vals):.3f}  (n={len(records)})")

    print(f"\n{'='*70}")
    print(f"Total responses: {len(all_len)}\n")

    # ── Correlations ────────────────────────────────────────────────────────
    r_len_cir, p_len_cir = pearsonr(all_len, all_cir)
    r_len_sr,  p_len_sr  = pearsonr(all_len, all_sr)

    task_len = [t[1] for t in task_means]
    task_cir = [t[2] for t in task_means]
    task_sr  = [t[3] for t in task_means]

    r_len_cir_t, p_len_cir_t = pearsonr(task_len, task_cir)
    r_len_sr_t,  p_len_sr_t  = pearsonr(task_len, task_sr)

    print(f"CoT Length vs CIR")
    print(f"  Response-level: r={r_len_cir:.4f}, p={p_len_cir:.2e}")
    print(f"  Task-level:     r={r_len_cir_t:.4f}, p={p_len_cir_t:.2e}\n")

    print(f"CoT Length vs SR")
    print(f"  Response-level: r={r_len_sr:.4f}, p={p_len_sr:.2e}")
    print(f"  Task-level:     r={r_len_sr_t:.4f}, p={p_len_sr_t:.2e}\n")

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    def scatter_with_fit(ax, x, y, xlabel, ylabel, title, annotate_names=None):
        ax.scatter(x, y, alpha=0.35 if annotate_names is None else 0.8,
                   s=10 if annotate_names is None else 50)
        r, p = pearsonr(x, y)
        m = np.polyfit(x, y, 1)
        xr = np.linspace(min(x), max(x), 100)
        ax.plot(xr, np.polyval(m, xr), "r--", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nr={r:.3f}, p={p:.2e} (n={len(x)})")
        if annotate_names:
            for name, xi, yi in zip(annotate_names, x, y):
                ax.annotate(name, (xi, yi), fontsize=5, alpha=0.7)

    # Response-level
    scatter_with_fit(axes[0, 0], all_len, all_cir,
                     "CoT Length (words)", "CIR", "Response-level: CoT Length vs CIR")
    scatter_with_fit(axes[0, 1], all_len, all_sr,
                     "CoT Length (words)", "SR", "Response-level: CoT Length vs SR")

    # Task-level
    names = [t[0] for t in task_means]
    scatter_with_fit(axes[1, 0], task_len, task_cir,
                     "CoT Length (words)", "CIR", "Task-level: CoT Length vs CIR",
                     annotate_names=names)
    scatter_with_fit(axes[1, 1], task_len, task_sr,
                     "CoT Length (words)", "SR", "Task-level: CoT Length vs SR",
                     annotate_names=names)

    plt.tight_layout()
    out_path = Path(__file__).parent / "cot_length_correlations.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
