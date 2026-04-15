"""
Ablation: CIR (JSD) vs CIR (Absolute Difference)

Shows that CIR computed with Jensen-Shannon divergence is highly correlated
with CIR computed with absolute difference |p_full - p_truncated|.

Data: grpo_qwen-3b-instruct, step 156, all tasks.
"""

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

ROOT = Path("/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-3b-instruct/cot_importance")
STEP = "step_156"


def compute_cir_abs(cie: dict) -> float:
    """Compute CIR using mean absolute difference instead of JSD."""
    p_full = cie["p_full"]
    log_probs_truncated = cie["log_probs_truncated"]
    num_words = cie["num_words"]
    if num_words == 0 or not log_probs_truncated:
        return 0.0
    abs_diffs = [abs(p_full - math.exp(lp)) for lp in log_probs_truncated]
    return sum(abs_diffs) / num_words


def load_task(task_dir: Path):
    """Load step_156 responses for a task. Returns list of (cir_jsd, cir_abs) tuples."""
    # find the checkpoint dir (e.g. -1.0)
    ckpt_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
    if not ckpt_dirs:
        return []

    pairs = []
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
            cir_jsd = cie.get("final_normalized_js")
            if cir_jsd is None:
                continue
            cir_abs = compute_cir_abs(cie)
            pairs.append((cir_jsd, cir_abs))
    return pairs


def main():
    all_jsd, all_abs = [], []
    task_means = []  # (task_name, mean_jsd, mean_abs)

    task_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(task_dirs)} task directories\n")

    for task_dir in task_dirs:
        task_name = task_dir.name.replace("_cot_importance", "").replace("_cot_importance_", "")
        pairs = load_task(task_dir)
        if not pairs:
            print(f"  [SKIP] {task_name} — no step_156 data")
            continue

        jsd_vals = [p[0] for p in pairs]
        abs_vals = [p[1] for p in pairs]
        all_jsd.extend(jsd_vals)
        all_abs.extend(abs_vals)

        mean_jsd = np.mean(jsd_vals)
        mean_abs = np.mean(abs_vals)
        task_means.append((task_name, mean_jsd, mean_abs))
        print(f"  {task_name:40s}  CIR_jsd={mean_jsd:.4f}  CIR_abs={mean_abs:.4f}  (n={len(pairs)})")

    print(f"\n{'='*70}")
    print(f"Total responses: {len(all_jsd)}")

    # Response-level correlation
    r_resp, p_resp = pearsonr(all_jsd, all_abs)
    print(f"\nResponse-level Pearson r = {r_resp:.4f}, p = {p_resp:.2e}")

    # Task-level correlation
    task_jsd = [t[1] for t in task_means]
    task_abs = [t[2] for t in task_means]
    r_task, p_task = pearsonr(task_jsd, task_abs)
    print(f"Task-level    Pearson r = {r_task:.4f}, p = {p_task:.2e}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(all_jsd, all_abs, alpha=0.3, s=10)
    axes[0].set_xlabel("CIR (JSD)")
    axes[0].set_ylabel("CIR (Abs Diff)")
    axes[0].set_title(f"Response-level  r={r_resp:.3f}, p={p_resp:.2e}\n(n={len(all_jsd)})")
    m = np.polyfit(all_jsd, all_abs, 1)
    x_range = np.linspace(min(all_jsd), max(all_jsd), 100)
    axes[0].plot(x_range, np.polyval(m, x_range), "r--", linewidth=1.5)

    axes[1].scatter(task_jsd, task_abs, alpha=0.8, s=50)
    for t, jsd, ab in task_means:
        axes[1].annotate(t, (jsd, ab), fontsize=6, alpha=0.7)
    axes[1].set_xlabel("CIR (JSD)")
    axes[1].set_ylabel("CIR (Abs Diff)")
    axes[1].set_title(f"Task-level  r={r_task:.3f}, p={p_task:.2e}\n(n={len(task_means)})")
    m2 = np.polyfit(task_jsd, task_abs, 1)
    x_range2 = np.linspace(min(task_jsd), max(task_jsd), 100)
    axes[1].plot(x_range2, np.polyval(m2, x_range2), "r--", linewidth=1.5)

    plt.tight_layout()
    out_path = Path(__file__).parent / "cir_jsd_vs_abs.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nScatter plot saved to {out_path}")


if __name__ == "__main__":
    main()
