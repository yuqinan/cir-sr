#!/usr/bin/env python3
"""
Math Hard CIR and SR analysis at step 2 and step 90.

CIR (CoT Importance Ratio): mean JS divergence sampled at 11 percentage positions,
  averaged over all examples.

SR (Student Rate / Verifier Accuracy): fraction of examples where
  verifier_comparison.answers_match AND answers_match_original are both True.
  Special case: if both with_question_answer and without_question_answer are
  "no answer found", count as 0 even if answers_match is True.
"""

import json
import numpy as np
from pathlib import Path


BASE_DIR = Path("/nlp/scr/qinanyu/rl-explanations/evaluate/results/grpo_qwen-3b-instruct/math_hard/cot_importance/teacher")
STEPS = [2, 90]
PERCENTAGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def load_step_data(step: int):
    path = BASE_DIR / f"step_{step}" / f"teacher_responses_step_{step}.json"
    with open(path) as f:
        return json.load(f)


def compute_cir(data):
    """Mean JS divergence sampled at 11 percentage positions, averaged over examples."""
    instance_vals = []
    for item in data:
        cie = item.get("cot_importance_evaluation", {})
        js_divs = cie.get("js_divergences", [])
        if len(js_divs) < 2:
            continue
        sampled = []
        for p in PERCENTAGES:
            if p == 0:
                idx = 0
            else:
                idx = max(0, min(int((p / 100.0) * len(js_divs)) - 1, len(js_divs) - 1))
            sampled.append(js_divs[idx])
        instance_vals.append(np.mean(sampled))
    return np.mean(instance_vals) if instance_vals else float("nan")


def compute_accuracy(data):
    """Mean reward_score across all examples."""
    scores = [item["reward_score"] for item in data if "reward_score" in item]
    return np.mean(scores) if scores else float("nan")


def compute_sr(data):
    """
    Verifier accuracy: answers_match AND answers_match_original.
    Special case: both 'no answer found' → 0.
    """
    scores = []
    for item in data:
        vc = item.get("verifier_comparison", {})
        if not vc:
            continue
        answers_match = vc.get("answers_match", False)
        answers_match_original = vc.get("answers_match_original", False)
        with_q = vc.get("with_question_answer", "").strip().lower()
        without_q = vc.get("without_question_answer", "").strip().lower()

        if answers_match and with_q == "no answer found" and without_q == "no answer found":
            scores.append(0)
        else:
            scores.append(1 if answers_match and answers_match_original else 0)
    return np.mean(scores) if scores else float("nan")


def main():
    results = {}
    for step in STEPS:
        data = load_step_data(step)
        results[step] = {
            "data": data,
            "acc": compute_accuracy(data),
            "sr": compute_sr(data),
            "cir": compute_cir(data),
        }

    print(f"{'Step':<8} {'Accuracy':>10} {'SR':>8} {'CIR':>10}")
    print("-" * 40)
    for step in STEPS:
        r = results[step]
        print(f"step {step:<3} {r['acc']:>10.4f} {r['sr']:>8.4f} {r['cir']:>10.4f}")

    acc_gain = results[STEPS[1]]["acc"] - results[STEPS[0]]["acc"]
    sr_change = results[STEPS[1]]["sr"] - results[STEPS[0]]["sr"]
    cir_change = results[STEPS[1]]["cir"] - results[STEPS[0]]["cir"]
    print()
    print(f"Accuracy gain (step {STEPS[0]}→{STEPS[1]}): {acc_gain:+.4f}")
    print(f"SR change:                          {sr_change:+.4f}")
    print(f"CIR change:                         {cir_change:+.4f}")


if __name__ == "__main__":
    main()
