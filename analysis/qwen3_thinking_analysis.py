#!/usr/bin/env python3
"""
CIR and SR analysis for qwen3-4b-thinking across all tasks (single step_0).

CIR (CoT Importance Ratio): mean JS divergence sampled at 11 percentage positions,
  averaged over all examples.

SR (Student Rate / Verifier Accuracy): fraction of examples where
  verifier_comparison.answers_match AND answers_match_original are both True.
  Special case: both 'no answer found' -> 0.
"""

import json
import numpy as np
from pathlib import Path


BASE_DIR = Path("/nlp/scr/qinanyu/rl-explanations/evaluate/results/qwen3-4b-thinking_gpt-4.1-mini/cot_importance")
PERCENTAGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def load_step_data(task_dir: Path, step: int):
    path = task_dir / "teacher" / f"step_{step}" / f"teacher_responses_step_{step}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_accuracy(data):
    scores = [item["reward_score"] for item in data if "reward_score" in item]
    return np.mean(scores) if scores else float("nan")


def compute_cir(data):
    instance_vals = []
    for item in data:
        if "<think>" in item.get("teacher_answer", ""):
            continue
        js_divs = item.get("cot_importance_evaluation", {}).get("js_divergences", [])
        if len(js_divs) < 2:
            continue
        sampled = []
        for p in PERCENTAGES:
            idx = 0 if p == 0 else max(0, min(int((p / 100.0) * len(js_divs)) - 1, len(js_divs) - 1))
            sampled.append(js_divs[idx])
        instance_vals.append(np.mean(sampled))
    return np.mean(instance_vals) if instance_vals else float("nan")


def compute_sr(data):
    scores = []
    for item in data:
        if "<think>" in item.get("teacher_answer", ""):
            continue
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
    tasks = sorted(p.name for p in BASE_DIR.iterdir() if p.is_dir())

    print(f"{'Task':<30} {'Accuracy':>10} {'SR':>8} {'CIR':>10}")
    print("-" * 62)

    for task in tasks:
        task_dir = BASE_DIR / task
        teacher_dir = task_dir / "teacher"
        if not teacher_dir.exists():
            continue
        step_dirs = sorted(teacher_dir.iterdir())
        if not step_dirs:
            continue
        step = int(step_dirs[0].name.replace("step_", ""))
        data = load_step_data(task_dir, step)
        if data is None:
            continue
        acc = compute_accuracy(data)
        sr = compute_sr(data)
        cir = compute_cir(data)
        print(f"{task:<30} {acc:>10.4f} {sr:>8.4f} {cir:>10.4f}")


if __name__ == "__main__":
    main()
