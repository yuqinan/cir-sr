# Outcome Rewards Do Not Guarantee Verifiable or Causally Important Reasoning
**Outcome Rewards Do Not Guarantee Verifiable or Causally Important Reasoning**
Qinan Yu, Alexa Tartaglini, Peter Hase, Carlos Guestrin, Christopher Potts

## Overview

Reinforcement Learning from Verifiable Rewards (RLVR) on chain-of-thought reasoning has become a standard part of language model post-training. A common assumption is that the reasoning chains trained through RLVR reliably represent how a model gets to its answer. We critically examine this assumption by developing two metrics:

- **Causal Importance of Reasoning (CIR)**: Measures the cumulative causal effect of reasoning tokens on the final answer by progressively truncating the chain-of-thought and measuring how predictions change.
- **Sufficiency of Reasoning (SR)**: Measures whether a verifier can arrive at an unambiguous answer based on the reasoning chain alone, without access to the original question.

### Key Findings

1. **RLVR does not reliably improve CIR or SR.** While RLVR improves task accuracy, it does not reliably translate into causally important or verifiable reasoning. Across 40 tasks, 19 show a decrease in CIR and 17 show decreased SR after training.
2. **RLVR can improve accuracy without reasoning.** Models can solve many tasks without chain-of-thought reasoning, meaning RLVR can improve accuracy without shaping the model's reasoning process.
3. **CIR and SR can be improved through simple modifications:**
   - **Supervised Fine-Tuning (SFT):** A small amount of SFT on expert reasoning traces before RLVR can remedy low CIR and SR.
   - **Auxiliary CIR/SR Rewards:** Augmenting the standard outcome-based reward with auxiliary CIR or SR reward signals during RLVR improves reasoning quality while maintaining accuracy.

## Metrics

<p align="center">
  <img src="assets/metrics_overview.png" width="700">
</p>

### CIR (Causal Importance of Reasoning)

CIR truncates the reasoning chain at each token position and measures whether the model's predicted answer changes. High CIR means the answer depends on the full reasoning chain (causally important); low CIR means the model has already decided the answer before generating reasoning.

$$\text{CIR} = \frac{1}{T} \sum_{k=1}^{T} \text{JS}(\text{Bernoulli}(p_k) \| \text{Bernoulli}(p_T))$$

### SR (Sufficiency of Reasoning)

SR evaluates whether a verifier model can determine the correct answer from the reasoning chain alone, without seeing the original question. High SR means the reasoning is self-contained and verifiable.

$$\text{SR}(q, t) = \begin{cases} 1 & \text{if } \hat{y}(q, t') = \hat{y}(t') \\ 0 & \text{otherwise} \end{cases}$$

## Setup

### Models
- **Base models:** Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B, Llama3.2-3B
- **Verifier:** gpt-4o-mini

### Datasets
- [ReasoningGym](https://github.com/open-thought/reasoning-gym) (40 selected tasks)
- [Math-Hard](https://github.com/hendrycks/math) (additional experiments)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Code and detailed instructions coming soon.

