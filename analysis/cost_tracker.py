"""
OpenAI cost tracking utility.
Prices sourced from https://platform.openai.com/docs/pricing
"""

# USD per 1M tokens
PRICING = {
    "gpt-4o":       {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "gpt-4.1":      {"input": 2.00,  "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40,  "output": 1.60},
    "o3":           {"input": 10.00, "output": 40.00},
    "o4-mini":      {"input": 1.10,  "output": 4.40},
}


class CostTracker:
    def __init__(self, model: str = "gpt-4o-mini"):
        if model not in PRICING:
            print(f"[warn] Unknown model '{model}', falling back to gpt-4o-mini pricing.")
            model = "gpt-4o-mini"
        self.model = model
        self.pricing = PRICING[model]
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.total_input_tokens  += prompt_tokens
        self.total_output_tokens += completion_tokens
        self.total_requests      += 1

    def input_cost(self) -> float:
        return (self.total_input_tokens / 1_000_000) * self.pricing["input"]

    def output_cost(self) -> float:
        return (self.total_output_tokens / 1_000_000) * self.pricing["output"]

    def total_cost(self) -> float:
        return self.input_cost() + self.output_cost()

    def print_summary(self, label: str = "") -> None:
        header = f"  Cost Summary — {self.model}" + (f" [{label}]" if label else "")
        sep = "─" * max(len(header), 52)
        print(f"\n{sep}")
        print(header)
        print(sep)
        print(f"  Requests      : {self.total_requests:>12,}")
        print(f"  Input tokens  : {self.total_input_tokens:>12,}   "
              f"@ ${self.pricing['input']:.2f}/1M  →  ${self.input_cost():.6f}")
        print(f"  Output tokens : {self.total_output_tokens:>12,}   "
              f"@ ${self.pricing['output']:.2f}/1M  →  ${self.output_cost():.6f}")
        print(f"  Total tokens  : {self.total_input_tokens + self.total_output_tokens:>12,}")
        print(f"  TOTAL COST    : ${self.total_cost():.6f}")
        print(sep)
