"""Hard $ cap. Updated after every LLM call; raises BudgetExceededError
once the cap is hit so the run aborts cleanly."""

from __future__ import annotations

from dataclasses import dataclass

PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":                 (1.00,  5.00),
    "claude-sonnet-4-6":                (3.00, 15.00),
    "claude-sonnet-5":                  (2.00, 10.00),  # promotional through Aug 2026
    "gpt-5.4-nano":                     (0.20,  1.25),
    "gpt-5.6-luna":                     (1.00,  6.00),
    "gpt-5.6-terra":                    (2.50, 15.00),
    "vertex_ai/gemini-3.1-flash-lite":  (0.25,  1.50),
    "vertex_ai/gemini-3.5-flash":       (1.50,  9.00),
    "vertex_ai/gemini-3-flash-preview": (0.50,  3.00),
    "vertex_ai/gemini-3.1-pro-preview": (2.00, 12.00),
}

MODEL_CLIENT_LABELS: dict[str, str] = {
    "claude-sonnet-5":                  "Claude Sonnet 5",
    "gpt-5.6-luna":                     "GPT 5.6 Luna",
    "vertex_ai/gemini-3.5-flash":       "Gemini 3.5 Flash",
}


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class CostTracker:
    cap_usd: float
    spent_usd: float = 0.0
    n_calls: int = 0

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        in_p, out_p = PRICING.get(model, (1.0, 1.0))
        self.spent_usd += (input_tokens / 1_000_000) * in_p + (output_tokens / 1_000_000) * out_p
        self.n_calls += 1

    def check(self) -> None:
        if self.spent_usd >= self.cap_usd:
            raise BudgetExceededError(
                f"spent ${self.spent_usd:.4f} of ${self.cap_usd:.2f} cap "
                f"across {self.n_calls} calls"
            )
