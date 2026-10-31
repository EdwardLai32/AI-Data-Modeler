"""Run-scoped reservations: retries consume the same limits as first attempts."""

from __future__ import annotations

from .errors import BudgetExceeded


class CallBudget:
    def __init__(self, state):
        self.state = state
        self.reserved = 0
        self.reserved_cost = 0.0

    def reserve(self, prompt: str, output_tokens: int) -> None:
        state = self.state
        cfg, settings = state.config, state.settings
        if cfg.strict_deadline and state.time_remaining <= 0:
            raise BudgetExceeded("Run deadline reached before the next AI request.")
        if state.usage.llm_calls >= cfg.max_llm_calls:
            raise BudgetExceeded("AI request limit reached (including retries).")
        # UTF-8 bytes give a conservative token bound; the extra allowance covers
        # message framing. Unknown/failed requests retain their reservation.
        tokens = len(prompt.encode("utf-8")) + 1024 + output_tokens
        spent = state.extras.get("reserved_llm_tokens", 0)
        if spent + tokens > cfg.max_llm_tokens:
            raise BudgetExceeded("Insufficient AI token budget for this request.")
        rates = (settings.input_price_per_million, settings.output_price_per_million)
        cost = 0.0
        if cfg.max_cost_usd is not None:
            if None in rates:
                raise BudgetExceeded("A dollar limit requires input and output price estimates.")
            # Cache writes can cost more than ordinary input; reserve at 2x.
            cost = ((tokens - output_tokens) * rates[0] * 2 + output_tokens * rates[1]) / 1e6
            if state.extras.get("reserved_llm_cost", 0.0) + cost > cfg.max_cost_usd:
                raise BudgetExceeded("Insufficient estimated spending budget for this request.")
        state.usage.llm_calls += 1
        state.extras["reserved_llm_tokens"] = spent + tokens
        state.extras["reserved_llm_cost"] = state.extras.get("reserved_llm_cost", 0.0) + cost
        self.reserved, self.reserved_cost = tokens, cost

    def record(self, usage) -> None:
        state = self.state
        actual = usage.total_prompt_tokens + usage.output_tokens
        state.extras["reserved_llm_tokens"] += actual - self.reserved
        if self.reserved_cost:
            state.extras["reserved_llm_cost"] += usage.cost_usd - self.reserved_cost
        self.reserved = 0
        self.reserved_cost = 0.0
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cost_usd",
        ):
            setattr(state.usage, name, getattr(state.usage, name) + getattr(usage, name))
        state.usage.pricing_configured = (
            state.settings.input_price_per_million is not None
            and state.settings.output_price_per_million is not None
        )
