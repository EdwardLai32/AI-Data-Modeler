"""Prove the prompt cache is actually read, not just written.

A measured run wrote 67,152 tokens to cache and read back 0: agent steps take
60-400s, so the default 5-minute ephemeral entry expired before the next agent
reused the prefix. Every call paid the write premium and collected nothing,
which is strictly worse than not caching.

This hits the real API twice with a gap longer than the old TTL and asserts the
second call reads the prefix back. Costs a few cents.
"""

from __future__ import annotations

import sys
import time

from pydantic import BaseModel, Field

from automl_architect.config import get_settings
from automl_architect.core.llm import LLMClient, PromptBlock

GAP_SECONDS = 330  # longer than the 5-minute TTL that failed


class Verdict(BaseModel):
    answer: str = Field(description="One short sentence.")


def main() -> int:
    settings = get_settings()
    print(f"cache_ttl setting: {settings.cache_ttl!r}")
    if not settings.enable_prompt_caching:
        print("prompt caching disabled; nothing to check")
        return 0

    client = LLMClient(settings=settings)
    # Must exceed the 1024-token minimum cacheable prefix, and be byte-identical
    # across both calls or the prefix match fails for an unrelated reason.
    stable = (
        "You are a data scientist reasoning about tabular datasets. "
        "Cite measured evidence for every claim. " * 120
    )
    blocks = [PromptBlock(stable, cache=True), PromptBlock("Answer briefly.", cache=False)]

    first = client.structured(
        output_model=Verdict, user="Is 0.79 a good ROC AUC?", system=blocks, effort="low"
    )
    print(
        f"call 1: write={first.usage.cache_write_tokens:,} "
        f"read={first.usage.cache_read_tokens:,} input={first.usage.input_tokens:,}"
    )

    print(f"waiting {GAP_SECONDS}s (longer than the old 5-minute TTL)...")
    time.sleep(GAP_SECONDS)

    second = client.structured(
        output_model=Verdict, user="Is 0.62 a good ROC AUC?", system=blocks, effort="low"
    )
    print(
        f"call 2: write={second.usage.cache_write_tokens:,} "
        f"read={second.usage.cache_read_tokens:,} input={second.usage.input_tokens:,}"
    )

    if second.usage.cache_read_tokens > 0:
        saved = second.usage.cache_read_tokens
        print(f"\nOK — the prefix survived a {GAP_SECONDS}s gap; {saved:,} tokens read from cache.")
        return 0

    print(
        f"\nFAILED — still no cache read after {GAP_SECONDS}s. Either the TTL is not "
        "reaching the request, or something in the prefix varies between calls."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
