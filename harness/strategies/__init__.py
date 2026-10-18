"""Reference strategies. No LLM code lives here or anywhere in Stage 0.

`BuyAndHold` is the benchmark every later stage has to beat after costs. Per
RESEARCH.md section 7, the honest expected outcome of Stage 2 is that a simple
rule does *not* beat it, and finding that out cheaply is the whole point.
"""

from harness.strategies.buy_and_hold import BuyAndHold

__all__ = ["BuyAndHold"]
