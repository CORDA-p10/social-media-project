"""Scorer protocol — pointwise scoring building blocks.

A Scorer attaches one float to each candidate. It's reusable across
rankers: an engagement-weighted ranker, a diversification ranker, and
a bridging ranker can all consume the same Scorer for their relevance
signal.

Scorers do not sort, filter, or truncate. Those are the ranker's job.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Candidate


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score_batch(self, candidates: list[Candidate]) -> list[float]:
        """One float per candidate, in the same order. Higher = ranked higher."""
        ...
