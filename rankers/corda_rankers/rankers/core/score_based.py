"""ScoreBasedRanker — base for any ranker built on exactly one Scorer.

A ranker declares the Scorer *class* as `scorer_class`; `self.scorer` is the
lazily-built, cached instance. Declaring by class keeps a condition's body a
one-liner (`scorer_class = HackerNewsScorer`) — to rank by a different scorer
you write a different ranker class, never pass one in.

The instance is built on first use (cached_property), and the registry only
ever builds the one selected ranker, so the other conditions' scorers never
run their setup. Anything heavier than construction — env reads, HTTP/Redis
clients, optional imports — lives inside the scorer (see ConstructiveLLMScorer).

This base intentionally does NOT define _order(): subclasses decide how the score
is used. SingleScoreRanker sorts by it pointwise; the set-aware rankers
(Bridging, Diversity) use it only as a relevance signal under their own MMR
selection. `self.scorer` is the shared piece, not the ordering policy.

It inherits Ranker, which bakes in the demotion + selection layer (already-seen,
future content heuristics, trim, seen-recording), so every scorer-based condition
only has to produce a full ordering via `_order`.
"""

from __future__ import annotations

from functools import cached_property

from .. import Ranker
from ...scorers import Scorer


class ScoreBasedRanker(Ranker):
    name: str
    scorer_class: type[Scorer]

    @cached_property
    def scorer(self) -> Scorer:
        return self.scorer_class()
