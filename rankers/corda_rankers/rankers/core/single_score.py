"""SingleScoreRanker — the pointwise (score → sort) ordering.

The shared ordering of every pointwise condition: score each candidate with the
pinned Scorer to get a *merit*, sort most-first. Conditions built on it
(chronological, engagement, constructive_gemini, constructive_c3) differ
*only* in `scorer_class`.

This implements `_order` (the full, untrimmed ranking), not `rank`: trimming,
already-seen demotion, and any future content demotion all live in the Ranker
base (rankers/__init__.py), which every condition shares. There is no recency
term — the old position-based age decay was removed in favour of seen-demotion.
"""

from __future__ import annotations

from .score_based import ScoreBasedRanker
from ...models import Candidate


class SingleScoreRanker(ScoreBasedRanker):
    def _order(
        self, candidates: list[Candidate], viewer_id: str | None = None
    ) -> list[Candidate]:
        scores = self.scorer.score_batch(candidates)
        ordered = sorted(zip(scores, candidates), key=lambda sc: -sc[0])
        return [c for _, c in ordered]
