"""ViewerRelativeSingleScoreRanker — the *viewer-aware* pointwise ordering.

The personalised twin of SingleScoreRanker: score each candidate with the pinned
Scorer *for this viewer* — `score_batch(candidates, viewer_id)` — then sort
most-first. Use it for conditions whose merit depends on who is looking (e.g.
echo-chamber homophily, which ranks by how aligned a post's favouriters are with
the viewer's embedding position). Conditions built on it collapse to a one-liner
— subclass + name a scorer class — exactly like SingleScoreRanker; the only
difference is that their Scorer's `score_batch` takes the viewer.

`viewer_id` arrives from the Ranker base's `rank(..., viewer_id=…)` via `_order`.
Trimming, already-seen demotion, and seen-recording all stay in the Ranker base,
so a personalised condition still implements nothing but its scorer.
"""

from __future__ import annotations

from .score_based import ScoreBasedRanker
from ...models import Candidate


class ViewerRelativeSingleScoreRanker(ScoreBasedRanker):
    def _order(
        self, candidates: list[Candidate], viewer_id: str | None = None
    ) -> list[Candidate]:
        scores = self.scorer.score_batch(candidates, viewer_id)
        ordered = sorted(zip(scores, candidates), key=lambda sc: -sc[0])
        return [c for _, c in ordered]
