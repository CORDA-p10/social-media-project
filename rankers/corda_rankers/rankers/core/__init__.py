"""Reusable ranker building blocks, kept apart from the per-condition rankers
(chronological.py, engagement.py, …).

Each module directly under rankers/ is one experimental *condition* the
registry in app.py can select; the shared mechanics those conditions are built
from live here. The hierarchy is Ranker → ScoreBasedRanker → SingleScoreRanker:
Ranker (rankers/__init__.py) bakes in the demotion + selection layer (rank() =
order → demote → trim → record), ScoreBasedRanker holds one Scorer
(`scorer_class` → cached `self.scorer`), SingleScoreRanker adds the pointwise
sort as `_order`. A condition then collapses to "subclass + name a scorer class"
— no scorer is ever passed in, and demotion/trim/seen-tracking come for free.

ViewerRelativeSingleScoreRanker is the personalised twin of SingleScoreRanker:
same pointwise sort, but it threads the viewer into scoring (`score_batch(
candidates, viewer_id)`) for conditions whose merit depends on who is looking.
"""

from __future__ import annotations

from .score_based import ScoreBasedRanker
from .single_score import SingleScoreRanker
from .viewer_relative import ViewerRelativeSingleScoreRanker

__all__ = ["ScoreBasedRanker", "SingleScoreRanker", "ViewerRelativeSingleScoreRanker"]
