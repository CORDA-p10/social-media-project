"""Engagement ranker — the overall-engagement condition.

A SingleScoreRanker whose merit is EngagementScorer: it sorts on
1 + replies + quotes + 2·favourites + 3·boosts alone. The feed still churns because the Ranker base's seen-demotion
sinks posts the viewer has already been shown — this replaced the old
position-based age decay. A different engagement signal (HN-"hot", Reddit hot, …)
is a different condition → its own class.
"""

from __future__ import annotations

from ..scorers.engagement import EngagementScorer
from .core import SingleScoreRanker


class EngagementRanker(SingleScoreRanker):
    name = "engagement"
    scorer_class = EngagementScorer
