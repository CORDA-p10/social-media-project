"""Echo-chamber condition (#3) — engagement × viewer-relative homophily.

The only *personalised* condition: it reorders per viewer, amplifying posts
favourited by the viewer's own side of the learned user-embedding and dimming
the other camp (see EchoChamberScorer). Built on ViewerRelativeSingleScoreRanker
— the viewer-aware pointwise base — so the body is just the scorer pin; trim,
per-viewer seen-demotion, and seen-recording all stay in the Ranker base.
Anonymous requests get a viewer-blind (all-neutral → engagement-ranked) feed,
consistent with the base skipping viewer-specific passes.
"""

from __future__ import annotations

from .core import ViewerRelativeSingleScoreRanker
from ..scorers.echo_chamber import EchoChamberScorer


class EngagementHomophilyRanker(ViewerRelativeSingleScoreRanker):
    name = "engagement_homophily"
    scorer_class = EchoChamberScorer
