"""Bridging ranker — promotes posts whose favouriters are spread evenly across
the learned user-embedding space, and dims posts favourited by a single tight
cluster or by two opposed extremes with an empty middle.

The whole condition is "pin the BridgingScorer and sort": the scorer emits
engagement-magnitude × bridging-factor per post (see scorers/bridging.py for the
online user-embedding + dispersion machinery), SingleScoreRanker sorts on it, and
the Ranker base supplies trim + per-viewer seen-demotion like every condition.

Capture-resistance motivation: a feed that rewards raw engagement rewards
whichever cluster is largest — its posts collect the most favourites. Bridging
keeps engagement as the magnitude but multiplies it down for posts approved
within a single region of the social space, so consensus-across-the-space posts
rise and one-cluster (divisive) posts are de-amplified — DSA/EMFA framing.

No camps, no precompute, no external artifact: the user embedding is learned
online from the favourite graph flowing through /rank and evolves over the run.
See the scorer's docstring for the warm-up / clean-degradation behaviour.
"""

from __future__ import annotations

from ..scorers.bridging import BridgingScorer
from .core import SingleScoreRanker


class EngagementBridgingRanker(SingleScoreRanker):
    name = "engagement_bridging"
    scorer_class = BridgingScorer
