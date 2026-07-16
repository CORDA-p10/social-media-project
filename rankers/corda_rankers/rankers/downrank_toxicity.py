"""downrank_toxicity condition — an engagement feed with toxic posts demoted.

A plain pointwise ranker over DownrankToxicityScorer, whose merit is already the
overlay engagement × (1 − toxicity) (see scorers/toxicity.py — Detoxify's
`toxicity` head, lazily loaded, Redis-cached). Trim and per-viewer seen-demotion
come from the Ranker base, so the body is just the scorer pin. In DSA/EMFA terms
this is de-amplification of toxic content layered over a standard engagement feed
— the cohesion-layer form.
"""

from __future__ import annotations

from ..scorers.toxicity import DownrankToxicityScorer
from .core import SingleScoreRanker


class DownrankToxicityRanker(SingleScoreRanker):
    name = "downrank_toxicity"
    scorer_class = DownrankToxicityScorer
