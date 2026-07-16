"""downrank_outrage condition — an engagement feed with high-outrage posts demoted.

A plain pointwise ranker over DownrankOutrageScorer, whose merit is already the
overlay engagement × (1 − s·c^γ) (see scorers/outrage.py — the team's NRC×MFD
percentile from outrage_pipeline, lazily loaded, Redis-cached). Trim and per-viewer
seen-demotion come from the Ranker base, so the body is just the scorer pin. The
moral-emotion twin of downrank_toxicity: de-amplification of outrage layered over a
standard engagement feed — the cohesion-layer form.
"""

from __future__ import annotations

from ..scorers.outrage import DownrankOutrageScorer
from .core import SingleScoreRanker


class DownrankOutrageRanker(SingleScoreRanker):
    name = "downrank_outrage"
    scorer_class = DownrankOutrageScorer
