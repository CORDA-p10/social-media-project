"""Recency scorer — scores a status by how recently it was posted.

Higher score = more recent. The score is the POSIX timestamp of `created_at`
(via utils.created_at_ts), so a pointwise sort over this scorer reproduces a
reverse-chronological feed (see rankers/chronological.py).

Kept as a standalone Scorer — not folded into the chronological ranker — so
other rankers can reuse a recency signal (e.g. a recency-decayed
constructiveness blend) without re-parsing timestamps. A ranker that *blends*
this with another scorer should min-max normalise it first: raw timestamps
are large numbers and would otherwise dominate any linear combination.
"""

from __future__ import annotations

from ..models import Candidate
from ..utils import created_at_ts


class RecencyScorer:
    name = "recency"

    def score_batch(self, candidates: list[Candidate]) -> list[float]:
        return [created_at_ts(c) for c in candidates]
