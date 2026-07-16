"""Chronological ranker — the experiment's control condition.

Pure reverse-chronological order: newest status first, with no engagement,
relevance or constructiveness signal. Every other ranker is measured against
this baseline. It is also Mastodon's stock public-timeline behaviour, and what
the Rails shim falls back to when the ranker service is unreachable (see
mastodon/config/initializers/corda_ranker.rb) — so making it an explicit named
condition keeps the control on the same code path as the treatments.

A SingleScoreRanker pinned to the `recency` scorer (scorers/recency.py): the
scorer returns each status's POSIX timestamp, the base sort orders descending
(newest first), and sorted()'s stability keeps same-timestamp statuses in
their incoming order — deterministic for replay. Keeping recency as a separate
Scorer leaves the signal reusable by other rankers.

Like every condition, demotion is baked into its Ranker base, so an authenticated
viewer sees newest-*unseen* first, with posts they have already been shown sunk
to the back. That layer is shared across all rankers, so the only cross-world
difference is the scorer. (For an anonymous public read-only viewer there is no
demotion, so it is exactly stock reverse-chronological — matching the Rails
shim's fallback when the ranker is down.)
"""

from __future__ import annotations

from ..scorers.recency import RecencyScorer
from .core import SingleScoreRanker


class ChronologicalRanker(SingleScoreRanker):
    name = "chronological"
    scorer_class = RecencyScorer
