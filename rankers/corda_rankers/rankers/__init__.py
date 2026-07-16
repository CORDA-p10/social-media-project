"""Ranker — the dispatchable unit, with demotion baked in.

A Ranker turns a candidate pool into a ranked, viewer-aware feed. Each module
directly under rankers/ is one experimental *condition*; the registry in app.py
maps a condition name (e.g. "engagement") to a Ranker factory.

A condition implements only `_order` — rank the FULL pool, most-first; no trim,
no personalisation. This base owns the public `rank()`: it runs `_order`, then a
pipeline of *demotion passes* over that ordering, trims to `limit`, and records
what the viewer was shown. Demotion is baked in here so it is identical across
every condition.

A demotion pass *sinks* matching posts to the back of the ordering — stable
(order within the kept and the sunk groups is preserved) — BEFORE the trim, so a
demoted post is only pushed down, and drops out of the feed only if enough
non-demoted posts exist to fill `limit`. Running on the full ordering (not a
pre-trimmed top-`limit`) lets demotion promote not-yet-demoted posts from
anywhere in the ranking into the visible feed.

Passes compose, and order = priority: a later pass sinks behind an earlier one,
so the most-demoted content ends up furthest back. Today there is one pass —
already-seen-by-this-viewer (a freshness signal; it replaced the old
position-based age decay). Future content heuristics (slurs, low verifiability,
…) plug in as additional `_demote_*` passes via the `_sink` primitive; see
`_demote`. In policy-facing terms (DSA/EMFA) this is de-amplification.

Anonymous requests (falsy viewer_id — the public read-only timeline) skip the
viewer-specific passes and record nothing, so they stay a plain ranked feed.

Rankers compose with Scorers (../scorers): pointwise rankers wrap a single Scorer
(ScoreBasedRanker → SingleScoreRanker); set-aware rankers (MMR) take a Scorer for
the relevance signal and layer their own selection on top.
"""

from __future__ import annotations

from functools import cached_property
from typing import Callable

from ..models import Candidate
from ..seen import SeenStore


class Ranker:
    # Concrete conditions set `name` (the registry key) and implement `_order`.
    name: str

    @cached_property
    def _seen(self) -> SeenStore:
        # Built once, lazily; self-configures from env (shared Redis client).
        return SeenStore()

    # ----- the contract concrete conditions fill ------------------------------

    def _order(
        self, candidates: list[Candidate], viewer_id: str | None = None
    ) -> list[Candidate]:
        """Rank the FULL candidate pool, most-first. No trim, no demotion — the
        base handles both. Each experimental condition implements this.

        `viewer_id` is passed through for *personalised* conditions (e.g. the
        echo-chamber ranker, which orders relative to the viewer's embedding
        position); viewer-blind conditions simply ignore it."""
        raise NotImplementedError

    # ----- the public ranking entrypoint --------------------------------------

    def rank(
        self,
        candidates: list[Candidate],
        limit: int,
        viewer_id: str | None = None,
    ) -> list[Candidate]:
        """Return candidates as a viewer-aware feed: order → demote → trim →
        record. `viewer_id` is the authenticated requester's account id, or None
        when anonymous (then the viewer-specific demotion passes are skipped)."""
        if not candidates:
            return []
        ordered = self._demote(self._order(candidates, viewer_id), viewer_id)
        served = ordered[:limit]
        if viewer_id:
            # Record what the viewer is actually shown (post-trim), so the next
            # fetch treats exactly these as already-seen.
            self._seen.mark_seen(viewer_id, [c.id for c in served])
        return served

    # ----- demotion pipeline --------------------------------------------------

    def _demote(
        self, ordered: list[Candidate], viewer_id: str | None
    ) -> list[Candidate]:
        """Run each demotion pass in turn. Order is priority — a later pass sinks
        behind an earlier one. Add future content heuristics here, e.g.:

            ordered = self._sink(ordered, contains_slur)
            ordered = self._sink(ordered, lambda c: low_verifiability(c))
        """
        ordered = self._demote_seen(ordered, viewer_id)
        return ordered

    def _demote_seen(
        self, ordered: list[Candidate], viewer_id: str | None
    ) -> list[Candidate]:
        """Sink posts this viewer has already been served. No-op for anonymous
        viewers or when nothing in the ordering has been seen."""
        if not viewer_id:
            return ordered
        seen = self._seen.seen_ids(viewer_id, [c.id for c in ordered])
        if not seen:
            return ordered
        return self._sink(ordered, lambda c: c.id in seen)

    @staticmethod
    def _sink(
        ordered: list[Candidate], predicate: Callable[[Candidate], bool]
    ) -> list[Candidate]:
        """Stable partition: candidates matching `predicate` move to the back,
        relative order within the kept and the sunk groups preserved. The single
        primitive every demotion pass is built from."""
        keep = [c for c in ordered if not predicate(c)]
        sink = [c for c in ordered if predicate(c)]
        return keep + sink
