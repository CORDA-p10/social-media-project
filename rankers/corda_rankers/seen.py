"""Per-viewer "already seen" store — the Redis id-set behind seen-demotion.

A viewer (an authenticated account) "sees" a status when the ranker serves it to
them in a /rank response. The Ranker base's seen-demotion pass (rankers/
__init__.py) reads this to sink posts a viewer has already been shown behind the
not-yet-seen ones before the feed is trimmed, and writes back what was served. A
fresh fetch then surfaces new content first — the freshness signal that replaced
the old position-based age decay.

This module is just the storage primitive (seen_ids / mark_seen / reset); the
demotion *policy* (when to sink, how it composes with other heuristics) lives in
the Ranker base.

State is per-viewer in Redis: one SET of status ids per account, keyed by viewer
id. It lives in the ranker's own Redis DB (CORDA_RANKER_REDIS_URL), alongside the
constructive cache — but under a disjoint prefix (corda:seen:v1 vs corda:cons:*),
so reset() can clear seen-state without touching the warm constructiveness cache.

Cross-run hygiene: Mastodon reuses status ids across `run.py --reset`
(TRUNCATE ... RESTART IDENTITY), so a seen-set left over from a prior run would
mark fresh statuses as seen and corrupt the next run. run.py calls the ranker's
POST /seen/reset right after a reset to clear corda:seen:v1:* (the constructive
cache survives). A TTL on each set is a secondary safety net for a run that dies
without a reset; it is deliberately generous vs a run's ~90 min wall-clock so a
rarely-sampled viewer never expires mid-run.

Reads degrade to "nothing seen" and writes are best-effort: a Redis hiccup makes
the feed fall back to plain ranked order (or re-show a post) rather than failing
the /rank request.
"""

from __future__ import annotations

import os

from .utils import get_redis

# Versioned and disjoint from corda:cons:* (the constructive cache). The
# /seen/reset endpoint (app.py) deletes corda:seen:v1:* via SeenStore.reset().
_PREFIX = "corda:seen:v1"

# Orphan cleanup only — the authoritative clear is /seen/reset on a fresh run.
_TTL_SECONDS = int(os.environ.get("CORDA_RANKER_SEEN_TTL", str(24 * 3600)))


class SeenStore:
    """Redis-backed per-viewer impression log. Self-configuring (pulls the shared
    Redis client from utils.get_redis()), so app.py builds it with no args;
    redis_client stays overridable for tests (e.g. a fakeredis instance)."""

    def __init__(self, *, redis_client=None, ttl_seconds: int = _TTL_SECONDS):
        self.redis = redis_client if redis_client is not None else get_redis()
        self.ttl = ttl_seconds

    def _key(self, viewer_id: str) -> str:
        return f"{_PREFIX}:{viewer_id}"

    def seen_ids(self, viewer_id: str, ids: list[str]) -> set[str]:
        """Subset of `ids` this viewer has already been served. Uses SMEMBERS +
        a Python intersection (vs SMISMEMBER) so it works on any redis-py; the
        per-viewer set is tiny (~tens of ids over a run). Empty set on any Redis
        failure — the caller then treats everything as unseen."""
        if self.redis is None or not ids:
            return set()
        try:
            members = self.redis.smembers(self._key(viewer_id))  # decode_responses → set[str]
        except Exception:  # noqa: BLE001 — degrade to "nothing seen"
            return set()
        return {i for i in ids if i in members}

    def mark_seen(self, viewer_id: str, ids: list[str]) -> None:
        """Record that the viewer was just served these status ids (refreshes the
        set's TTL). Best-effort: a failed write only means a post may be re-shown."""
        if self.redis is None or not ids:
            return
        try:
            key = self._key(viewer_id)
            self.redis.sadd(key, *ids)
            self.redis.expire(key, self.ttl)
        except Exception:  # noqa: BLE001
            pass

    def reset(self) -> int:
        """Delete every viewer's seen-set (corda:seen:v1:*), leaving the rest of
        the ranker DB — notably the corda:cons:* constructive cache — intact.
        Returns the number of keys removed. Backs the POST /seen/reset endpoint
        that run.py calls on every full reset."""
        if self.redis is None:
            return 0
        removed = 0
        for key in self.redis.scan_iter(match=f"{_PREFIX}:*", count=500):
            removed += self.redis.delete(key)
        return removed
