"""Shared helpers used across the ranker service.

Small, dependency-light utilities that more than one scorer/ranker (or app.py)
needs — kept here rather than duplicated, or parked behind a specific module:

- get_redis(): the one lazy Redis connection (the constructive score cache and
  app.py's /health probe both use it).
- created_at_ts(): Mastodon ISO-8601 `created_at` → POSIX timestamp (the
  recency scorer and the HackerNews scorer both parse it).
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from html import unescape

import redis

from .models import Candidate

# Ranker uses Redis DB 1 by default (Mastodon → DB 0, no key collisions);
# entrypoint.sh derives CORDA_RANKER_REDIS_URL from REDIS_URL when unset.
_REDIS_URL = (
    os.environ.get("CORDA_RANKER_REDIS_URL")
    or os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/1")
)

_client: "redis.Redis | None" = None


def get_redis() -> "redis.Redis":
    """Lazy module-level Redis singleton. decode_responses=True so cached
    scores come back as str (parsed to float by the constructive scorer)."""
    global _client
    if _client is None:
        _client = redis.from_url(_REDIS_URL, decode_responses=True)
    return _client


def created_at_ts(c: Candidate) -> float:
    """Mastodon's ISO-8601 `created_at` → POSIX timestamp. The trailing 'Z'
    is swapped for '+00:00' so datetime.fromisoformat accepts it."""
    return datetime.fromisoformat(c.created_at.replace("Z", "+00:00")).timestamp()


# ── status text extraction (shared by the constructiveness scorers) ──────────
_BLOCK_TAGS_RE = re.compile(r"</?(?:p|br|div|li|ul|ol)[^>]*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def plain_text(html: str) -> str:
    """Mastodon status content is HTML; reduce it to plain text. Block tags
    become spaces so words don't merge across them. Capped so a long quote-post
    can't blow up a prompt / feature vector."""
    text = _BLOCK_TAGS_RE.sub(" ", html)
    text = _TAG_RE.sub("", text)
    return _WS_RE.sub(" ", unescape(text)).strip()[:2000]


def candidate_text(c: Candidate) -> str:
    """User-visible text for a status, plain-texted. Mastodon's StatusSerializer
    returns empty top-level `content` for reblogs (boosts) — the real content is
    in `reblog.content` — so fall back to that, else every boost reads as empty."""
    html = getattr(c, "content", "") or ""
    if not html:
        reblog = getattr(c, "reblog", None)
        if isinstance(reblog, dict):
            html = reblog.get("content") or ""
    return plain_text(html)
