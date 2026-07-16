"""Engagement scorer — weighted engagement merit across interaction types.

Emits, per candidate,

    1 + replies_count + quotes_count + 2·favourites_count + 3·reblogs_count

The +1 is the implicit-author-vote baseline (a post with no engagement scores 1,
not 0 — as LikeScorer's favourites+1 did), and the weights order the interaction
types by endorsement strength: a boost (wholesale re-share) counts most, a
favourite next, a reply or quote least — a reply is engagement but not
necessarily approval. All four are DIRECT per-candidate counts straight off the
Mastodon StatusSerializer (favourites_count, reblogs_count, replies_count,
quotes_count), so the scorer stays pure and lock-free.

`replies_count` is DIRECT replies only (a post's immediate children), NOT the
full descendant thread: the ranker's candidate pool is the public timeline, which
carries no replies at all, so a subtree count is not reconstructible from the
/rank payload without a Ruby-side traversal — a deliberate design compromise.

This is the shared engagement-magnitude base for the engagement family:
EngagementRanker pins it directly, and BridgingScorer / EchoChamberScorer /
DownrankToxicityScorer / DownrankOutrageScorer all use it as the magnitude they
modulate. (LikeScorer — favourites+1 alone — stays as a standalone building block.)
"""

from __future__ import annotations

from ..models import Candidate


class EngagementScorer:
    name = "engagement"

    def score_batch(self, candidates: list[Candidate]) -> list[float]:
        out: list[float] = []
        for c in candidates:
            replies = float(getattr(c, "replies_count", 0) or 0)
            quotes = float(getattr(c, "quotes_count", 0) or 0)
            favourites = float(getattr(c, "favourites_count", 0) or 0)
            boosts = float(getattr(c, "reblogs_count", 0) or 0)
            out.append(1.0 + replies + quotes + 2.0 * favourites + 3.0 * boosts)
        return out
