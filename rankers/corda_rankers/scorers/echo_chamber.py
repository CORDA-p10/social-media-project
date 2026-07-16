"""Echo-chamber scorer — the viewer-relative mirror of bridging.

Where BridgingScorer rewards posts whose favouriter cloud is spread *evenly*
across the learned user-embedding space (a per-*post* property, identical for
every viewer), this scorer rewards posts whose favouriters sit *on the viewer's
own side* of that space — amplifying in-group resonance and dimming the other
camp. It is the only personalised condition in the lineup: the same post ranks
differently for different viewers.

It deliberately reuses bridging's machinery by subclassing it: the online
favourite-graph accumulation (`_observe`), the throttled user re-embedding
(`_refit`/`_maybe_refit`, the U×U-Gram ideal-point SVD), the `_emb` positions,
the numpy handle, the lock, and the EngagementScorer engagement base are all inherited
unchanged. Only the per-post *factor* differs — dispersion → viewer-alignment —
so the two conditions share one embedding definition and #3 is a thin overlay
on the code #4 already ships.

Factor, per post p and viewer v:

  * v's position is `_emb[v]` (v is just another user in the graph). If v isn't
    embedded yet (a brand-new / pure-lurker viewer who has favourited nothing),
    or is at the origin (a "centrist" with no side), or the request is anonymous
    (viewer_id falsy), there is no side to mirror → factor 0.5 (neutral), so the
    feed ranks ≈ by engagement. This viewer cold-start is inherent and is the one
    case bridging never has (dispersion needs no viewer).
  * affinity = mean cosine similarity between v and each embedded favouriter of p
    (the viewer excluded). Direction, not magnitude: "same camp", regardless of
    how extreme. A same-side post → ~+1; the opposite camp → ~−1; a cross-cutting
    post liked by both sides → ~0 (the +1s and −1s cancel) → neutral. So this is
    the exact inverse of bridging: bridging peaks on the cross-cutting posts this
    scorer sends to the middle.
  * confidence w = 1 − 1/k (k = embedded favouriters), same as bridging: a thinly
    -favourited post is pulled toward neutral rather than trusting one or two likes.
  * factor = logistic(gain · affinity · w) ∈ (0, 1): aligned → >0.5 (amplified),
    opposed → <0.5 (dimmed), unknown/cross-cutting → 0.5. `gain` sharpens the
    cosine contrast (env CORDA_RANKER_ECHO_GAIN).

Final merit = engagement magnitude (EngagementScorer) × factor, exactly
as bridging: alignment modulates *which* approved posts rise for you, but a post
still has to be approved to rise at all.

Cold start / degradation mirrors bridging: until the graph fills and v has
favourited enough to be placed, factors sit near 0.5 and the feed ≈ ranks by
engagement magnitude; the echo sharpens as the favourite graph fills in.
"""

from __future__ import annotations

import os

from ..models import Candidate
from .bridging import BridgingScorer, _sigmoid

# Self-configuring from env (all optional). DIMS/REFIT mirror the bridging
# embedding they share the code of; GAIN is the echo-specific cosine temperature.
_ECHO_DIMS = int(os.environ.get("CORDA_RANKER_ECHO_DIMS", "16"))
_ECHO_REFIT_EVERY = int(os.environ.get("CORDA_RANKER_ECHO_REFIT_EVERY", "1"))
_ECHO_GAIN = float(os.environ.get("CORDA_RANKER_ECHO_GAIN", "2.5"))


class EchoChamberScorer(BridgingScorer):
    name = "echo_chamber"

    def __init__(
        self,
        dims: int | None = None,
        refit_every: int | None = None,
        gain: float | None = None,
    ):
        super().__init__(
            dims=dims if dims is not None else _ECHO_DIMS,
            refit_every=refit_every if refit_every is not None else _ECHO_REFIT_EVERY,
        )
        self.gain = gain if gain is not None else _ECHO_GAIN

    # ── per-post viewer alignment ─────────────────────────────────────────────

    def _alignment(self, c: Candidate, vv, viewer_id: str) -> float | None:
        """Confidence-weighted mean cosine similarity between the viewer vector
        `vv` and post c's embedded favouriters (viewer excluded). None when there
        is nothing to compare (no embedded favouriters). Assumes `vv` is a real,
        non-origin vector — the caller handles the viewer-cold-start cases."""
        np = self._np
        vecs = [
            self._emb[a]
            for a in self._fav.get(c.id, ())
            if a in self._emb and a != viewer_id
        ]
        k = len(vecs)
        if k == 0:
            return None
        x = np.vstack(vecs)
        xn = np.linalg.norm(x, axis=1)
        ok = xn > 0.0
        if not ok.any():
            return None
        vnorm = np.linalg.norm(vv)
        cos = (x[ok] @ vv) / (xn[ok] * vnorm)      # cosine to each favouriter
        w = 1.0 - 1.0 / k                          # confidence in the cloud
        return float(cos.mean()) * w

    # ── Scorer protocol (viewer-aware) ────────────────────────────────────────

    def score_batch(
        self, candidates: list[Candidate], viewer_id: str | None = None
    ) -> list[float]:
        relevance = self._relevance.score_batch(candidates)   # pure, lock-free
        with self._lock:
            self._maybe_refit(self._observe(candidates))
            vv = self._emb.get(viewer_id) if viewer_id else None
            # No viewer / not-yet-embedded / centrist (origin) → no side to mirror.
            if vv is not None and float(self._np.linalg.norm(vv)) == 0.0:
                vv = None
            aligns = (
                [None] * len(candidates)
                if vv is None
                else [self._alignment(c, vv, viewer_id) for c in candidates]
            )

        out: list[float] = []
        for rel, a in zip(relevance, aligns):
            factor = 0.5 if a is None else _sigmoid(self.gain * a)
            out.append(rel * factor)
        return out
