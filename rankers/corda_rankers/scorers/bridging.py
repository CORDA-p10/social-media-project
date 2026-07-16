"""Bridging scorer — rewards posts whose favouriters are spread *evenly* across
the learned user-embedding space, and dims posts favourited by a single tight
cluster, or by two opposed extremes with an empty middle.

Bridging is a per-post property: how *broadly*, across the social space, a post
earned its approval. We model that space continuously rather than with discrete
camps — no fixed number of groups, no cluster boundaries, no precompute.

Fully online, learned from the favourite graph that flows through the ranker:

1. Every /rank request carries each candidate's current `favouriter_ids`
   (Mastodon queries the Favourite table per request). We accumulate
   `status_id -> {favouriter account ids}` across the run; favourites only ever
   accrue, so the latest set seen for a status is its full set so far.

2. From that bipartite (user × status) graph we embed *users* as ideological
   positions. Each user's favourite row is L2-normalised (so activity — how
   *many* posts they favourited — drops out) and the matrix mean-centred, then
   SVD'd through the U×U Gram matrix (cost set by the user count ~hundreds, not
   the post count). The top-d axes are the dominant *cleavages* in who-favourites-
   what; magnitudes are kept, so a user's *direction* is their side and its
   *length* their leaning strength — the average user sits at the origin, a
   one-bloc partisan far out.

3. A post's bridging factor is a logistic of how *evenly* its favouriter cloud
   fills the space. Spread = the standard deviation of the favouriters' distances
   to their centroid — high only when favouriters sit at a *range* of distances
   from the centre (some central, some peripheral): graded coverage from the
   middle out to both wings. It is ~0 both for a single tight blob AND for a
   two-pole cloud whose favouriters all sit at the same radius, so a post liked
   only by two opposed extremes is not mistaken for bridging (a plain mean
   distance would *maximise* on exactly that polarised configuration). The
   spread is standardised against the *run-global* mean/std of spreads — a
   cumulative, pool-independent reference (see _accumulate), so a given cloud
   gets the same factor regardless of what else is in the request — and passed
   through a logistic. The factor therefore lands in (0, 1) with NO floor: an
   average-spread post → ~0.5, a divisive post → small-but-positive (so
   popularity still differentiates it — nothing is annihilated), a bridging post
   → near 1. The standardised value is scaled by a confidence weight
   w(k) = 1 − 1/k (k = embedded favouriters): a thinly-favourited post is pulled
   toward neutral 0.5 rather than scoring a confident-looking factor off 1–2
   likes, and it contributes to the global reference only in proportion to w
   (so a degenerate single-favouriter cloud — spread trivially 0 — is ignored).
   No arbitrary floor and no min-favourites cut-off; both were magic constants.

4. Final merit = engagement magnitude × bridging factor. Magnitude is the shared
   EngagementScorer (1 + replies + quotes + 2·favs + 3·boosts): bridging
   modulates *which* approved posts rise, but
   a post still has to be approved to rise at all, and magnitude orders posts of
   equal spread. SingleScoreRanker sorts on this; the Ranker base trims/demotes.

Cold start / degradation: until favourites accumulate, the embedding is empty or
thin, spreads are tiny/uncertain and factors sit near the neutral 0.5, so the
feed ranks ≈ by engagement magnitude — then bridging sharpens as the graph fills in. This
is inherent: cross-group approval cannot be measured before groups have approved
anything. The global reference self-heals: the noisy warm-up spreads become a
vanishing fraction of the cumulative estimate as the run goes on.

The embedding is refit whenever the graph has grown (throttled by
CORDA_RANKER_BRIDGING_REFIT_EVERY), so the ranking evolves continuously over the
run. State is mutated under a lock: the ranker is a single uvicorn worker, but
FastAPI runs sync routes in a threadpool, so /rank calls can overlap.
"""

from __future__ import annotations

import math
import os
import threading
from functools import cached_property

from ..models import Candidate
from .engagement import EngagementScorer

# Self-configuring from env so app.py can register the bare class. Both optional.
_DIMS = int(os.environ.get("CORDA_RANKER_BRIDGING_DIMS", "16"))
_REFIT_EVERY = int(os.environ.get("CORDA_RANKER_BRIDGING_REFIT_EVERY", "1"))


def _sigmoid(x: float) -> float:
    """Numerically stable logistic."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class BridgingScorer:
    name = "bridging"

    def __init__(self, dims: int | None = None, refit_every: int | None = None):
        # Self-configure from env when not given; explicit values still win (tests).
        self.dims = dims if dims is not None else _DIMS
        self.refit_every = max(1, refit_every if refit_every is not None else _REFIT_EVERY)

        self._relevance = EngagementScorer()  # engagement magnitude base
        self._fav: dict[str, set[str]] = {}     # status_id -> favouriter ids (graph)
        self._emb: dict[str, "object"] = {}     # account_id -> position vector (np.ndarray)
        self._calls = 0
        self._fitted_edges = -1                 # edge count at last refit
        # Run-global confidence-weighted mean/variance of spreads (Welford) — the
        # pool-independent reference the logistic standardises against.
        self._w_total = 0.0
        self._mean = 0.0
        self._m2 = 0.0
        self._lock = threading.Lock()

    @cached_property
    def _np(self):
        # Lazy: the package imports fine without numpy; only the bridging
        # condition pulls it in (mirrors the constructive scorer's httpx import).
        import numpy
        return numpy

    # ── online favourite graph + user embedding ──────────────────────────────

    def _observe(self, candidates: list[Candidate]) -> int:
        """Fold this batch's favouriters into the accumulated graph; return the
        running edge count. Favourites are monotonic, so the latest set seen for
        a status supersedes the previous one."""
        for c in candidates:
            if c.favouriter_ids:
                self._fav[c.id] = set(c.favouriter_ids)
        return sum(len(s) for s in self._fav.values())

    def _refit(self) -> None:
        """Re-embed users from the accumulated favourite graph as ideological
        positions. Each user's favourite row is L2-normalised (so activity — how
        *many* posts they favourited — drops out), the matrix is mean-centred,
        then SVD'd via the U×U Gram matrix (cost set by the user count ~hundreds,
        not the post count). The top-d axes are the dominant *contrasts*
        (cleavages) in who-favourites-what; magnitudes are kept, so a user's
        embedding *direction* is their side and its *length* their leaning
        strength — the average ('centrist') user sits at the origin, a one-bloc
        partisan far out. A degenerate graph (<2 accounts) yields an empty
        embedding, so factors fall to neutral and the feed ranks by engagement magnitude."""
        np = self._np
        accounts = sorted({a for s in self._fav.values() for a in s})
        if len(accounts) < 2:
            self._emb = {}
            return
        a_idx = {a: i for i, a in enumerate(accounts)}
        posts = [s for s in self._fav.values() if s]
        # users × posts incidence (dense; U ~ hundreds, so the Gram matrix is cheap).
        M = np.zeros((len(accounts), len(posts)), dtype=np.float64)
        for j, s in enumerate(posts):
            for a in s:
                M[a_idx[a], j] = 1.0
        # Row-normalise: each user contributes equally regardless of how many
        # posts they favourited — removes the activity term that would otherwise
        # dominate the embedding magnitude.
        row_norms = np.linalg.norm(M, axis=1, keepdims=True)
        np.divide(M, row_norms, out=M, where=row_norms > 0)
        # Mean-centre so the axes are deviations from the average user: a balanced
        # user maps to the origin and the leading axes are cleavages, not the
        # all-positive "overall popularity" mode an uncentred SVD would lead with.
        M -= M.mean(axis=0)
        gram = M @ M.T                                  # U×U user covariance
        # Symmetric PSD ⇒ eigh (eigenvalues ascending); the top-d eigenpairs are
        # the principal user axes (PCA / ideal-point estimation of the favourites).
        vals, vecs = np.linalg.eigh(gram)
        d = min(self.dims, len(accounts))
        # Keep the magnitude — it now means leaning strength (extremity), not
        # activity; centrists sit near the origin, partisans far out.
        emb = vecs[:, -d:] * np.sqrt(np.clip(vals[-d:], 0.0, None))
        self._emb = {a: emb[i] for a, i in a_idx.items()}

    def _maybe_refit(self, n_edges: int) -> None:
        self._calls += 1
        grew = n_edges > self._fitted_edges
        if grew and (not self._emb or self._calls % self.refit_every == 0):
            self._refit()
            self._fitted_edges = n_edges

    # ── per-post spread + confidence ─────────────────────────────────────────

    def _spread(self, c: Candidate) -> tuple[float | None, float]:
        """(spread, confidence) for a post. Spread = the standard deviation of the
        favouriters' distances to their cloud centroid — high only when they sit at
        a *range* of distances from the centre (graded coverage from the middle out
        to both wings). It is ~0 for a single tight blob AND for a two-pole / shell
        cloud whose favouriters are all equidistant from the centre, so two opposed
        extremes are not mistaken for bridging (the old mean-distance maximised on
        exactly that). Confidence w = 1 − 1/k grows with favouriter count k; spread
        is None with no embedded favouriters, and is necessarily 0 for k < 3 (≤2
        points are always equidistant from their centroid), which w handles."""
        vecs = [self._emb[a] for a in self._fav.get(c.id, ()) if a in self._emb]
        k = len(vecs)
        if k == 0:
            return None, 0.0
        np = self._np
        x = np.vstack(vecs)
        radii = np.linalg.norm(x - x.mean(axis=0), axis=1)  # each favouriter's distance to the centre
        return float(radii.std()), 1.0 - 1.0 / k

    def _accumulate(self, d: float, w: float) -> None:
        """Fold one confidence-weighted dispersion into the run-global mean/variance
        (weighted Welford). Cumulative over the whole run, so the reference is
        pool-independent and self-heals as warm-up noise becomes a vanishing
        fraction; the weight keeps degenerate (k=1) estimates out."""
        self._w_total += w
        delta = d - self._mean
        self._mean += (w / self._w_total) * delta
        self._m2 += w * delta * (d - self._mean)

    def _std(self) -> float:
        return (self._m2 / self._w_total) ** 0.5 if self._w_total > 0.0 else 0.0

    # ── Scorer protocol ──────────────────────────────────────────────────────

    def score_batch(self, candidates: list[Candidate]) -> list[float]:
        relevance = self._relevance.score_batch(candidates)  # pure, lock-free
        with self._lock:
            self._maybe_refit(self._observe(candidates))
            spread = [self._spread(c) for c in candidates]
            for d, w in spread:
                if d is not None and w > 0.0:
                    self._accumulate(d, w)
            mean, std = self._mean, self._std()

        out: list[float] = []
        for rel, (d, w) in zip(relevance, spread):
            if d is None or std <= 0.0:
                factor = 0.5                                  # unknown → neutral
            else:
                factor = _sigmoid(((d - mean) / std) * w)     # confidence-weighted logistic
            out.append(rel * factor)
        return out
