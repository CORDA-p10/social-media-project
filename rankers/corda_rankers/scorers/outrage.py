"""Outrage downrank scorer — engagement × (1 − s · c^γ).

The `downrank_outrage` condition, as a cohesion-layer *overlay* mirroring
downrank_toxicity: the engagement merit (EngagementScorer, exactly as
EngagementRanker uses) is multiplied by (1 − s·c^γ), demoting moral-outrage
content relative to an otherwise-normal engagement feed (DSA/EMFA de-amplification).

`c` is the team's reproducible outrage percentile from scorers/outrage_pipeline:
NRC anger/disgust word-rate × MFD 2.0 moral hit-rate, each percentile-ranked
against the frozen reference corpus (reference_distribution.csv, N≈7.8k), so
c = pct_nrc × pct_mfd ∈ [0, 1]. The overlay curve is `1 − s·c^γ` with s (strength)
and γ (curvature) from the env:
  * s = CORDA_OUTRAGE_STRENGTH (default 1.0): the max penalty. s=1 sends the
    extreme tail to ~0, matching downrank_toxicity's full strength (both downranks
    on equal footing); s<1 leaves a floor of (1−s).
  * γ = CORDA_OUTRAGE_GAMMA (default 2.0): a soft threshold. Because c is mid-heavy
    (median ≈ 0.19, unlike Detoxify's zero-inflated toxicity), γ>1 keeps ordinary
    posts near-untouched (median factor ≈ 0.97 at γ=2) while collapsing the outrage
    tail; γ=1 would give the median post a ~19% haircut.

License: NRC arrives via the `nrclex` runtime dependency (not vendored into our
source tree); MFD 2.0 is freely redistributable (outrage_mfd.dic sits beside this file).
The built image contains both — don't publish it.

Cost / isolation: outrage_pipeline (nltk + nrclex + scipy) is imported LAZILY in
`_engine`, so only the world whose CORDA_RANKER_NAME=downrank_outrage loads it.
The combined percentile `c` is a property of the *text*, so it is cached in Redis
by a content hash — a warm /rank costs zero pipeline calls. The overlay (s, γ) is
applied *after* the cache, so tuning s/γ needs no re-scoring.

Fail-safe: any pipeline error falls back to c = 0.0 (no penalty) for the affected
posts — and is NOT cached — so a hiccup degrades to a plain engagement feed rather
than failing the /rank request.
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import cached_property

from ..models import Candidate
from ..utils import candidate_text, get_redis
from .engagement import EngagementScorer

logger = logging.getLogger("corda_rankers.outrage")

_CACHE_PREFIX = "corda:outrage:v1"
_DEFAULT_STRENGTH = float(os.environ.get("CORDA_OUTRAGE_STRENGTH", "1.0"))
_DEFAULT_GAMMA = float(os.environ.get("CORDA_OUTRAGE_GAMMA", "2.0"))


class DownrankOutrageScorer:
    name = "downrank_outrage"

    def __init__(self, *, strength: float | None = None, gamma: float | None = None, redis_client=None):
        # Env-defaulted so a ranker can pin the bare class; all overridable in tests.
        self.strength = strength if strength is not None else _DEFAULT_STRENGTH
        self.gamma = gamma if gamma is not None else _DEFAULT_GAMMA
        self.redis = redis_client if redis_client is not None else get_redis()
        self._engagement = EngagementScorer()  # shared engagement-magnitude base

    @cached_property
    def _engine(self):
        # Lazy: pulls outrage_pipeline (nltk + nrclex + scipy) and loads the MFD
        # dictionary + frozen reference distribution. Only the one world running
        # this condition pays it. Built once, on the first uncached batch. The
        # reference is built from numpy (not the pipeline's pandas from_csv) to
        # keep pandas out of the image.
        import numpy as np

        from . import outrage_pipeline as op

        ref_csv = op.DEFAULT_REFERENCE_CSV
        d = np.genfromtxt(ref_csv, delimiter=",", names=True)
        reference = op.ReferenceDistribution(
            nrc_sorted=np.sort(d["nrc_anger_disgust"]),
            mfd_sorted=np.sort(d["mfd_hit_rate"]),
            n_reference=int(len(d)),
        )
        mfd = op.MFD2Scorer(ref_csv.parent / "outrage_mfd.dic")
        logger.info(
            "loaded outrage pipeline (reference N=%d, strength=%.2f gamma=%.2f)",
            reference.n_reference, self.strength, self.gamma,
        )
        return op, reference, mfd

    # ----- public API ----------------------------------------------------

    def score_batch(self, candidates: list[Candidate]) -> list[float]:
        if not candidates:
            return []
        outrage = self._outrage_batch(candidates)
        mag = self._engagement.score_batch(candidates)
        # Overlay: engagement magnitude (EngagementScorer) × (1 − s · c^γ).
        return [
            m * (1.0 - self.strength * (o ** self.gamma))
            for m, o in zip(mag, outrage)
        ]

    # ----- outrage percentile c (pipeline + Redis cache) -----------------

    def _combined(self, text: str) -> float:
        op, reference, mfd = self._engine
        nrc = op.nrc_anger_disgust_score(text)
        moral = mfd.score(text)
        return reference.percentile_nrc(nrc) * reference.percentile_mfd(moral)

    def _outrage_batch(self, candidates: list[Candidate]) -> list[float]:
        texts = [candidate_text(c) for c in candidates]
        keys = [self._cache_key(t) for t in texts]
        c_vals: list[float | None] = [None] * len(texts)
        self._fill_from_cache(keys, c_vals)
        n_cached = sum(1 for v in c_vals if v is not None)

        # Empty text can't be outraged — 0.0, no pipeline work.
        for i, t in enumerate(texts):
            if c_vals[i] is None and not t:
                c_vals[i] = 0.0

        todo = [i for i, v in enumerate(c_vals) if v is None]
        n_scored = n_failed = 0
        if todo:
            try:
                for i in todo:
                    v = self._combined(texts[i])
                    c_vals[i] = min(1.0, max(0.0, float(v)))
                    self._cache_put(keys[i], c_vals[i])
                n_scored = len(todo)
            except Exception as e:  # noqa: BLE001
                logger.warning("outrage scoring failed for %d texts: %s", len(todo), e)
                for i in todo:
                    c_vals[i] = 0.0  # no penalty; deliberately NOT cached
                n_failed = len(todo)

        logger.info(
            "outrage score_batch: n=%d cached=%d scored=%d failed=%d",
            len(candidates), n_cached, n_scored, n_failed,
        )
        return [0.0 if v is None else v for v in c_vals]

    # ----- caching (content-hash keyed; caches c, not the overlay) --------

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{_CACHE_PREFIX}:{digest[:32]}"

    def _fill_from_cache(self, keys: list[str], c_vals: list[float | None]) -> None:
        if self.redis is None or not keys:
            return
        try:
            cached = self.redis.mget(keys)
        except Exception as e:  # noqa: BLE001
            logger.warning("outrage cache read failed: %s", e)
            return
        for i, raw in enumerate(cached):
            if raw is not None:
                try:
                    c_vals[i] = float(raw)
                except (TypeError, ValueError):
                    pass

    def _cache_put(self, key: str, value: float) -> None:
        if self.redis is None:
            return
        try:
            self.redis.set(key, value)
        except Exception as e:  # noqa: BLE001
            logger.warning("outrage cache write failed: %s", e)
