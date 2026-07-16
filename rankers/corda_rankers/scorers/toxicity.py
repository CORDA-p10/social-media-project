"""Toxicity downrank scorer — engagement × (1 − toxicity).

The `downrank_toxicity` condition, as a cohesion-layer *overlay*: it takes the
engagement merit (EngagementScorer, exactly as EngagementRanker uses) and
multiplies it by (1 − toxicity). A toxic post keeps its place only if its
engagement is high enough to survive the penalty; a clean post is never boosted.
Flagged content is demoted *relative to* an otherwise-normal engagement feed —
in DSA/EMFA terms, de-amplification.

Toxicity is Detoxify's `toxicity` head — a calibrated probability in [0, 1] — so
no reference-corpus percentile is needed (unlike the dictionary-based outrage
pipeline). Model defaults to `unbiased` (RoBERTa-base, identity-bias-mitigated —
the best-accuracy English head; the model runs in only one world, so its ~500 MB
of weights is not a concern). Override via CORDA_TOXICITY_MODEL — e.g.
`unbiased-small` (a ~10× smaller ALBERT at near-equal accuracy) for a leaner
image, or `original`.

Cost / isolation: `detoxify` (and its torch dependency) is imported LAZILY inside
`_model`, so only the one world whose CORDA_RANKER_NAME=downrank_toxicity — i.e.
the one that instantiates this scorer — ever imports torch or loads the weights.
Importing this module (which app.py does to build the registry) does not.
Toxicity is a property of the *text*, so scores are cached in Redis keyed by a
content hash (+ model name), exactly like ConstructiveLLMScorer: a warm cache
makes a /rank over a 250-status pool cost zero model calls.

Fail-safe: any model error falls back to toxicity 0.0 (no penalty) for the
affected posts — and is NOT cached — so a hiccup degrades to a plain engagement
feed rather than failing the /rank request.
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import cached_property

from ..models import Candidate
from ..utils import candidate_text, get_redis
from .engagement import EngagementScorer

logger = logging.getLogger("corda_rankers.toxicity")

_CACHE_PREFIX = "corda:tox:v1"
_DEFAULT_MODEL = os.environ.get("CORDA_TOXICITY_MODEL", "unbiased")


class DownrankToxicityScorer:
    name = "downrank_toxicity"

    def __init__(self, *, model: str | None = None, redis_client=None):
        # Env-defaulted so a ranker can pin the bare class; both stay overridable
        # (a fake redis_client / a different model in tests).
        self.model_name = model or _DEFAULT_MODEL
        self.redis = redis_client if redis_client is not None else get_redis()
        self._engagement = EngagementScorer()  # shared engagement-magnitude base

    @cached_property
    def _model(self):
        # Lazy: detoxify pulls in torch, so only the world that actually runs
        # this condition imports it. Built once, on the first uncached batch.
        from detoxify import Detoxify

        logger.info("loading Detoxify(%r)", self.model_name)
        return Detoxify(self.model_name)

    # ----- public API ----------------------------------------------------

    def score_batch(self, candidates: list[Candidate]) -> list[float]:
        if not candidates:
            return []
        tox = self._toxicity_batch(candidates)
        mag = self._engagement.score_batch(candidates)
        # Overlay: engagement magnitude (EngagementScorer) × (1 − toxicity).
        return [m * (1.0 - t) for m, t in zip(mag, tox)]

    # ----- toxicity (Detoxify + Redis cache) -----------------------------

    def _toxicity_batch(self, candidates: list[Candidate]) -> list[float]:
        texts = [candidate_text(c) for c in candidates]
        keys = [self._cache_key(t) for t in texts]
        tox: list[float | None] = [None] * len(texts)
        self._fill_from_cache(keys, tox)
        n_cached = sum(1 for t in tox if t is not None)

        # Empty text can't be toxic — 0.0, no model call.
        for i, t in enumerate(texts):
            if tox[i] is None and not t:
                tox[i] = 0.0

        todo = [i for i, t in enumerate(tox) if t is None]
        n_scored = n_failed = 0
        if todo:
            try:
                preds = self._model.predict([texts[i] for i in todo])
                raw = preds["toxicity"]
                vals = raw if isinstance(raw, (list, tuple)) else [raw]
                for j, i in enumerate(todo):
                    v = min(1.0, max(0.0, float(vals[j])))
                    tox[i] = v
                    self._cache_put(keys[i], v)
                n_scored = len(todo)
            except Exception as e:  # noqa: BLE001
                logger.warning("toxicity scoring failed for %d texts: %s", len(todo), e)
                for i in todo:
                    tox[i] = 0.0  # no penalty; deliberately NOT cached
                n_failed = len(todo)

        logger.info(
            "toxicity score_batch: n=%d cached=%d scored=%d failed=%d",
            len(candidates), n_cached, n_scored, n_failed,
        )
        return [0.0 if t is None else t for t in tox]

    # ----- caching (content-hash keyed, mirrors ConstructiveLLMScorer) ----

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self.model_name}\n{text}".encode("utf-8")).hexdigest()
        return f"{_CACHE_PREFIX}:{digest[:32]}"

    def _fill_from_cache(self, keys: list[str], tox: list[float | None]) -> None:
        if self.redis is None or not keys:
            return
        try:
            cached = self.redis.mget(keys)
        except Exception as e:  # noqa: BLE001
            logger.warning("toxicity cache read failed: %s", e)
            return
        for i, raw in enumerate(cached):
            if raw is not None:
                try:
                    tox[i] = float(raw)
                except (TypeError, ValueError):
                    pass

    def _cache_put(self, key: str, value: float) -> None:
        if self.redis is None:
            return
        try:
            self.redis.set(key, value)
        except Exception as e:  # noqa: BLE001
            logger.warning("toxicity cache write failed: %s", e)
