"""Outrage downrank scorer — engagement × (1 − s · c^γ).

The `downrank_outrage` condition, as a cohesion-layer *overlay* mirroring
downrank_toxicity: the engagement merit (EngagementScorer, exactly as
EngagementRanker uses) is multiplied by (1 − s·c^γ), demoting moral-outrage
content relative to an otherwise-normal engagement feed (DSA/EMFA de-amplification).

`c` is the team's reproducible outrage percentile, computed by the inline
lexicon pipeline below: NRC anger/disgust word-rate × MFD 2.0 moral hit-rate,
each percentile-ranked against the frozen reference corpus
(outrage_reference.csv, N≈7.8k), so c = pct_nrc × pct_mfd ∈ [0, 1].
The overlay curve is `1 − s·c^γ` with s (strength) and γ (curvature) from the env:
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

Cost / isolation: the lexicon stack (nltk + nrclex) is imported LAZILY, so only
the world whose CORDA_RANKER_NAME=downrank_outrage loads it. The combined
percentile `c` is a property of the *text*, so it is cached in Redis by a content
hash — a warm /rank costs zero pipeline calls. The overlay (s, γ) is applied
*after* the cache, so tuning s/γ needs no re-scoring.

Fail-safe: any pipeline error falls back to c = 0.0 (no penalty) for the affected
posts — and is NOT cached — so a hiccup degrades to a plain engagement feed rather
than failing the /rank request.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from functools import cache, cached_property
from pathlib import Path

from ..models import Candidate
from ..utils import candidate_text, get_redis
from .engagement import EngagementScorer

logger = logging.getLogger("corda_rankers.outrage")

_CACHE_PREFIX = "corda:outrage:v1"
_DEFAULT_STRENGTH = float(os.environ.get("CORDA_OUTRAGE_STRENGTH", "1.0"))
_DEFAULT_GAMMA = float(os.environ.get("CORDA_OUTRAGE_GAMMA", "2.0"))


# ═════════════════════════════════════════════════════════════════════════
# Outrage lexicon pipeline (NRC anger/disgust × MFD 2.0, percentile-ranked
# against a frozen reference corpus).
#
# Component choices, briefly:
#   * NRC anger/disgust word RATE — (# anger/disgust tokens) / (# tokens) —
#     NOT NRC positive/negative, which collapses to a net-valence signal and
#     cancels on sarcasm the way VADER's compound does (verified: "Oh how
#     wonderful, another politician caught lying" nets zero on pos/neg but
#     correctly nonzero on anger/disgust). A rate, not a raw count: raw count
#     correlated 0.434 with text length in the reference corpus — longer text
#     scored "angrier" independent of emotional density.
#   * MFD 2.0 (Frimer et al., 2017) moral-content hit-rate, chosen over eMFD
#     for auditability (simple word-list hit-rate vs crowd-sourced
#     probabilities) per the team's DSA-transparency tiebreaker.
#   * Each component is percentile-ranked against a FROZEN reference
#     distribution rather than against the current batch, so the same text
#     always gets the same score regardless of what it is scored alongside —
#     reproducible and auditable, and ties at literal 0.0 on one component
#     (zero-inflation: 51.6% of reference texts score 0 on NRC, 22.3% on MFD —
#     a lexicon-coverage property of short informal text) don't collapse the
#     combined score.
#
# Reference corpus: the team's own cleaned Twitter/X corpus
# (final_cleaned_tweets.csv, N=7,842), frozen into outrage_reference.csv
# via ReferenceDistribution.build_from_texts(). Chosen over MFRC (Reddit) for
# register match — both components are rate-based specifically to control for
# length, and Reddit text is much longer/more discursive than tweets. Caveats:
# the corpus was assembled via a 16-keyword filter (including slur/hate-speech
# terms), so percentiles read as "relative to this politically/emotionally
# charged corpus", not "relative to typical online speech". Bootstrap-checked
# stability: percentiles stable by ~N=3–4k at p90–p99.9, so N=7,842 supports a
# downranking threshold anywhere at or below p99.
# ═════════════════════════════════════════════════════════════════════════

_TOKEN_RE = re.compile(r"[a-z']+")
_REFERENCE_CSV = Path(__file__).with_name("outrage_reference.csv")
_MFD_DIC = Path(__file__).with_name("outrage_mfd.dic")


@cache
def _nlp():
    """Lazy NLP stack (nltk + the NRC lexicon bundled inside nrclex) — built
    once, only in the world that actually scores. Returns (nltk, wordnet,
    lemmatizer, lexicon)."""
    try:
        import nltk
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer
        from nrclex.core import _load_bundled_lexicon
    except ImportError as exc:
        raise ImportError(
            "Install with: pip install NRCLex nltk\n"
            "Then: python -c \"import nltk; nltk.download('wordnet'); "
            "nltk.download('omw-1.4'); nltk.download('punkt_tab'); "
            "nltk.download('averaged_perceptron_tagger_eng')\""
        ) from exc
    return nltk, wordnet, WordNetLemmatizer(), _load_bundled_lexicon()


def _wordnet_pos(wordnet, tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def nrc_anger_disgust_score(text: str) -> float:
    """Rate of NRC-lexicon words tagged 'anger' or 'disgust':
    (# anger/disgust-tagged tokens) / (# tokens).

    Uses POS-aware lemmatization (nltk pos_tag + WordNetLemmatizer with the
    correct POS) instead of NRCLex's default noun-only lemmatizer, which
    silently misses inflected moral-violation verbs — "lied" and "stole" are
    NOT matched by the default, while "lying"/"stolen" are (verified
    empirically); exactly the verbs outrage detection cares about."""
    nltk, wordnet, lemmatizer, lexicon = _nlp()
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    lemmas = [lemmatizer.lemmatize(w.lower(), _wordnet_pos(wordnet, t)) for w, t in tagged]
    if not lemmas:
        return 0.0
    count = sum(
        1 for lemma in lemmas for t in lexicon.get(lemma, []) if t in ("anger", "disgust")
    )
    return count / len(lemmas)


def _load_mfd2(dic_path: str | Path) -> dict[str, list[str]]:
    """Parse the official MFD 2.0 .dic (LIWC-style) format: a category-code
    table between the first pair of '%' lines, then word/code entries. Multi-
    word phrases (e.g. "civil rights") are reconstructed by joining all
    non-numeric-code tokens."""
    nummap: dict[str, str] = {}
    mfd2: dict[str, list[str]] = {}
    wordmode = True
    with open(dic_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line[0] == "%":
                wordmode = not wordmode
                continue
            ent = line.strip().split()
            if wordmode:
                wordkey = " ".join(e for e in ent if e not in nummap)
                mfd2[wordkey] = [nummap[e] for e in ent if e in nummap]
            else:
                nummap[ent[0]] = ent[1]
    return mfd2


@dataclass
class MFD2Scorer:
    """Moral Foundations Dictionary 2.0 hit-rate scorer: (# tokens matching
    an MFD 2.0 entry) / (# tokens), including multi-word phrase matches."""

    dic_path: str | Path
    _single_word: set[str] = field(init=False, repr=False)
    _multi_word: list[tuple[str, list[str]]] = field(init=False, repr=False)

    def __post_init__(self):
        lexicon = _load_mfd2(self.dic_path)
        self._single_word = {w for w in lexicon if " " not in w}
        self._multi_word = [(w, codes) for w, codes in lexicon.items() if " " in w]

    def score(self, text: str) -> float:
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return 0.0
        n_hits = sum(1 for t in tokens if t in self._single_word)
        lower_text = text.lower()
        for phrase, _codes in self._multi_word:
            n_hits += lower_text.count(phrase)
        return n_hits / len(tokens)


@dataclass
class ReferenceDistribution:
    """A frozen NRC/MFD score distribution from a reference corpus, used to
    give new posts a stable, reproducible percentile score. New posts never
    affect each other's scores (np.searchsorted against the frozen arrays;
    the left/right-insertion average handles ties like rankdata 'average')."""

    nrc_sorted: "object"  # np.ndarray
    mfd_sorted: "object"  # np.ndarray
    n_reference: int

    @classmethod
    def build_from_texts(cls, texts: list[str], mfd_dic_path: str | Path) -> "ReferenceDistribution":
        """Score a raw text list and freeze it as a reference distribution —
        this is how outrage_reference.csv was originally built."""
        import numpy as np

        mfd_scorer = MFD2Scorer(mfd_dic_path)
        nrc = np.array([nrc_anger_disgust_score(t) for t in texts])
        mfd = np.array([mfd_scorer.score(t) for t in texts])
        return cls(nrc_sorted=np.sort(nrc), mfd_sorted=np.sort(mfd), n_reference=len(texts))

    def _percentile(self, sorted_arr, value: float) -> float:
        import numpy as np

        lo = np.searchsorted(sorted_arr, value, side="left")
        hi = np.searchsorted(sorted_arr, value, side="right")
        return ((lo + hi) / 2) / self.n_reference

    def percentile_nrc(self, value: float) -> float:
        """Percentile (0-1) of `value` within the frozen NRC distribution."""
        return self._percentile(self.nrc_sorted, value)

    def percentile_mfd(self, value: float) -> float:
        """Percentile (0-1) of `value` within the frozen MFD distribution."""
        return self._percentile(self.mfd_sorted, value)


# ═════════════════════════════════════════════════════════════════════════
# The scorer (Scorer protocol) — overlay over the pipeline above.
# ═════════════════════════════════════════════════════════════════════════

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
        # Lazy: loads the frozen reference distribution + the MFD dictionary
        # (and, transitively, the nltk/nrclex stack on first score). Only the
        # one world running this condition pays it. Built once.
        import numpy as np

        d = np.genfromtxt(_REFERENCE_CSV, delimiter=",", names=True)
        reference = ReferenceDistribution(
            nrc_sorted=np.sort(d["nrc_anger_disgust"]),
            mfd_sorted=np.sort(d["mfd_hit_rate"]),
            n_reference=int(len(d)),
        )
        mfd = MFD2Scorer(_MFD_DIC)
        logger.info(
            "loaded outrage pipeline (reference N=%d, strength=%.2f gamma=%.2f)",
            reference.n_reference, self.strength, self.gamma,
        )
        return reference, mfd

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
        reference, mfd = self._engine
        nrc = nrc_anger_disgust_score(text)
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
