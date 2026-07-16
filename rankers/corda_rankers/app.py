"""FastAPI ranker. POST /rank returns candidates in ranked order.

One protocol — Ranker — so dispatch is trivial: call rank(candidates,
limit) and return the result. The active ranker is selected at boot
via CORDA_RANKER_NAME (which ranker in the registry is live; distinct
from CORDA_RANKER_URL, which is Mastodon's pointer to this service).
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from fastapi import FastAPI

from .models import RankRequest, RankResponse
from .rankers import Ranker
from .seen import SeenStore
from .rankers.engagement_bridging import EngagementBridgingRanker
from .rankers.chronological import ChronologicalRanker
from .rankers.downrank_toxicity import DownrankToxicityRanker
from .rankers.downrank_outrage import DownrankOutrageRanker
from .rankers.engagement_homophily import EngagementHomophilyRanker
from .rankers.engagement import EngagementRanker
from .utils import get_redis

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("corda_rankers")

RANKERS: dict[str, Callable[[], Ranker]] = {
    "chronological":       ChronologicalRanker,
    "engagement":          EngagementRanker,
    "engagement_bridging": EngagementBridgingRanker,
    "engagement_homophily": EngagementHomophilyRanker,
    "downrank_toxicity":   DownrankToxicityRanker,
    "downrank_outrage":    DownrankOutrageRanker,
}

_RANKER_NAME = os.environ.get("CORDA_RANKER_NAME")
if _RANKER_NAME not in RANKERS:
    raise RuntimeError(
        f"CORDA_RANKER_NAME={_RANKER_NAME!r} is not a known ranker. "
        f"Available: {sorted(RANKERS)}"
    )
_ACTIVE: Ranker = RANKERS[_RANKER_NAME]()
logger.info("ranker selected: %s (%s)", _RANKER_NAME, type(_ACTIVE).__name__)

app = FastAPI(title="corda-rankers", version="1.0.0")


@app.get("/health")
def health() -> dict:
    try:
        get_redis().ping()
        return {"status": "ok", "ranker": _RANKER_NAME}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest) -> RankResponse:
    if not req.candidates:
        return RankResponse(ordered_ids=[])
    ordered = _ACTIVE.rank(req.candidates, req.limit, viewer_id=req.viewer_id)
    return RankResponse(ordered_ids=[c.id for c in ordered])


@app.post("/seen/reset")
def seen_reset() -> dict:
    """Clear every per-viewer seen-set (corda:seen:v1:*) for a fresh run, leaving
    the constructive cache intact. run.py calls this right after a `--reset`,
    because status ids are reused across resets and a stale seen-set would mark
    fresh statuses as already-seen. Safe to call anytime (idempotent)."""
    removed = SeenStore().reset()
    logger.info("seen/reset: cleared %d viewer seen-set(s)", removed)
    return {"status": "ok", "cleared": removed}
