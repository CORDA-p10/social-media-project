"""Wire format. Candidate stays permissive (`extra="allow"`) so the Ruby
side can keep posting Mastodon's full REST::StatusSerializer output without
us having to track every field. Add typed accessors here as scorers grow."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Candidate(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    created_at: str
    favouriter_ids: list[str] = Field(default_factory=list)
    reblogger_ids:  list[str] = Field(default_factory=list)
    replier_ids:    list[str] = Field(default_factory=list)


class RankRequest(BaseModel):
    candidates: list[Candidate]
    limit: int = 20
    viewer_id: str | None = None


class RankResponse(BaseModel):
    ordered_ids: list[str]
