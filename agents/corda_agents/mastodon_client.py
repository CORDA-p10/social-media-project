"""Thin Mastodon REST client — one per agent (each carries its own OAuth token)."""

from __future__ import annotations

from typing import Any

import httpx

from .agent_actions import Action, Boost, Favourite, Post, Quote, Reply


class MastodonClient:
    """One per agent, but all agents in a run SHARE a single httpx.Client
    (connection pool) passed as *http*. Otherwise ~1000 agents each hold their
    own pooled connection and exhaust the process's file-descriptor limit
    mid-run (httpx then raises ConnectError "Too many open files"). The
    per-agent OAuth token rides as a per-request header rather than being baked
    into a per-agent client. If *http* is None (e.g. one-off use in reset.py)
    the instance owns a private client and closes it in close()."""

    def __init__(self, base_url: str, access_token: str, *,
                 http: httpx.Client | None = None, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {access_token}"}
        self._owns_http = http is None
        self._http = http if http is not None else httpx.Client(timeout=timeout)

    def public_timeline(self, limit: int = 20) -> list[dict[str, Any]]:
        r = self._http.get(f"{self.base_url}/api/v1/timelines/public",
                           params={"limit": limit}, headers=self._headers)
        r.raise_for_status()
        return r.json()

    def execute(self, action: Action) -> dict[str, Any]:
        base = self.base_url
        h = self._headers
        # No status here sets `visibility`: posts, replies, and quotes all inherit
        # the account's default privacy (public) — which is how replies/quotes
        # always posted. Post's visibility field was removed too, so an agent has
        # no say in it either.
        if isinstance(action, Post):
            r = self._http.post(f"{base}/api/v1/statuses", headers=h,
                                json={"status": action.content})
        elif isinstance(action, Reply):
            r = self._http.post(f"{base}/api/v1/statuses", headers=h,
                                json={"status": action.content, "in_reply_to_id": action.in_reply_to_id})
        elif isinstance(action, Favourite):
            r = self._http.post(f"{base}/api/v1/statuses/{action.status_id}/favourite", headers=h)
        elif isinstance(action, Boost):
            r = self._http.post(f"{base}/api/v1/statuses/{action.status_id}/reblog", headers=h)
        elif isinstance(action, Quote):
            r = self._http.post(f"{base}/api/v1/statuses", headers=h,
                                json={"status": action.content,
                                      "quoted_status_id": action.quoted_status_id})
        else:
            raise ValueError(f"unknown action: {action!r}")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()
