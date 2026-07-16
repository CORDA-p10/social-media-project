"""Action schema — what an agent can do per tick. A plain Pydantic union
(resolved by the per-variant `type` literal); the LLM emits a list of actions
as structured output and the harness validates and dispatches each via
MastodonClient. Per-tick caps: at least one action; at most one of reply/boost;
at most one of post/quote; favourites are unbounded."""

from __future__ import annotations

import json
from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# Loose safety cap on the list length to prevent a runaway response (e.g.
# a model that decides to favourite 200 things). Well above any plausible
# tick given the feed window. The meaningful constraint is the per-tick
# caps below, not this number.
MAX_ACTIONS_PER_TICK = 20


class Post(BaseModel):
    type: Literal["post"] = "post"
    content: str = Field(..., min_length=1, max_length=500)


class Reply(BaseModel):
    type: Literal["reply"] = "reply"
    in_reply_to_id: str
    content: str = Field(..., min_length=1, max_length=500)


class Favourite(BaseModel):
    type: Literal["favourite"] = "favourite"
    status_id: str


class Boost(BaseModel):
    type: Literal["boost"] = "boost"
    status_id: str


class Quote(BaseModel):
    type: Literal["quote"] = "quote"
    quoted_status_id: str
    content: str = Field(..., min_length=1, max_length=500)


Action = Union[Post, Reply, Favourite, Boost, Quote]


def _find_actions(node, depth: int = 0):
    """Dig the actions list out of a wrapped structured-output payload.

    OpenAI/Gemini return {"actions": [...]} directly. Claude via litellm
    inconsistently wraps the tool result — observed {"params": {"actions": …}},
    {"value": {"actions": …}}, and {"actions": {"actions": …}} across calls — so
    rather than chase a fixed key, search the tree for the first `actions` value
    that is a list (json-decoding strings on the way). Returns the list, or None.
    """
    if depth > 8:
        return None
    if isinstance(node, str):
        try:
            node = json.loads(node)
        except Exception:
            return None
    if isinstance(node, dict):
        a = node.get("actions")
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except Exception:
                pass
        if isinstance(a, list):
            return a
        for v in node.values():
            r = _find_actions(v, depth + 1)
            if r is not None:
                return r
    elif isinstance(node, list):
        if node and all(isinstance(x, dict) and "type" in x for x in node):
            return node
        for v in node:
            r = _find_actions(v, depth + 1)
            if r is not None:
                return r
    return None


# Models sometimes emit Mastodon's *native* API verb, or the US spelling, instead
# of ours — "reblog" for a boost, "status" for a post, "favorite" for a favourite.
# Map exactly those and nothing else (kept deliberately minimal — no speculative
# synonym table). Anything else the model invents fails validation → tick dropped.
_TYPE_ALIASES = {"status": "post", "reblog": "boost", "favorite": "favourite"}


def _canon_action(item):
    """Map an action dict's `type` through the minimal alias table (status→post,
    reblog→boost) and leave everything else exactly as the model emitted it.
    Non-dicts pass through untouched."""
    if not isinstance(item, dict):
        return item
    t = item.get("type")
    if not isinstance(t, str):
        return item
    canon = _TYPE_ALIASES.get(t.strip().lower())
    return {**item, "type": canon} if canon and canon != t else item


def _normalize_actions(lst):
    """Apply _canon_action across the list, json-decoding any string-encoded
    items first (litellm/Anthropic sometimes delivers each item as a string)."""
    out = []
    for item in lst:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception:  # noqa: BLE001
                out.append(item)
                continue
        out.append(_canon_action(item))
    return out


class ActionEnvelope(BaseModel):
    actions: list[Action] = Field(..., min_length=1, max_length=MAX_ACTIONS_PER_TICK)

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, data):
        # If `actions` isn't already a top-level list (Claude/litellm wraps it
        # under a varying key or double-nests it), find it anywhere in the tree.
        # Either way, normalize each action's `type` before the union resolves
        # (native-API verbs like "reblog"/"status" → boost/post; see _canon_action).
        if isinstance(data, dict) and isinstance(data.get("actions"), list):
            return {"actions": _normalize_actions(data["actions"])}
        found = _find_actions(data)
        return {"actions": _normalize_actions(found)} if found is not None else data

    @field_validator("actions", mode="before")
    @classmethod
    def _parse_string_actions(cls, v):
        # Litellm-routed Anthropic responses can deliver `actions` as a
        # JSON-encoded string (or each list item as a string), because
        # Anthropic's structured-output schema doesn't natively support
        # discriminated unions and litellm flattens to strings. Unwrap both.
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, list):
            return [json.loads(item) if isinstance(item, str) else item for item in v]
        return v

    @model_validator(mode="after")
    def _one_respond_one_author(self):
        # At most one "respond" (reply/boost) and one "authored" (post/quote) per
        # tick. We do NOT clamp — silently keeping one and dropping the other is a
        # fabricated choice — so a tick emitting more than one is rejected here and
        # dropped whole by the run loop. (Feed-absent status_ids are pruned
        # per-action in Agent._validate; that is removing the un-actionable, not an
        # arbitrary pick.)
        n_respond = sum(1 for a in self.actions if isinstance(a, (Reply, Boost)))
        n_author = sum(1 for a in self.actions if isinstance(a, (Post, Quote)))
        if n_respond > 1:
            raise ValueError(f"more than one reply/boost in a tick ({n_respond})")
        if n_author > 1:
            raise ValueError(f"more than one post/quote in a tick ({n_author})")
        return self
