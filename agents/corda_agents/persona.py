"""Persona — one entry of personas.yaml per agent.

The full persona record is carried verbatim in `data`; the agent's system prompt
(agent.py) renders from it. Only the fields the simulation needs *outside* the
prompt are promoted to typed attributes: `id` (Mastodon username, tokens.yaml
key, ranker account id), `display_name`, `engagement_level` (the activity-weight
tier run.py samples with), `model`, and `is_adversarial` (assigned post-sampling
by the loader, not taken from the YAML)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _handle_from_user_id(user_id: str) -> str:
    """YAML `user_id` → Mastodon handle: drop the zero-padding from the numeric
    tail ("user_0018" → "user_18"). Ids without a numeric tail pass through."""
    prefix, _, num = user_id.rpartition("_")
    return f"{prefix}_{int(num)}" if prefix and num.isdigit() else user_id


def _display_from_handle(handle: str) -> str:
    """Handle → display name: "user_18" → "User 18"."""
    return handle.replace("_", " ").strip().capitalize()


class Persona(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Stable handle: the Mastodon username, the tokens.yaml key, and the account
    # id the ranker sees. The YAML `user_id` minus its zero-padding.
    id: str
    display_name: str

    # Coarse engagement tier ("lurker" / "casual" / "regular" / "active"),
    # assigned uniformly at random by the loader (the persona pool's own
    # activity_profile is synthetic and unused). A simulation-control attribute —
    # it sets the agent's activity weight in run.py — and is NOT shown to the LLM.
    engagement_level: str

    model: str

    # Adversarial flag — set post-sampling by run.load_personas_yaml (a per-model
    # ratio, round(ratio × group)), NOT read from the YAML, so the adversarial
    # load is controlled independently of the pool's own labels. Drives the
    # adversarial paragraph in the identity prompt (agent.py).
    is_adversarial: bool = False

    # The whole persona record from personas.yaml. agent.py renders the prompt
    # from it, indexing nested keys directly (e.g. data["demographic_data"]).
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, entry: dict[str, Any], model: str, engagement_level: str) -> "Persona":
        """Construct a Persona from one personas.yaml entry. The handle is
        `user_id` minus zero-padding, the display name is inferred from the
        handle (the YAML `display_name` is unused), and the record is carried
        whole in `data`. `model` and `engagement_level` are both assigned by the
        loader (round-robin models; uniform-random engagement tier — the YAML's
        own activity_profile is synthetic and unused). `is_adversarial` stays
        False here — the loader sets it afterwards; the YAML's own is_adversarial
        is intentionally ignored."""
        handle = _handle_from_user_id(str(entry["user_id"]))
        return cls(
            id=handle,
            display_name=_display_from_handle(handle),
            engagement_level=engagement_level,
            model=model,
            data=entry,
        )
