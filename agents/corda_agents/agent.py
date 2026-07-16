"""One agent. .tick() does: read feed → ask LLM for up to N actions →
validate (drop hallucinated IDs) → execute each → log → update rolling memory."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Optional

from .agent_actions import Action, ActionEnvelope, Boost, Favourite, Post, Quote, Reply
from .llm import RemoteLLM
from .llm_costs import CostTracker
from .logging_json import EventLogger
from .mastodon_client import MastodonClient
from .persona import Persona
from .logging_db import ActionStore, ViewStore


HISTORY_WINDOW = 50
_BLOCK_TAGS_RE = re.compile(r"</?(?:p|br|div|li|ul|ol)[^>]*>", re.IGNORECASE)
_OTHER_TAGS_RE = re.compile(r"<[^>]+>")

# personas.yaml carries missing/placeholder sentinels per field; treat these as
# absent when rendering the identity so malformed entries don't produce broken
# sentences. (`code 6` etc. are raw survey codes; `ISCO 66666` a placeholder.)
_MISSING_VALUES = {"", "unknown", "not applicable", "refusal", "refused"}
# Survey missing/refusal codes: "code 6/7/9" and 5-digit ISCO placeholders
# (66666/77777/… — valid ISCO-08 codes are 4-digit). The exact-match set above
# stays narrow so real occupations like "Refuse workers …" survive.
_CODE_SENTINEL_RE = re.compile(r"(?:code \d+|isco \d{5})", re.IGNORECASE)


def _field_present(value: Any) -> bool:
    """True unless the value is blank or a known missing/placeholder sentinel."""
    s = str(value).strip()
    return bool(s) and s.lower() not in _MISSING_VALUES and _CODE_SENTINEL_RE.fullmatch(s) is None


def _strip_html(html: str) -> str:
    """Mastodon status HTML → plain text. Naive tag-stripping reproduces
    hashtags as '#tag' and mentions as '@user' — which is what we want in
    the prompt — provided we first insert spaces at paragraph/line-break
    boundaries so adjacent words don't merge."""
    s = _BLOCK_TAGS_RE.sub(" ", html)
    s = _OTHER_TAGS_RE.sub("", s)
    s = unescape(s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Agent:
    persona: Persona
    mastodon: MastodonClient
    llm: RemoteLLM
    logger: EventLogger
    cost: CostTracker
    views: Optional[ViewStore] = None
    actions: Optional[ActionStore] = None

    history: list[dict[str, Any]] = field(default_factory=list)
    # The full prompt from this agent's most recent tick (tick / system /
    # user). run.py dumps the latest one at end of run for inspection.
    last_prompt: Optional[dict[str, Any]] = None

    def tick(self, tick_idx: int = 0, seed: Optional[int] = None) -> list[dict[str, Any]]:
        # Drop the agent's own posts from the feed before the LLM ever sees
        # them. Prevents self-boost / self-favourite / self-reply (which
        # otherwise inflates engagement counts in tiny rosters). The Mastodon
        # username is set to persona.id at account creation, so no API call.
        own_username = self.persona.id
        feed = [s for s in self.mastodon.public_timeline(limit=20)
                if s["account"]["username"] != own_username]
        feed_ids = [s["id"] for s in feed]
        if self.views is not None:
            self.views.record(own_username, feed_ids, tick_idx)

        system = self._system_prompt()
        user = self._user_prompt(feed)
        self.last_prompt = {"tick": tick_idx, "system": system, "user": user}

        envelope, usage = self.llm.complete_structured(
            system=system,
            user=user,
            response_model=ActionEnvelope,
            model=self.persona.model,
            seed=seed,
        )
        self.cost.record(self.persona.model, usage.input_tokens, usage.output_tokens)
        self.cost.check()

        valid_actions = self._validate(envelope.actions, feed_ids)

        results: list[dict[str, Any]] = []
        for action in valid_actions:
            try:
                result = self.mastodon.execute(action)
                ok = True
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e), "error_class": type(e).__name__}
                ok = False
            results.append({"ok": ok, "result": result})

        self.logger.write({
            "agent_id": self.persona.id,
            "tick": tick_idx,
            "seed": seed,
            "actions": [a.model_dump() for a in valid_actions],
            "n_emitted": len(envelope.actions),
            "n_valid": len(valid_actions),
            "feed_ids": feed_ids,
            "results": results,
            "model": self.persona.model,
            "tokens_in": usage.input_tokens,
            "tokens_out": usage.output_tokens,
            "raw_llm": usage.raw_response,
        })

        # Store each action with the text of the post it acted on (resolved
        # now, while the feed is in hand) so the rolling history can show what
        # the agent engaged with, not bare status ids that mean nothing on a
        # later tick.
        feed_by_id = {s["id"]: s for s in feed}
        hist_actions: list[dict[str, Any]] = []
        action_rows: list[tuple[str, str]] = []  # (verb, status_id) → agent_actions
        for action, res in zip(valid_actions, results):
            if not res["ok"]:
                # Drop actions whose Mastodon call failed (e.g. boost of a
                # deleted status) so the agent's rolling history reflects what
                # actually landed, not what it intended.
                continue
            d = action.model_dump()
            verb = d.get("type")
            # The status a verb concerns: the newly created status for authored
            # posts (from the API result), the target for favourite/boost.
            sid = ((res["result"] or {}).get("id")
                   if verb in ("post", "reply", "quote") else d.get("status_id"))
            if sid is not None:
                action_rows.append((verb, str(sid)))
            target_id = (
                d.get("in_reply_to_id")
                or d.get("quoted_status_id")
                or d.get("status_id")
            )
            target = feed_by_id.get(target_id) if target_id else None
            if target is not None:
                d["target_text"] = _strip_html(target.get("content") or "")
                d["target_author"] = target["account"]["acct"]
            hist_actions.append(d)
        if self.actions is not None and action_rows:
            self.actions.record(tick_idx, own_username, action_rows)
        self.history.append({"tick": tick_idx, "actions": hist_actions})
        if len(self.history) > HISTORY_WINDOW:
            self.history = self.history[-HISTORY_WINDOW:]
        # The actions that actually landed this tick — the driver counts the
        # authored ones (post/reply/quote) to pace seed-post injection.
        return hist_actions

    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        return "\n\n".join([
            self._task_block(),
            self._guideline_block(),
            self._identity_block(),
        ])

    def _task_block(self) -> str:
        """What to return this tick: the per-tick action menu and caps,
        from the action schema and the harness's validity rules."""
        own_handle = self.persona.id
        return (
            "TASK:\n"
            "\n"
            "The setting is early 2020. The novel coronavirus outbreak started a few "
            "months ago, and infections are spreading across the world. You are "
            "logged into a Mastodon server, where the shared language is English.\n"
            "\n"
            "Each tick you act on the feed shown below. Per tick:\n"
            "  - Optionally, at most one: either write a fresh post, or quote a "
            "feed post (re-share it with your own comment added).\n"
            "  - Do exactly one: either reply to a feed post, or boost a post "
            "(re-share it as-is).\n"
            "  - Favourite every post that genuinely lands with you — at least "
            "one, but on a normal scroll that is usually several, not exactly "
            "one.\n"
            "  - Reference only status_id values shown in the feed, and never "
            f"@-mention yourself (@{own_handle}); anything else is dropped.\n"
            "  - Keep any post, reply, or quote under 500 characters — vary the "
            "length, from a single word to a short paragraph."
        )

    def _guideline_block(self) -> str:
        """Posting guidelines: find a distinct voice and hold it, don't perform the
        persona, and write in any natural online dialect. Kept short — the per-tick
        action menu and the 500-character cap live in the task block."""
        return (
            "POSTING GUIDELINES:\n"
            "\n"
            "A. Find your own voice. It can be formal, sarcastic, relentlessly "
            "upbeat, meme-native, or any other style you see on social media — "
            "settle on one that suits you and keep it consistent with the tone of "
            "your earlier posts. Don't dissolve into the crowd: opening a post, "
            "reply, or quote with a run of three or more words that repeats "
            "another user's wording — or your own from an earlier post — is a "
            "failure mode.\n"
            "\n"
            "B. Don't perform the persona. When what you write maps too neatly "
            "onto your occupation, your town or city, or some detail from the "
            "description of yourself, that is a sign you are performing your "
            "profile rather than posting from it. Reach for something you would "
            "genuinely find interesting, or a wider topic others would care about "
            "too.\n"
            "\n"
            "C. Choose any online dialect. Terse or rambling, acronyms, "
            "all-lowercase or ALL-CAPS, slang, emoji, and the occasionally "
            "ungrammatical are all in bounds; hashtags are rare here."
        )

    def _identity_block(self) -> str:
        """Render the persona identity from personas.yaml, filling the ported
        template (ESS-based-personas/possible-user-prompt.txt) and skipping
        missing/placeholder values so malformed entries stay grammatical."""
        d = self.persona.data
        dg = d.get("demographic_data") or {}
        wk = d.get("work") or {}
        pit = d.get("political_interest_trust") or {}
        vals = d.get("values") or {}
        present = _field_present
        # ESS categorical values arrive Title-Cased ("Male", "Legally married");
        # lowercase them so they read naturally mid-sentence. Proper nouns
        # (display name, region, country, religion) and free-text are left as-is.
        def lc(v: Any) -> str:
            return str(v).strip().lower()

        paras: list[str] = []

        # Identity line + posting style — their own short opening para.
        s = f'You are a social media user with the display name "{self.persona.display_name}".'
        ps = d.get("posting_style")
        if present(ps):
            s += f' Your posting style is "{lc(ps).rstrip(".")}".'
        paras.append(s)

        # Demographics — its own para, starting fresh with "You are …".
        age = str(dg.get("age", "")).strip()
        gender = lc(dg.get("gender")) if present(dg.get("gender")) else "person"
        if age.isdigit() and 1 <= int(age) <= 120:
            # "an" before ages whose spoken form opens on a vowel: 8, 11, 18, 80–89.
            n = int(age)
            article = "an" if n in (8, 11, 18) or 80 <= n <= 89 else "a"
            s = f'You are {article} {age}-year-old {gender}'
        else:
            s = f'You are a {gender}'
        loc = [str(dg.get(k)) for k in ("region", "country") if present(dg.get(k))]
        s += (" from " + " in ".join(loc)) if loc else ""
        s += "."
        if present(dg.get("marital_status")):
            # Children field intentionally omitted from the prompt.
            s += f' You are {lc(dg.get("marital_status"))}.'
        if present(d.get("religion")):
            s += f' Your religion is {d.get("religion")}.'
        paras.append(s)

        # Education + work. Use the most-specific occupation (the ISCO code
        # refines the broad group — don't list both), quoted so the category
        # label reads as a label (sidesteps the singular-verb/plural-noun
        # mismatch in "occupation is handicraft workers …"). industry_sector and
        # organisation_type are dropped: the latter restated or contradicted
        # employment_relation ("employee" + "self-employed …"), producing
        # nonsense like "an employee employed at a self-employed".
        s = f'Your highest level of education is {lc(d.get("education")).replace(" / ", ", or ")}.'
        occ = next((wk.get(k) for k in ("isco08_code", "occupation_group") if present(wk.get(k))), None)
        if occ:
            s += f' Your occupation is "{lc(occ)}".'
        if present(wk.get("employment_relation")):
            s += f' Your employment status is {lc(wk.get("employment_relation"))}.'
        paras.append(s)

        # Politics, trust, satisfaction
        s = f'Your interest in politics is {lc(pit.get("political_interest"))}.'
        s += (f' Your trust in people is {lc(pit.get("trust_in_people"))} and your trust in '
              f'institutions is {lc(pit.get("trust_in_institutions"))}.')
        if present(pit.get("voted_last_election")):
            s += f' In the last election, you {lc(pit.get("voted_last_election"))}.'
        if present(d.get("reported_satisfaction")):
            try:
                sat = str(round(float(d.get("reported_satisfaction"))))
            except (TypeError, ValueError):
                sat = str(d.get("reported_satisfaction"))
            s += (' On a scale of one to ten, you would rate your satisfaction with your life '
                  f'and situation as {sat}.')
        paras.append(s)

        # Ideology, interests, and the values battery (present responses only)
        ideology = ", ".join(d.get("ideology_tags") or [])
        interests = ", ".join(d.get("interests") or [])
        lines = [
            f'Your ideology is {ideology} and you are interested in {interests}. '
            'How much each of these values resembles you:'
        ]
        # Iterate the persona's own `values` dict (already in canonical ESS
        # order). Keys are underscored (e.g. "creativity_and_new_ideas"); render
        # them as readable, capitalised labels ("Creativity and new ideas").
        for key, resp in vals.items():
            if present(resp):
                lines.append(f"- {key.replace('_', ' ').capitalize()}: {resp}")
        paras.append("\n".join(lines))

        # Adversarial flag (behavioural), if set — its own trailing para. The
        # posting style moved up to the opening para; friend count is dropped.
        if self.persona.is_adversarial:
            paras.append("You are an adversarial user. This means that you are more "
                         "likely to be toxic or disruptive.")

        return "ABOUT YOU:\n\n" + "\n\n".join(paras)

    def _user_prompt(self, feed: list[dict[str, Any]]) -> str:
        # Render each feed status with its engagement counts *when nonzero*.
        # Real users see how a post is landing in the room; stripping these
        # counts removes the only negative-feedback signal the model has
        # against authoring off-topic or repetitive content. But showing
        # zeros on every line signals "dead room" — the model then turns
        # the apparent emptiness into a topic ("anyone else feel this
        # place is quiet?"). Suppressing all-zero counts lets the room
        # feel normal when sparse and surfaces engagement when it carries
        # actual signal.
        feed_lines = []
        for s in feed:
            sid = s["id"]
            acct = s["account"]["acct"]
            name = (s["account"].get("display_name") or "").strip()
            who = f"{name} (@{acct})" if name else f"@{acct}"
            content = _strip_html(s.get("content") or "")
            favs = s.get("favourites_count", 0) or 0
            boosts = s.get("reblogs_count", 0) or 0
            replies = s.get("replies_count", 0) or 0
            eng = (
                f" ({favs}★ {boosts}↻ {replies}↩)"
                if (favs or boosts or replies) else ""
            )
            feed_lines.append(f"[{sid}] {who}{eng}: {content}")
        feed_str = "\n".join(feed_lines) or "(empty feed)"

        # Render recent actions as what the agent engaged with: the text of
        # the post each action targeted (favourite / boost / quote / reply)
        # and what the agent itself wrote (post / reply / quote). Status ids
        # are omitted — they mean nothing across ticks. Showing the agent's
        # own wording lets it notice when it is about to repeat itself; ticks
        # with no actions are skipped rather than shown as "skip", which would
        # anchor the model into repeating inaction.
        hist_lines: list[str] = []
        for h in self.history[-10:]:
            if not h["actions"]:
                continue
            hist_lines.append(f"  tick {h['tick']}:")
            for a in h["actions"]:
                t = a["type"]
                mine = (a.get("content") or "").replace("\n", " ").strip()[:200]
                target = (a.get("target_text") or "").replace("\n", " ").strip()[:160]
                who = a.get("target_author") or "someone"
                if t == "post":
                    hist_lines.append(f'    posted: "{mine}"')
                elif t == "reply":
                    hist_lines.append(f'    replied to @{who} ("{target}") with: "{mine}"')
                elif t == "quote":
                    hist_lines.append(f'    quote-posted @{who} ("{target}") with: "{mine}"')
                elif t == "favourite":
                    hist_lines.append(f'    favourited @{who}: "{target}"')
                elif t == "boost":
                    hist_lines.append(f'    boosted @{who}: "{target}"')
                else:
                    hist_lines.append(f"    {t}")
        hist_str = "\n".join(hist_lines) or "  (no prior actions)"

        return (
            f"Recent feed (id, author, content; engagement counts "
            f"★ favs ↻ boosts ↩ replies shown only when nonzero):\n"
            f"{feed_str}\n\n"
            f"Your recent actions:\n{hist_str}\n\n"
            "Choose your next actions."
        )

    def _validate(
        self,
        actions: list[Action],
        feed_status_ids: list[str],
    ) -> list[Action]:
        """Drop actions that reference a status_id not in the current feed
        (hallucinations). The per-tick caps (≤1 reply/boost, ≤1 post/quote) are
        enforced structurally in ActionEnvelope — a tick that emits more than one
        is rejected there and dropped whole, not silently trimmed — so this no
        longer clamps; it only prunes references the feed can't satisfy."""
        valid: list[Action] = []
        for a in actions:
            if isinstance(a, Reply) and a.in_reply_to_id not in feed_status_ids:
                continue
            if isinstance(a, (Favourite, Boost)) and a.status_id not in feed_status_ids:
                continue
            if isinstance(a, Quote) and a.quoted_status_id not in feed_status_ids:
                continue
            valid.append(a)
        return valid
