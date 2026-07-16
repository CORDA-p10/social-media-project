"""Run the agent simulation.

Usage:
    python -m corda_agents.run                # wipe + recreate accounts +
                                                mint tokens, then run (default)
    python -m corda_agents.run --resume       # skip reset; resume to the target tick

What --reset does, in order (implementation lives in `reset.py`):
  1. TRUNCATE every data table CASCADE (preserving Rails internals and
     seeded role definitions).
  2. Create admin (Owner role) + one account per persona in a single
     `rails runner` call. Admin password printed once — save it.
  3. Set display names: admin → ADMIN_DISPLAY_NAME; each persona → its
     `display_name` from the persona row.
  4. Fetch DiceBear "personas"-style avatar WebPs in parallel (seed-stamped,
     cached), attach all in one `rails runner` call.
  5. Set Server settings → Branding contact via Form::AdminSettings (same
     path as the admin web form).
  6. Register an OAuth app and mint a token per persona.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
from pathlib import Path

import httpx
import yaml

from .agent import Agent
from .llm import RemoteLLM
from .llm_costs import BudgetExceededError, CostTracker
from .logging_json import EventLogger
from .mastodon_client import MastodonClient
from .persona import Persona
from .logging_db import ActionStore, RunStore, ViewStore
from .reset import _log, reset
from .run_inject import SeedPostInjector


# Activity tiers and their relative pick weights for the per-tick sampler: an
# "active" agent is drawn 8× as often as a "lurker" (1:2:4:8, no within-tier
# draw). engagement_level is assigned uniformly at random by load_personas_yaml —
# the persona pool's own activity_profile is synthetic, so it is not used.
# See docs/B - Agents/User archetype matrix.md.
ENGAGEMENT_BASE_WEIGHT = {"lurker": 1, "casual": 2, "regular": 4, "active": 8}


def load_personas_yaml(
    path: Path,
    seed: int | None,
    num_agents: int | None,
    models: list[str],
    adversarial_ratio: float,
) -> list[Persona]:
    """Load the agent population from personas.yaml — a list with one entry per
    agent. When `num_agents` is set and smaller than the file, take a seeded
    sample (reproducible for a given seed); otherwise use every entry.

    `models` is assigned round-robin over the (seeded) sample order, so a
    3-model list splits the population 1:1:1 by agent count — deterministic
    for a given seed. A single-item list gives every agent the same model.
    `engagement_level` is assigned uniformly at random (seeded) — the pool's own
    activity_profile is synthetic, so it is not used.

    Adversarial status is assigned HERE, not read from the YAML: within each
    model's agents, `round(adversarial_ratio × group_size)` are marked
    adversarial (seeded ⇒ reproducible). So every model carries the same
    adversarial *fraction* regardless of the persona pool's own is_adversarial
    labels, and the count scales with num_agents."""
    entries = yaml.safe_load(path.read_text())
    if num_agents is not None and num_agents < len(entries):
        entries = random.Random(seed).sample(entries, num_agents)
    # engagement_level: uniform-random tier from a seeded, independent stream.
    eng_rng = random.Random(None if seed is None else seed + 3)
    levels = list(ENGAGEMENT_BASE_WEIGHT)
    personas = [
        Persona.from_yaml(e, model=models[i % len(models)],
                          engagement_level=eng_rng.choice(levels))
        for i, e in enumerate(entries)
    ]

    # Assign adversarials per model, ignoring any YAML is_adversarial. A separate
    # seeded stream (offset from the sampler's seed so the two don't correlate)
    # picks round(ratio × group) within each model's group.
    by_model: dict[str, list[Persona]] = {}
    for p in personas:
        by_model.setdefault(p.model, []).append(p)
    adv_rng = random.Random(None if seed is None else seed + 7)
    n_adv = 0
    for group in by_model.values():
        k = min(round(adversarial_ratio * len(group)), len(group))
        for p in adv_rng.sample(group, k):
            p.is_adversarial = True
            n_adv += 1
    _log(f"adversarials: round({adversarial_ratio} × per-model group) "
         f"= {n_adv} of {len(personas)} agents across {len(by_model)} models "
         f"(YAML is_adversarial ignored)")
    return personas


# ── simulation loop ───────────────────────────────────────────────────────

def _print_last_tick_prompt(agents: list[Agent]) -> None:
    """Print, in full, the prompt the most-recently-acting agent was given —
    its system prompt plus its user prompt (the feed slice and rolling
    history it saw) — so a finished run can be eyeballed for what an agent
    actually receives."""
    ticked_agents = [a for a in agents if a.last_prompt is not None]
    if not ticked_agents:
        return
    agent = max(ticked_agents, key=lambda a: a.last_prompt["tick"])
    p = agent.last_prompt
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"[run] last-tick prompt — tick {p['tick']}, agent {agent.persona.id}")
    print(bar)
    print("\n--- system prompt ---\n")
    print(p["system"])
    print("\n--- user prompt (feed slice + rolling history) ---\n")
    print(p["user"])
    print(bar)


def simulate(cfg: dict, personas: list[Persona], resume: bool = False) -> None:
    """The core run loop. Assumes accounts and tokens exist."""
    tokens_path = Path(cfg["tokens_file"])
    if not tokens_path.exists():
        raise SystemExit(f"{tokens_path} not found. First run? Use:  python -m corda_agents.run --reset")
    tokens = yaml.safe_load(tokens_path.read_text())
    base_url = cfg["mastodon_base_url"]

    run_seed: int | None = cfg.get("seed")
    if run_seed is not None:
        random.seed(run_seed)

    cost = CostTracker(cap_usd=cfg["budget_usd"])
    llm = RemoteLLM(base_url=os.environ["OPENAI_BASE_URL"])
    # ONE shared connection pool + ONE shared logger for ALL agents. A per-agent
    # httpx client or log-file handle each holds a file descriptor, and ~1000 of
    # them exhausts the process's fd limit mid-run — httpx then raises
    # ConnectError "Too many open files". Sharing keeps fd use flat.
    http = httpx.Client(timeout=10.0)
    logger = EventLogger(Path(cfg["log_file"]))
    views = ViewStore()
    actions = ActionStore()
    runs = RunStore()

    agents: list[Agent] = []
    for persona in personas:
        if persona.id not in tokens:
            raise KeyError(f"no token for persona {persona.id!r} in {cfg['tokens_file']}")
        agents.append(Agent(
            persona=persona,
            mastodon=MastodonClient(base_url, tokens[persona.id], http=http),
            llm=llm,
            logger=logger,
            cost=cost,
            views=views,
            actions=actions,
        ))

    n_ticks = int(cfg["ticks"])

    # Per-agent activity weight for the tick sampling below: the persona's
    # engagement tier alone (ENGAGEMENT_BASE_WEIGHT, module scope) — so an
    # "active" agent is drawn 8× as often as a "lurker". No within-tier draw:
    # every agent in a tier shares one weight.
    weights = [
        ENGAGEMENT_BASE_WEIGHT.get(a.persona.engagement_level, 1)
        for a in agents
    ]

    print(f"[run] {len(agents)} agents, {n_ticks} ticks, ${cost.cap_usd:.2f} budget, seed={run_seed}")

    # Per-run stats row (config + live progress + final tallies) in the world's
    # own DB; meta reads it for the 'running' pill (ended_at NULL = running).
    # Skipped for a 0-tick provision-only pass.
    ranker_name = os.environ.get("CORDA_RANKER_NAME", "unknown")
    tallies = {"post": 0, "reply": 0, "quote": 0, "favourite": 0, "boost": 0}
    start_tick = 0
    n_agent_posts = 0  # agent-authored TOP-LEVEL statuses so far: post / quote
    if resume:
        # Continue the existing run row from its last completed tick (per the
        # run-stats), carrying its tallies and spend forward so the budget cap
        # and cumulative counts stay honest across the interruption.
        start_tick, base_calls, base_cost, tallies = runs.resume(
            ranker_name, run_seed, len(agents), n_ticks)
        cost.spent_usd, cost.n_calls = base_cost, base_calls
        n_agent_posts = tallies.get("post", 0) + tallies.get("quote", 0)
        print(f"[run] resuming from tick {start_tick}/{n_ticks} "
              f"(prior spend ${base_cost:.4f}, {n_agent_posts} authored posts)", flush=True)
    elif n_ticks > 0:
        runs.start(ranker_name, run_seed, len(agents), n_ticks)
    ticks_done = start_tick

    # Real-world seed-post injection (config `seed_data`, null = off).
    # The injected count is topped up to floor(agent_posts / 4) + INITIAL_POSTS
    # after every tick; the base lands 20 injected posts now, before the first
    # tick, as the world's starting feed. See run_inject.py for the cadence.
    injector: SeedPostInjector | None = None
    if cfg.get("seed_data"):
        injector = SeedPostInjector(Path(cfg["seed_data"]))
        if resume:
            injector.resume_from(n_agent_posts)  # don't re-inject the pre-interruption backlog
        injector.maintain(n_agent_posts)

    # Realign the tick sampler's RNG on resume: the global RNG (seeded above) is
    # consumed exactly once per tick by random.choices below and nowhere else, so
    # replaying start_tick draws restores the state an uninterrupted run would
    # hold at start_tick — the resumed portion then samples the identical agent
    # sequence. No-op for a fresh run (start_tick == 0).
    if run_seed is not None and start_tick:
        for _ in range(start_tick):
            random.choices(agents, weights=weights, k=1)

    backoff = 0.0
    BACKOFF_CAP = 15.0

    try:
        for t in range(start_tick, n_ticks):
            # One agent acts per tick, sampled with Pareto-weighted probability
            # — so total agent-actions = n_ticks (not n_ticks × N), and a few
            # heavy-weight agents dominate. Pick is drawn from the seeded RNG
            # so replay is reproducible.
            agent = random.choices(agents, weights=weights, k=1)[0]
            # sha256-derived per-(tick, agent) seed: stable across processes
            # (built-in hash() is salted per-process and would break replay).
            tick_seed = (
                int.from_bytes(
                    hashlib.sha256(f"{run_seed}|{t}|{agent.persona.id}".encode()).digest()[:4],
                    "big",
                ) & 0x7FFFFFFF  # 31-bit mask: Vertex AI's seed is a signed INT32 (max 2**31-1); the raw uint32 is rejected ~half the time
                if run_seed is not None else None
            )
            try:
                landed = agent.tick(tick_idx=t, seed=tick_seed)
                n_agent_posts += sum(
                    1 for a in landed if a.get("type") in ("post", "quote")
                )
                for _act in landed:
                    _t = _act.get("type")
                    if _t in tallies:
                        tallies[_t] += 1
                backoff = 0.0  # success → clear any error backoff
            except BudgetExceededError as e:
                print(f"[run] {e} — aborting at tick {t}")
                return
            except Exception as e:  # noqa: BLE001
                agent.logger.write({"error": str(e), "error_class": type(e).__name__,
                                    "agent_id": agent.persona.id, "tick": t})
                backoff = min(backoff * 2 if backoff else 0.5, BACKOFF_CAP)

            if injector is not None:
                injector.maintain(n_agent_posts)

            ticks_done = t + 1
            step = max(1, n_ticks // 100)
            if (t + 1) % step == 0 or (t + 1) == n_ticks:
                pct = 100 * (t + 1) / n_ticks
                print(
                    f"[run] tick {t + 1}/{n_ticks} ({pct:.0f}%) "
                    f"spent=${cost.spent_usd:.4f} calls={cost.n_calls}",
                    flush=True,
                )
                runs.heartbeat(ticks_done, cost.n_calls, cost.spent_usd, tallies)

            if backoff:
                time.sleep(backoff)
    finally:
        http.close()
        if injector is not None:
            injector.close()  # shut down the Rails seed-post co-process
        runs.finish(ticks_done, cost.n_calls, cost.spent_usd, tallies)
        print(f"[run] spent ${cost.spent_usd:.4f} across {cost.n_calls} LLM calls")
        _print_last_tick_prompt(agents)


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent simulation.")
    parser.add_argument("config", type=Path, nargs="?", default=Path("config.yaml"))
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip reset; resume the existing run from its last completed tick "
             "(per run-stats) up to the config target. (Default: reset, then run from scratch.)",
    )
    parser.add_argument(
        "--num-agents", type=int, default=None, metavar="N",
        help=(
            "How many personas to generate. "
            "Defaults to `num_agents` in config.yaml. "
            "Generation is deterministic when a seed is set in config."
        ),
    )
    parser.add_argument(
        "--ticks", type=int, default=None, metavar="N",
        help="Total agent-actions across the run. Defaults to `ticks` in config.yaml.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.num_agents is not None:
        cfg["num_agents"] = args.num_agents
    if args.ticks is not None:
        cfg["ticks"] = args.ticks
    run_seed: int | None = cfg.get("seed")

    personas = load_personas_yaml(
        Path(cfg["personas_yaml"]),
        seed=run_seed,
        num_agents=int(cfg["num_agents"]),
        models=cfg["models"],
        adversarial_ratio=float(cfg.get("adversarial_ratio", 0.1)),
    )

    _log(f"config: {args.config}")
    _log(f"personas: {len(personas)} created")

    if not args.resume:
        _log("=== reset ===")
        reset(cfg, personas)
        reset_ranker_seen()

    _log("=== simulate ===")
    simulate(cfg, personas, resume=args.resume)


def reset_ranker_seen() -> None:
    """Clear the ranker's per-viewer seen-sets after a fresh reset. Mastodon
    reuses status ids across `--reset` (TRUNCATE ... RESTART IDENTITY), so a
    seen-set from a prior run would mark fresh statuses as already-seen and sink
    them in every feed. Best-effort: the ranker is a local sidecar, but a hiccup
    here must not abort the run (a stale seen-set only mis-orders some feeds)."""
    url = os.environ.get("CORDA_RANKER_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        r = httpx.post(f"{url}/seen/reset", timeout=10.0)
        r.raise_for_status()
        _log(f"ranker seen-sets cleared: {r.json().get('cleared')} key(s)")
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: could not clear ranker seen-sets ({type(e).__name__}: {e}); "
             "continuing — feeds may treat prior-run statuses as already-seen")


if __name__ == "__main__":
    main()
