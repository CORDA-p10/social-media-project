"""Postgres run-telemetry writers, written from the agent-run side into the
world's own Mastodon database (DATABASE_URL). Two of them:

  ViewStore   → corda_views:   one row per (viewer, status, tick) — what each
                agent's prompt showed (ground-truth exposure, not what the
                ranker served); meta /export aggregates it into view_count.
  ActionStore → agent_actions: one row per landed action (post/reply/quote/
                favourite/boost) with its tick and the status it concerns —
                the created status for authored posts, the target for
                favourite/boost; meta /export LEFT JOINs it to add a `tick`
                column to both CSVs. Injected posts (run_inject) have no row.

Both tables sit next to the statuses they reference and are TRUNCATEd by the
same reset wipe (reset.py truncates every non-Rails-internal table).

Best-effort by design: a Postgres hiccup must not kill a paid LLM run —
record() swallows errors (warning on the first and every 500th) and the sim
continues; a lost batch only thins the exposure data.
"""

from __future__ import annotations

import csv
import io
import os
import subprocess

_DDL = """
CREATE TABLE IF NOT EXISTS corda_views (
    id         bigserial   PRIMARY KEY,
    status_id  bigint      NOT NULL,
    viewer     text        NOT NULL,
    tick       integer     NOT NULL,
    viewed_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS corda_views_status_idx ON corda_views (status_id);
"""

_COPY = "COPY corda_views (status_id, viewer, tick) FROM STDIN WITH (FORMAT csv)"


class ViewStore:
    """One shared instance for all agents (constructed in simulate()). Without
    a DATABASE_URL (a run outside the world containers) it is a disabled no-op,
    announced once at startup."""

    def __init__(self, database_url: str | None = None):
        self._url = database_url or os.environ.get("DATABASE_URL")
        self._ensured = False  # table DDL ran successfully at least once
        self._failures = 0
        if not self._url:
            print("[views] DATABASE_URL not set — view recording disabled")

    def _psql(self, sql: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["psql", self._url, "-q", "-v", "ON_ERROR_STOP=1", "-c", sql],
            input=stdin, capture_output=True, timeout=15,
        )

    def record(self, viewer: str, status_ids: list[str], tick: int) -> None:
        """One row per status shown to `viewer` on `tick`. Best-effort."""
        if not self._url or not status_ids:
            return
        try:
            if not self._ensured:
                self._run_or_raise(_DDL)
                self._ensured = True
            buf = io.StringIO()
            writer = csv.writer(buf)
            for sid in status_ids:
                writer.writerow([int(sid), viewer, tick])
            self._run_or_raise(_COPY, stdin=buf.getvalue().encode())
        except Exception as e:  # noqa: BLE001
            self._failures += 1
            if self._failures == 1 or self._failures % 500 == 0:
                print(
                    f"[views] WARNING: insert failed ({type(e).__name__}: {e}) — "
                    f"{self._failures} failure(s) so far; continuing without these rows"
                )

    def _run_or_raise(self, sql: str, stdin: bytes | None = None) -> None:
        proc = self._psql(sql, stdin=stdin)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300])


_ACTIONS_DDL = """
CREATE TABLE IF NOT EXISTS agent_actions (
    id         bigserial   PRIMARY KEY,
    tick       integer     NOT NULL,
    actor      text        NOT NULL,
    verb       text        NOT NULL,
    status_id  bigint      NOT NULL,
    acted_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_actions_status_verb_idx ON agent_actions (status_id, verb);
CREATE INDEX IF NOT EXISTS agent_actions_actor_status_idx ON agent_actions (actor, status_id);
"""

_ACTIONS_COPY = "COPY agent_actions (tick, actor, verb, status_id) FROM STDIN WITH (FORMAT csv)"


class ActionStore:
    """Tick-stamped agent actions → agent_actions (see module docstring). Same
    shared-instance, psql-CLI, best-effort design as ViewStore. `record` takes
    the tick's landed actions in one COPY (one psql process per tick)."""

    def __init__(self, database_url: str | None = None):
        self._url = database_url or os.environ.get("DATABASE_URL")
        self._ensured = False
        self._failures = 0
        if not self._url:
            print("[actions] DATABASE_URL not set — action recording disabled")

    def _psql(self, sql: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["psql", self._url, "-q", "-v", "ON_ERROR_STOP=1", "-c", sql],
            input=stdin, capture_output=True, timeout=15,
        )

    def record(self, tick: int, actor: str, rows: list[tuple[str, str]]) -> None:
        """`rows` is [(verb, status_id), …] for the actions that landed this
        tick for `actor`. Best-effort."""
        if not self._url or not rows:
            return
        try:
            if not self._ensured:
                self._run_or_raise(_ACTIONS_DDL)
                self._ensured = True
            buf = io.StringIO()
            writer = csv.writer(buf)
            for verb, sid in rows:
                writer.writerow([tick, actor, verb, int(sid)])
            self._run_or_raise(_ACTIONS_COPY, stdin=buf.getvalue().encode())
        except Exception as e:  # noqa: BLE001
            self._failures += 1
            if self._failures == 1 or self._failures % 500 == 0:
                print(
                    f"[actions] WARNING: insert failed ({type(e).__name__}: {e}) — "
                    f"{self._failures} failure(s) so far; continuing without these rows"
                )

    def _run_or_raise(self, sql: str, stdin: bytes | None = None) -> None:
        proc = self._psql(sql, stdin=stdin)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300])


_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS corda_runs (
    id              bigserial     PRIMARY KEY,
    ranker_name     text          NOT NULL,
    seed            integer,
    num_agents      integer,
    ticks_target    integer,
    ticks_done      integer       NOT NULL DEFAULT 0,
    llm_calls       integer       NOT NULL DEFAULT 0,
    llm_cost_usd    numeric(12,4) NOT NULL DEFAULT 0,
    duration        numeric(12,2),
    started_at      timestamptz   NOT NULL DEFAULT now(),
    ended_at        timestamptz,
    last_tick_at    timestamptz,
    action_count    integer       NOT NULL DEFAULT 0,
    post_count      integer       NOT NULL DEFAULT 0,
    favourite_count integer       NOT NULL DEFAULT 0,
    boost_count     integer       NOT NULL DEFAULT 0,
    quote_count     integer       NOT NULL DEFAULT 0,
    reply_count     integer       NOT NULL DEFAULT 0
);
"""


class RunStore:
    """One row per sim run → corda_runs, in the world's own DB: config +
    live progress (heartbeat) + final tallies. `ended_at IS NULL` ⇒ still running,
    which the meta page reads for its 'running' pill; `duration` is seconds elapsed
    (updated each heartbeat, finalised at finish). Same shared-instance, psql-CLI,
    best-effort design as ViewStore/ActionStore, and TRUNCATEd by the same reset
    wipe — so each run starts from a clean row."""

    def __init__(self, database_url: str | None = None):
        self._url = database_url or os.environ.get("DATABASE_URL")
        self._id: int | None = None
        self._failures = 0
        if not self._url:
            print("[runs] DATABASE_URL not set — run-stats disabled")

    def _psql(self, sql: str, capture: bool = False) -> subprocess.CompletedProcess:
        flags = ["-tA", "-q"] if capture else ["-q"]
        return subprocess.run(
            ["psql", self._url, *flags, "-v", "ON_ERROR_STOP=1", "-c", sql],
            capture_output=True, timeout=15,
        )

    def start(self, ranker_name: str, seed, num_agents: int, ticks_target: int) -> None:
        """Ensure the table exists and INSERT the run row; remember its id."""
        if not self._url:
            return
        try:
            self._run_or_raise(_RUNS_DDL)
            rn = str(ranker_name).replace("'", "''")
            s = "NULL" if seed is None else int(seed)
            proc = self._psql(
                f"INSERT INTO corda_runs (ranker_name, seed, num_agents, ticks_target) "
                f"VALUES ('{rn}', {s}, {int(num_agents)}, {int(ticks_target)}) RETURNING id",
                capture=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300])
            self._id = int(proc.stdout.decode().split()[0])
        except Exception as e:  # noqa: BLE001
            self._warn(e)

    def resume(self, ranker_name: str, seed, num_agents: int, ticks_target: int):
        """--resume: adopt the latest run row and continue it, returning
        (start_tick, llm_calls, llm_cost_usd, tallies) so the caller carries
        progress + spend forward. If no row exists yet (e.g. a freshly provisioned
        world), fall back to a fresh start() and return zeros."""
        zero = {"post": 0, "reply": 0, "quote": 0, "favourite": 0, "boost": 0}
        if not self._url:
            return 0, 0, 0.0, dict(zero)
        try:
            self._run_or_raise(_RUNS_DDL)
            proc = self._psql(
                "SELECT id, ticks_done, llm_calls, llm_cost_usd, post_count, "
                "reply_count, quote_count, favourite_count, boost_count "
                "FROM corda_runs ORDER BY id DESC LIMIT 1",
                capture=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300])
            out = proc.stdout.decode().strip()
            if not out:  # nothing to resume — behave like a fresh start
                self.start(ranker_name, seed, num_agents, ticks_target)
                return 0, 0, 0.0, dict(zero)
            f = out.split("|")
            self._id = int(f[0])
            tallies = {"post": int(f[4]), "reply": int(f[5]), "quote": int(f[6]),
                       "favourite": int(f[7]), "boost": int(f[8])}
            return int(f[1]), int(f[2]), float(f[3]), tallies
        except Exception as e:  # noqa: BLE001
            self._warn(e)
            return 0, 0, 0.0, dict(zero)

    def heartbeat(self, ticks_done: int, llm_calls: int, llm_cost_usd: float, tallies: dict) -> None:
        self._update(ticks_done, llm_calls, llm_cost_usd, tallies, final=False)

    def finish(self, ticks_done: int, llm_calls: int, llm_cost_usd: float, tallies: dict) -> None:
        self._update(ticks_done, llm_calls, llm_cost_usd, tallies, final=True)

    def _update(self, ticks_done, llm_calls, llm_cost_usd, tallies, final) -> None:
        if not self._url or self._id is None:
            return
        try:
            cols = (
                f"ticks_done={int(ticks_done)}, llm_calls={int(llm_calls)}, "
                f"llm_cost_usd={float(llm_cost_usd):.4f}, last_tick_at=now(), "
                f"duration=EXTRACT(EPOCH FROM (now() - started_at)), "
                f"action_count={sum(int(v) for v in tallies.values())}, "
                f"post_count={int(tallies.get('post', 0))}, "
                f"favourite_count={int(tallies.get('favourite', 0))}, "
                f"boost_count={int(tallies.get('boost', 0))}, "
                f"quote_count={int(tallies.get('quote', 0))}, "
                f"reply_count={int(tallies.get('reply', 0))}"
            )
            if final:
                cols += ", ended_at=now()"
            self._run_or_raise(f"UPDATE corda_runs SET {cols} WHERE id={self._id}")
        except Exception as e:  # noqa: BLE001
            self._warn(e)

    def _run_or_raise(self, sql: str) -> None:
        proc = self._psql(sql)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300])

    def _warn(self, e: Exception) -> None:
        self._failures += 1
        if self._failures == 1 or self._failures % 100 == 0:
            print(f"[runs] WARNING: write failed ({type(e).__name__}: {e}) — "
                  f"{self._failures} so far; continuing")
