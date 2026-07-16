"""Mid-run injection of real-world seed posts into the world.

The dataset (config `seed_data`) is a CSV of scraped tweets with flattened
columns (`author/userName`, `author/id`, `createdAt`, `text`, `isQuote`,
`isRetweet`, `isReply`, …). Rows are injected chronologically (sorted by
`createdAt`); retweets, quote tweets, and replies are skipped at load — they
reference content that doesn't exist in the world — and the skip count is
logged. Each author's display name comes from the seed data (`author/name`);
the username from `author/userName`.

Cadence — the injected-post count is topped up to

    target = floor(agent_authored_posts / 4) + INITIAL_POSTS

at the end of every tick (and once before tick 0, giving the world its
20 starting posts). "Agent-authored posts" counts TOP-LEVEL statuses only —
the post and quote actions — because those are what the public timeline
shows; replies (the bulk of agent output) and boosts never appear as
standalone feed items, so counting them would flood the visible feed with
injected posts. Most ticks add 0 to the count, so top-ups are sparse — the
cadence tracks the running post count, not the tick number, and a top-up
fires within the tick whose posts crossed a multiple of 4:

    agent posts:  0…3   4…7   8…11  ...
    injected:      20    21    22   ...

The visible timeline therefore holds one injected post per four agent posts
(plus the 20 starters).

Injected authors get an Account plus a MINIMAL User row — random throwaway
password, no OAuth token, never logs in. The User row is not optional:
Mastodon's web layer (profile pages, status pages, and the /embed pages the
meta dashboard renders) 404s any local account without one, even though the
API serves its statuses fine. No token still rules out the REST API, so
posting happens inside Rails via a single persistent `rails runner`
co-process: it boots once (~15s, at injector construction) and then serves
one command per injected post — find-or-create the author,
`Status.create!` attributed to the "Post from X" Doorkeeper app.
The app name is the provenance label: it is what the meta page's /export emits
in the `client` column (agent posts say "corda-agents"), and what Mastodon's
web UI shows as the posting application.

Transport is deliberately dumb: the runner polls a command FILE and writes a
result FILE, both under /tmp, with readiness signalled through a state file
and the runner's stdout/stderr appended to a log file. No pipes, no FIFOs —
every earlier pipe-based transport (stdin, named FIFO, stdout readline) hit
hard-to-observe buffering/lifecycle deadlocks under the driver; plain files
poll at 200ms, which is negligible against the one-post-per-~4-ticks cadence,
and leave everything inspectable on disk when something goes wrong.
"""

from __future__ import annotations

import csv
import html
import json
import os
import subprocess
import textwrap
import time
from datetime import datetime
from pathlib import Path

from .reset import EMAIL_DOMAIN, MASTODON_DIR, _log

APP_NAME = "Post from X"
INITIAL_POSTS = 20       # injected before tick 0 — the world's starting feed
MAX_CHARS = 500          # Mastodon's default status length limit
_RUNNER_PATH = Path("/tmp/corda_seed_runner.rb")
_STATE_PATH = Path("/tmp/corda_seed_state")      # "ready" once Rails booted
_CMD_PATH = Path("/tmp/corda_seed_cmd.json")     # request:  {seq, username, text}
_RES_PATH = Path("/tmp/corda_seed_res.json")     # response: {seq, ok, id|error}
_STOP_PATH = Path("/tmp/corda_seed_stop")        # touch → runner exits
_LOG_PATH = Path("/tmp/corda_seed_runner.log")
_BOOT_TIMEOUT = 300.0    # Rails boot on a busy 2-core box can be slow
_CMD_TIMEOUT = 60.0
_POLL = 0.2

_RUNNER_SCRIPT = textwrap.dedent(f"""
    app = Doorkeeper::Application.find_or_create_by!(name: {APP_NAME!r}) do |a|
      a.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
      a.scopes       = "read write follow"
    end
    CMD, RES, STATE, STOP = {str(_CMD_PATH)!r}, {str(_RES_PATH)!r}, {str(_STATE_PATH)!r}, {str(_STOP_PATH)!r}
    done = -1
    File.write(STATE, "ready")
    loop do
      break if File.exist?(STOP)
      unless File.exist?(CMD)
        sleep 0.2
        next
      end
      begin
        spec = JSON.parse(File.read(CMD))
        next (sleep 0.2) if spec['seq'].to_i <= done
        done = spec['seq'].to_i
        account = Account.find_or_create_by!(username: spec['username'], domain: nil) do |a|
          a.display_name = spec['display_name'].to_s   # set on create only
        end
        if account.user.nil?
          Devise.stretches = 4
          u = User.new(
            email:                      "#{{spec['username']}}@{EMAIL_DOMAIN}",
            password:                   SecureRandom.hex(16),
            agreement:                  true,
            approved:                   true,
            confirmed_at:               Time.now.utc,
            bypass_registration_checks: true,
            account:                    account,
          )
          u.save!
          u.approve! unless u.approved?  # set_approved forces false on closed registrations
        end
        status = Status.create!(
          account:     account,
          text:        spec['text'],
          visibility:  :public,
          application: app,
        )
        reply = {{seq: done, ok: true, id: status.id.to_s}}
      rescue => e
        reply = {{seq: done, ok: false, error: "#{{e.class}}: #{{e.message}}"}}
      end
      File.write(RES + ".tmp", reply.to_json)
      File.rename(RES + ".tmp", RES)
    end
""")


_CREATED_AT_FMT = "%a %b %d %H:%M:%S %z %Y"  # e.g. "Thu Jan 30 08:13:16 +0000 2020"


def _load_dataset(path: Path) -> list[dict[str, str]]:
    """Usable rows, sorted chronologically by createdAt: {username,
    display_name, text}. Retweets, quote tweets, and replies are skipped —
    their text refers to another post that doesn't exist in this world — as
    are rows with no author, no text, or an unparseable date."""
    kept: list[tuple[datetime, dict[str, str]]] = []
    skipped = unusable = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if any((row.get(k) or "").strip().lower() == "true"
                   for k in ("isRetweet", "isQuote", "isReply")):
                skipped += 1
                continue
            username = (row.get("author/userName") or "").strip()
            display_name = (row.get("author/name") or "").strip() or username
            # Scraper text carries HTML entities (&amp; …) — unescape so agents
            # read "&", not markup.
            text = html.unescape((row.get("text") or row.get("fullText") or "").strip())
            try:
                created = datetime.strptime((row.get("createdAt") or "").strip(),
                                            _CREATED_AT_FMT)
            except ValueError:
                created = None
            if not username or not text or created is None:
                unusable += 1
                continue
            if len(text) > MAX_CHARS:
                text = text[: MAX_CHARS - 1] + "…"
            kept.append((created, {
                "username": username,
                "display_name": display_name,
                "text": text,
            }))
    kept.sort(key=lambda pair: pair[0])
    _log(f"injector: {path} → {len(kept)} usable posts, chronological "
         f"(skipped {skipped} retweet/quote/reply rows, {unusable} unusable)")
    return [item for _, item in kept]


class SeedPostInjector:
    """Owns the dataset cursor, the Rails co-process, and the cadence."""

    def __init__(self, dataset_path: Path) -> None:
        self.items = _load_dataset(dataset_path)
        self.injected = 0
        self._cursor = 0
        self._seq = 0
        self._consecutive_failures = 0
        self.disabled = False
        self._proc = self._start_runner()

    # ── cadence ─────────────────────────────────────────────────────────

    @staticmethod
    def target(n_agent_posts: int) -> int:
        return n_agent_posts // 4 + INITIAL_POSTS

    def maintain(self, n_agent_posts: int) -> None:
        """Top the injected count up to target. Called after every tick."""
        while not self.disabled and self.injected < self.target(n_agent_posts):
            if not self._inject_next():
                break

    def resume_from(self, n_agent_posts: int) -> None:
        """On --resume, the world already holds ~target(n_agent_posts) seed posts
        from before the interruption. Advance the counters past them so maintain()
        tops up only new demand rather than re-injecting the whole backlog (which
        would double the injected feed and its cost)."""
        self.injected = self.target(n_agent_posts)
        self._cursor = min(self.injected, len(self.items))

    def close(self) -> None:
        _STOP_PATH.touch()  # runner's poll loop sees it and exits
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    # ── injection ───────────────────────────────────────────────────────

    def _inject_next(self) -> bool:
        """Post the next dataset item. False when the dataset is exhausted or
        the injector tripped its failure breaker."""
        while self._cursor < len(self.items):
            item = self.items[self._cursor]
            self._cursor += 1
            try:
                status_id = self._send(item)
            except Exception as e:  # noqa: BLE001 — a bad item must not kill the run
                self._consecutive_failures += 1
                _log(f"injector: item {self._cursor - 1} (@{item['username']}) failed "
                     f"({type(e).__name__}: {str(e)[:300]}) — skipping")
                if self._consecutive_failures >= 3:
                    self.disabled = True
                    _log("injector: 3 consecutive failures — disabling injection "
                         "for the rest of the run")
                    return False
                continue
            self._consecutive_failures = 0
            self.injected += 1
            _log(f"injector: posted as @{item['username']} "
                 f"(status id {status_id}, {self.injected} injected)")
            return True
        _log(f"injector: dataset exhausted after {self.injected} injected posts")
        return False

    # ── Rails co-process (file-based transport) ─────────────────────────

    def _start_runner(self) -> subprocess.Popen | None:
        if not self.items:
            return None
        for p in (_STATE_PATH, _CMD_PATH, _RES_PATH, _STOP_PATH):
            p.unlink(missing_ok=True)
        _RUNNER_PATH.write_text(_RUNNER_SCRIPT)
        log_f = _LOG_PATH.open("w")
        proc = subprocess.Popen(
            ["gosu", "mastodon", "bash", "-c",
             f"cd {MASTODON_DIR} && RAILS_ENV=production "
             f"bundle exec rails runner {_RUNNER_PATH}"],
            stdin=subprocess.DEVNULL, stdout=log_f, stderr=log_f,
        )
        _log("injector: booting Rails seed-post runner…")
        deadline = time.monotonic() + _BOOT_TIMEOUT
        while time.monotonic() < deadline:
            if _STATE_PATH.exists() and _STATE_PATH.read_text() == "ready":
                _log(f"injector: runner ready (client app: {APP_NAME!r})")
                return proc
            if proc.poll() is not None:
                break
            time.sleep(_POLL)
        proc.kill()
        self.disabled = True
        tail = _LOG_PATH.read_text()[-400:] if _LOG_PATH.exists() else ""
        _log(f"injector: seed-post runner failed to boot — injection disabled "
             f"(log tail: {tail!r})")
        return None

    def _send(self, item: dict[str, str]) -> str:
        """One command round-trip via cmd/res files. Returns the new status id."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("seed-post runner is not running")
        self._seq += 1
        _RES_PATH.unlink(missing_ok=True)
        tmp = _CMD_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({"seq": self._seq, **item}))
        tmp.rename(_CMD_PATH)  # atomic — the runner never sees a partial file
        deadline = time.monotonic() + _CMD_TIMEOUT
        while time.monotonic() < deadline:
            if _RES_PATH.exists():
                reply = json.loads(_RES_PATH.read_text())
                if reply.get("seq") == self._seq:
                    if not reply.get("ok"):
                        raise RuntimeError(reply.get("error", "unknown runner error"))
                    return reply["id"]
            if self._proc.poll() is not None:
                raise RuntimeError("seed-post runner exited unexpectedly")
            time.sleep(_POLL)
        raise TimeoutError(f"runner produced no reply within {_CMD_TIMEOUT}s")
