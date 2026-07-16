"""Wipe + recreate the Mastodon instance for a fresh simulation run.

Imported and invoked by `corda_agents.run` unless `--no-reset` is passed
(reset is the default). See that module's docstring for the high-level
step list. Steps batch their work to bound Rails-boot overhead and peak
memory:

  * Every `rails runner` boot is ~10–30s of pure Rails startup, so we
    collapse the small scripts into two combined runners (one before
    account creation, one after).
  * Account creation and avatar attach are run as separate batched
    passes. RSA keypair generation (account creation) and Paperclip
    image processing (avatar attach) are each memory-heavy; doing both
    in the same process for ~1000 accounts exhausts memory. Splitting
    them lets each phase use a larger batch.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import yaml

from .llm_costs import MODEL_CLIENT_LABELS
from .persona import Persona


MASTODON_DIR = Path(os.environ.get("MASTODON_DIR", "/opt/mastodon"))

EMAIL_DOMAIN = "aisafetyaachen.org"
ADMIN_USERNAME = "admin"
ADMIN_EMAIL = f"corda@{EMAIL_DOMAIN}"
ADMIN_DISPLAY_NAME = "CORDA"
ADMIN_AVATAR_PATH = Path(__file__).resolve().parent.parent / "corda_logo.png"

DICEBEAR_BACKGROUND_COLORS: tuple[str, ...] = (
    "93a7ff", "a9e775", "ff7a9a", "b379f7", "ff6674", "89e6e4", "ffcc65",
)

# Avatar attach shards across this many parallel `rails runner` processes — the
# box has 2 cores, and attach is the heaviest provisioning phase.
_ATTACH_WORKERS = 2

_T0 = time.monotonic()


def _log(msg: str) -> None:
    print(f"[run] {msg}  (+{int(time.monotonic() - _T0)}s)", flush=True)


def _rails(*args: str, env: dict[str, str] | None = None) -> None:
    """Run a rails command as the mastodon user, with production env. Rails'
    own INFO chatter (Paperclip, Sidekiq, initializers) is captured and only
    reprinted if the command fails — keeps the reset transcript readable."""
    full_env = {"RAILS_ENV": "production", **(env or {})}
    env_str = " ".join(f"{k}={v}" for k, v in full_env.items())
    inner = " ".join(["bundle", "exec", "rails", *args])
    cmd = ["gosu", "mastodon", "bash", "-c", f"cd {MASTODON_DIR} && {env_str} {inner}"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"rails {' '.join(args)} failed with exit {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def _rails_capture(script_path: Path) -> str:
    """Run `rails runner SCRIPT` and return stdout. Used when the script
    prints tab-prefixed result lines we need to parse."""
    inner = f"cd {MASTODON_DIR} && RAILS_ENV=production bundle exec rails runner {script_path}"
    cmd = ["gosu", "mastodon", "bash", "-c", inner]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"rails runner {script_path} failed with exit {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result.stdout


def _wipe_db_and_redis() -> None:
    """One Rails boot that TRUNCATEs every data table CASCADE and flushes
    Redis. Preserves what Rails / Mastodon structurally need
    (schema_migrations, user_roles — other tables FK into it) and the
    `settings` table, so admin tweaks like custom CSS and "disable Trends"
    survive a reset. `_finalize_branding_and_tokens` rewrites the contact
    rows it owns regardless."""
    script_path = Path("/tmp/corda_wipe.rb")
    script_path.write_text(textwrap.dedent("""
        keep = %w[schema_migrations ar_internal_metadata settings user_roles]
        tables = ActiveRecord::Base.connection.tables - keep
        quoted = tables.map { |t| %("#{t}") }.join(", ")
        ActiveRecord::Base.connection.execute(
          "TRUNCATE TABLE #{quoted} RESTART IDENTITY CASCADE"
        )
        # Sidekiq.redis is just the idiomatic handle to Mastodon's Redis DB 0 —
        # it works whether or not the Sidekiq worker runs (it doesn't here, see
        # supervisord.conf). With no fan-out/queues, DB 0 holds little beyond the
        # Rails cache, but flushing it still drops stale cached counts/settings
        # that would otherwise reference truncated rows. NOTE: this only touches
        # DB 0 — the ranker's per-viewer seen-sets live in CORDA_RANKER_REDIS_URL
        # (DB 1/3/5) and are cleared separately via the ranker's /seen/reset
        # (called from run.py after this reset).
        Sidekiq.redis { |conn| conn.flushdb }
    """))
    _log("wiping data tables + Redis DB 0 (Rails cache)")
    _rails("runner", str(script_path))


def _create_accounts_bulk(specs: list[dict[str, str | None]]) -> dict[str, str]:
    """Create all N accounts in a single `rails runner` pass. Account creation
    is light now — one shared RSA keypair (not one per account) plus minimal
    bcrypt cost — so there's no memory reason to batch. Avatar attach is still a
    separate pass (see `_attach_avatars_bulk`).

    Each spec is {"username": str, "email": str, "role": Optional[str],
    "display_name": Optional[str]}. Returns {username: generated_password}.
    """
    script_path = Path("/tmp/corda_create_accounts.rb")
    script_path.write_text(textwrap.dedent("""
        require 'json'
        require 'securerandom'
        # Closed sim: no password guards anything of value — agents authenticate
        # with OAuth tokens, and the only password login is the admin web UI. Drop
        # bcrypt's work factor to its minimum (4) so hashing ~1000 passwords stops
        # dominating account creation (Mastodon's default cost 12 ≈ ~250ms each ≈
        # 4+ min for 1000 accounts).
        Devise.stretches = 4
        # Closed sim never federates, so per-account RSA identity keys are dead
        # weight. Account#generate_keys (a before_create) skips keygen when both
        # keys are already set, so generate ONE keypair here and reuse it for
        # every account — turning ~1000 RSA-2048 keygens into one.
        require 'openssl'
        keypair = OpenSSL::PKey::RSA.new(2048)
        shared_private_key = keypair.to_pem
        shared_public_key  = keypair.public_key.to_pem
        specs = JSON.parse(File.read(ARGV[0]))
        specs.each do |spec|
          password = SecureRandom.hex(16)
          account_attrs = {
            username:    spec['username'],
            private_key: shared_private_key,
            public_key:  shared_public_key,
          }
          account_attrs[:display_name] = spec['display_name'] if spec['display_name']
          user = User.new(
            email:                          spec['email'],
            password:                       password,
            agreement:                      true,
            approved:                       true,
            confirmed_at:                   Time.now.utc,
            bypass_registration_checks:     true,
            account_attributes:             account_attrs,
          )
          user.role = UserRole.find_by!(name: spec['role']) if spec['role']
          user.save!
          user.approve! unless user.approved?
          puts "ACCOUNT\\t#{spec['username']}\\t#{password}"
        end
    """))
    specs_path = Path("/tmp/corda_create_accounts.json")
    specs_path.write_text(json.dumps(specs))
    _log(f"creating {len(specs)} accounts")
    inner = (
        f"cd {MASTODON_DIR} && RAILS_ENV=production "
        f"bundle exec rails runner {script_path} {specs_path}"
    )
    cmd = ["gosu", "mastodon", "bash", "-c", inner]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"rails runner (account creation) failed with exit {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

    passwords: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if line.startswith("ACCOUNT\t"):
            parts = line.split("\t")
            if len(parts) == 3:
                _, username, password = parts
                passwords[username] = password

    if len(passwords) != len(specs):
        raise RuntimeError(
            f"created {len(passwords)} accounts for {len(specs)} specs"
        )
    return passwords


def _attach_avatars_bulk(avatar_by_username: dict[str, str]) -> None:
    """Attach avatars to existing accounts. The image work is libvips (fast); the
    per-account cost is file read + account.save! + Rails overhead, so we shard
    the work across _ATTACH_WORKERS parallel `rails runner` processes to use both
    cores. Run separately from account creation. No per-record GC.start — memory
    fits comfortably for ~1000 small WebP avatars (validated peak ~2.3 GB)."""
    if not avatar_by_username:
        return
    script_path = Path("/tmp/corda_attach_avatars.rb")
    script_path.write_text(textwrap.dedent("""
        require 'json'
        pairs = JSON.parse(File.read(ARGV[0]))
        pairs.each do |username, avatar_path|
          account = Account.find_by!(username: username, domain: nil)
          File.open(avatar_path, 'rb') do |f|
            account.avatar = f
            account.save!
          end
        end
    """))

    items = list(avatar_by_username.items())
    n_workers = min(_ATTACH_WORKERS, len(items))
    # Round-robin into disjoint shards — each worker attaches different accounts,
    # so there's no DB write contention between them.
    shards = [dict(items[i::n_workers]) for i in range(n_workers)]
    _log(f"attaching {len(items)} avatars across {n_workers} workers")

    procs = []
    for idx, shard in enumerate(shards):
        shard_path = Path(f"/tmp/corda_attach_avatars_{idx}.json")
        shard_path.write_text(json.dumps(shard))
        inner = (
            f"cd {MASTODON_DIR} && RAILS_ENV=production "
            f"bundle exec rails runner {script_path} {shard_path}"
        )
        cmd = ["gosu", "mastodon", "bash", "-c", inner]
        procs.append((idx, subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, text=True)))

    errors = []
    for idx, p in procs:
        out, err = p.communicate()
        if p.returncode != 0:
            errors.append(
                f"avatar attach worker {idx} failed with exit {p.returncode}\n"
                f"--- stdout ---\n{out}\n--- stderr ---\n{err}"
            )
    if errors:
        raise RuntimeError("\n".join(errors))


def _fetch_avatars_parallel(
    usernames: list[str],
    out_dir: Path,
    seed: int | None,
    style: str = "personas",
    max_workers: int = 5,
    max_retries: int = 6,
) -> dict[str, Path]:
    """Download a DiceBear WebP avatar per username in parallel into *out_dir*,
    which is a persistent cache (a Docker volume shared across worlds). Returns
    {username: local_webp_path}. Run this *before* attaching, so the slow
    network leg doesn't serialise behind Rails image processing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    bg = ",".join(DICEBEAR_BACKGROUND_COLORS)
    cached_before = sum(
        1 for u in usernames if (out_dir / f"{u}_s{seed}.webp").exists()
    )

    def fetch_one(username: str) -> tuple[str, Path]:
        s = f"{username}_s{seed}"
        path = out_dir / f"{s}.webp"
        if path.exists() and path.stat().st_size > 0:
            return username, path  # cache hit — skip the network entirely
        url = f"https://api.dicebear.com/10.x/{style}/webp?seed={s}&backgroundColor={bg}"
        delay = 1.0
        for attempt in range(max_retries):
            r = client.get(url)
            if r.status_code == 429:
                wait = float(r.headers.get("retry-after", delay))
                time.sleep(wait)
                delay = min(delay * 2, 30.0)
                continue
            r.raise_for_status()
            path.write_bytes(r.content)
            return username, path
        raise RuntimeError(f"DiceBear 429 persisted after {max_retries} retries for {username!r}")

    with httpx.Client(timeout=15.0) as client, ThreadPoolExecutor(max_workers=max_workers) as pool:
        for username, path in pool.map(fetch_one, usernames):
            paths[username] = path

    _log(f"avatars ready: {len(paths)} ({cached_before} cached, {len(paths) - cached_before} downloaded) [{style}]")
    return paths


def _finalize_branding_and_tokens(
    contact_username: str,
    contact_email: str,
    app_by_username: dict[str, str],
) -> dict[str, str]:
    """One Rails boot that (a) sets Administration → Server settings →
    Branding → Contact via Form::AdminSettings (same path as the admin web
    form, so cache hooks fire) and (b) creates one Doorkeeper OAuth app per
    distinct app name and mints each user's token from their assigned app.
    The app name is the "client" label Mastodon shows on posts (and /export's
    client column) — in mixed-model runs each model gets its own label, so
    agent posts are attributable to a model straight from the UI/export.
    Returns {username: access_token}."""
    spec_path = Path("/tmp/corda_finalize_specs.json")
    spec_path.write_text(json.dumps(app_by_username))
    script_path = Path("/tmp/corda_finalize.rb")
    script_path.write_text(textwrap.dedent(f"""
        require 'json'
        form = Form::AdminSettings.new(
          site_contact_username: {contact_username!r},
          site_contact_email:    {contact_email!r},
        )
        unless form.save
          raise "Form::AdminSettings save failed: " + form.errors.full_messages.join("; ")
        end

        app_by_username = JSON.parse(File.read({str(spec_path)!r}))
        apps = Hash.new do |cache, name|
          cache[name] = Doorkeeper::Application.find_or_create_by!(name: name) do |a|
            a.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
            a.scopes       = "read write follow"
          end
        end

        app_by_username.each do |username, app_name|
          account = Account.find_by(username: username, domain: nil)
          raise "no account for username=#{{username.inspect}}" unless account
          user = account.user
          raise "no user for username=#{{username.inspect}}"      unless user

          token = Doorkeeper::AccessToken.create!(
            application:       apps[app_name],
            resource_owner_id: user.id,
            scopes:            "read write follow",
            expires_in:        nil,
          )
          puts "TOKEN\\t#{{username}}\\t#{{token.plaintext_token}}"
        end
    """))
    n_apps = len(set(app_by_username.values()))
    _log(f"setting branding contact + minting OAuth tokens "
         f"({contact_username} <{contact_email}>, {len(app_by_username)} tokens "
         f"across {n_apps} client app(s))")
    stdout = _rails_capture(script_path)

    tokens: dict[str, str] = {}
    for line in stdout.splitlines():
        if line.startswith("TOKEN\t"):
            parts = line.split("\t")
            if len(parts) == 3:
                _, username, token = parts
                tokens[username] = token
    if len(tokens) != len(app_by_username):
        raise RuntimeError(
            f"minted {len(tokens)} tokens for {len(app_by_username)} usernames\n"
            f"--- stdout ---\n{stdout}"
        )
    return tokens


def reset(
    cfg: dict,
    personas: list[Persona],
) -> None:
    """TRUNCATE all data tables, recreate admin + personas, and mint OAuth
    tokens. Seed content comes from the injector (see run_inject.py), not
    from any pre-provisioned account."""
    domain = EMAIL_DOMAIN

    _wipe_db_and_redis()

    # Persona avatars come from a persistent cache (a Docker volume shared across
    # worlds — see docker-compose.yaml), keyed by (username, run seed), so each is
    # fetched once ever and the 2nd/3rd worlds of a run reuse world 1's files.
    avatar_paths = _fetch_avatars_parallel(
        usernames=[p.id for p in personas],
        out_dir=Path("/opt/avatar-cache"),
        seed=cfg.get("seed"),
    )

    specs: list[dict[str, str | None]] = [
        {
            "username": ADMIN_USERNAME,
            "email": ADMIN_EMAIL,
            "role": "Owner",
            "display_name": ADMIN_DISPLAY_NAME,
        },
        *[
            {
                "username": p.id,
                "email": f"{p.id}@{domain}",
                "role": None,
                "display_name": p.display_name,
            }
            for p in personas
        ],
    ]
    passwords = _create_accounts_bulk(specs)
    admin_pass = passwords[ADMIN_USERNAME]
    _log(f"admin password: {admin_pass}  (save this — needed to sign in)")

    accounts = {p.id: {"username": p.id, "password": passwords[p.id]} for p in personas}
    Path("accounts.yaml").write_text(yaml.safe_dump(accounts, sort_keys=True))
    _log(f"wrote accounts.yaml ({len(accounts)} agent accounts)")

    avatar_by_username: dict[str, str] = {
        ADMIN_USERNAME: str(ADMIN_AVATAR_PATH),
        **{p.id: str(avatar_paths[p.id]) for p in personas},
    }
    _attach_avatars_bulk(avatar_by_username)

    tokens = _finalize_branding_and_tokens(
        contact_username=ADMIN_USERNAME,
        contact_email=ADMIN_EMAIL,
        app_by_username={
            p.id: MODEL_CLIENT_LABELS.get(p.model, p.model)
            for p in personas
        },
    )
    Path(cfg["tokens_file"]).write_text(yaml.safe_dump(tokens, sort_keys=True))
    _log(f"wrote {cfg['tokens_file']} ({len(tokens)} tokens)")
