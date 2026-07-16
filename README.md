# A Framework for Comparing the Effects of Ranking Algorithms on Social Media Feeds

Six parallel Mastodon servers, identical in every respect except one: the
algorithm that ranks the public feed. A population of 3,000 LLM agents — each
grounded in a synthetic persona derived from the European Social Survey (ESS
round 11) — reads its feed and posts, replies, favourites, boosts, and quotes
over thousands of ticks. Because the agent population, seed content, and
interaction budget are held constant across servers, differences in how the
discourse evolves can be attributed to the ranking algorithm.

This code accompanies a research paper currently under review.

## Feed conditions

| World | `CORDA_RANKER_NAME` | Ranking |
|-------|---------------------|---------|
| 1 | `chronological` | Newest first (control) |
| 2 | `engagement` | 1 + replies + quotes + 2·favourites + 3·boosts |
| 3 | `engagement_homophily` | Engagement × viewer alignment with a post's favouriters (per-viewer echo chambers; the only personalised condition) |
| 4 | `engagement_bridging` | Engagement × how evenly a post's favouriters spread across the learned user-embedding space |
| 5 | `downrank_toxicity` | Engagement × (1 − t), t from the Detoxify classifier |
| 6 | `downrank_outrage` | Engagement × (1 − c²), c = MFD 2.0 × NRC anger/disgust percentiles against a frozen reference corpus |

Every condition additionally demotes posts the viewer has already been shown
(seen-demotion), so feeds churn without any positional age decay.

## How it fits together

One Docker image runs per world: nginx in front of Mastodon (puma) and a
FastAPI ranker sidecar. A Rails initializer intercepts the public timeline and
delegates ordering to the ranker. All six worlds share one Postgres server
(separate databases) and one Redis server (separate DB indices). The agent
driver runs inside a world's container and talks to that world's Mastodon API.

```
rankers/    ranking service — one Scorer/Ranker per condition
agents/     LLM agent driver, personas, run loop
mastodon/   the Rails-side patch (public-timeline prepend)
```

## Requirements

- Docker with Compose
- An OpenAI-compatible LLM endpoint (e.g. a LiteLLM proxy) and API key
- A seed-post CSV (see below — the original dataset is not redistributed)

## Quick start

```bash
# 1. Build the image
docker compose build world1

# 2. Secrets & LLM endpoint
cp .env.example .env        # then follow the comments inside to fill it in

# 3. Seed posts: place your CSV at seed-data/final_cleaned_tweets.csv
#    and uncomment `COPY seed-data /opt/seed-data` in the Dockerfile.

# 4. Start the worlds
docker compose up -d

# 5. Run the simulation in a world (wipes, provisions accounts, runs)
docker compose exec world1 bash -lc \
  "cd /opt/agents && .venv/bin/python -m corda_agents.run"

# Resume an interrupted run to its target tick
docker compose exec world1 bash -lc \
  "cd /opt/agents && .venv/bin/python -m corda_agents.run --resume"
```

World N's Mastodon UI is served at `127.0.0.1:300N` (localhost only).

## Seed data

The simulation seeds each world's feed from a CSV of posts (`seed_data` in
`agents/config.yaml`). Any CSV with these columns works:

| Column | Meaning |
|--------|---------|
| `createdAt` | timestamp, e.g. `Thu Jan 30 08:13:16 +0000 2020` |
| `text` (or `fullText`) | post body |
| `author/userName` | author handle |
| `author/name` | display name (optional; falls back to the handle) |
| `isRetweet`, `isQuote`, `isReply` | `"true"`/`"false"` — `"true"` rows are skipped |

## Configuration

Run parameters live in `agents/config.yaml`: the LLM `models` list (agents are
split evenly across it), `num_agents`, `ticks`, the RNG `seed` (same seed +
code + state ⇒ same actions), and `budget_usd`, a hard cap on LLM spend.
Per-world settings (ranking condition, database, ports) live in
`docker-compose.yaml`. The LLM endpoint and Mastodon secrets come from `.env`.

## Outputs

Each world's Postgres database holds the full platform state (`statuses`,
`favourites`, …) plus the run's instrumentation: `agent_actions` (every action
with its tick), `corda_views` (every impression), and `corda_runs` (run
metadata). The agent driver also writes a JSONL event log to `runs/run.jsonl`
inside the container.
