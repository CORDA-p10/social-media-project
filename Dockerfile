# syntax=docker/dockerfile:1.7
# Mastodon + custom ranker. One image, run once per "world" via
# docker-compose (see docker-compose.yml).
#
# Build context: src/  (Dockerfile sits at src/Dockerfile)
# Process tree: nginx :PORT  →  puma :3000, ranker :8000
# External services: Postgres + Redis (DATABASE_URL, REDIS_URL from compose).
# (Streaming intentionally not run — closed instance doesn't need real-time pushes.)

ARG MASTODON_TAG=v4.5
FROM ghcr.io/mastodon/mastodon:${MASTODON_TAG}

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx \
        supervisor \
        gosu \
        postgresql-client \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV LIMITED_FEDERATION_MODE=true \
    AUTHORIZED_FETCH=true \
    SINGLE_USER_MODE=false \
    DEFAULT_LOCALE=en \
    RAILS_ENV=production \
    NODE_ENV=production \
    PYTHONUNBUFFERED=1 \
    RAILS_SERVE_STATIC_FILES=true \
    RAILS_LOG_TO_STDOUT=true \
    ES_ENABLED=false \
    SMTP_DELIVERY_METHOD=test \
    DISABLE_REGISTRATION_CAPTCHA=true \
    BIND=0.0.0.0 \
    CORDA_RANKER_URL=http://127.0.0.1:8000 \
    CORDA_RANKER_PORT=8000

# Mastodon-side patch: the ranker prepend (config/initializers/corda_ranker.rb).
COPY --chown=mastodon:mastodon mastodon/config/initializers/corda_ranker.rb \
    /opt/mastodon/config/initializers/corda_ranker.rb

# Process orchestration + reverse proxy.
RUN mkdir -p /var/lib/nginx /var/log/nginx /run
COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisor/conf.d/mastodon.conf
COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh

# ── Ranker venv ──
COPY rankers/pyproject.toml /opt/rankers/pyproject.toml
RUN mkdir -p /opt/rankers/corda_rankers \
    && touch /opt/rankers/corda_rankers/__init__.py \
    && python3 -m venv /opt/rankers/.venv
# Pin torch to the CPU-only wheel (this box has no GPU) — ~200 MB vs the ~2 GB
# CUDA build that detoxify's torch dep would otherwise pull. Installed BEFORE
# the ranker deps so the resolve below sees torch already satisfied and doesn't
# fetch the CUDA wheel. (detoxify → downrank_toxicity only.)
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/rankers/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/rankers/.venv/bin/pip install --compile /opt/rankers
COPY rankers /opt/rankers
RUN /opt/rankers/.venv/bin/pip install --no-cache-dir --no-deps -e /opt/rankers

# downrank_outrage: bake the nltk data its POS-aware lemmatiser needs. Only the
# outrage world loads it (lazy), but — like the ranker deps — it installs image-
# wide. NLTK_DATA persists to the running container so nltk finds the data.
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN /opt/rankers/.venv/bin/python -m nltk.downloader -d /usr/local/share/nltk_data \
        wordnet omw-1.4 punkt_tab averaged_perceptron_tagger_eng

# ── Agents venv ──
COPY agents/pyproject.toml /opt/agents/pyproject.toml
RUN mkdir -p /opt/agents/corda_agents \
    && touch /opt/agents/corda_agents/__init__.py \
    && python3 -m venv /opt/agents/.venv
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/agents/.venv/bin/pip install --compile /opt/agents
COPY agents /opt/agents
RUN /opt/agents/.venv/bin/pip install --no-cache-dir --no-deps -e /opt/agents

# Real-world seed-post datasets for the injector (config `seed_data`).
# COPY seed-data /opt/seed-data

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
