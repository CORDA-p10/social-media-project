#!/usr/bin/env bash
set -euo pipefail

: "${PORT:=10000}"
: "${LOCAL_DOMAIN:?LOCAL_DOMAIN must be set}"
: "${DATABASE_URL:?DATABASE_URL must be set}"
: "${REDIS_URL:?REDIS_URL must be set}"

export DATABASE_URL="${DATABASE_URL/postgres:\/\//postgresql://}"

# Mastodon → Redis DB 0; ranker → DB 1 (no key collisions with sidekiq).
if [[ -z "${CORDA_RANKER_REDIS_URL:-}" ]]; then
    export CORDA_RANKER_REDIS_URL=$(python3 -c '
import os
from urllib.parse import urlparse, urlunparse
u = urlparse(os.environ["REDIS_URL"])
print(urlunparse(u._replace(path="/1")))
')
fi

: "${SECRET_KEY_BASE:?SECRET_KEY_BASE must be set}"
: "${OTP_SECRET:?OTP_SECRET must be set}"
: "${VAPID_PUBLIC_KEY:?VAPID_PUBLIC_KEY must be set (generate once with: bundle exec rake mastodon:webpush:generate_vapid_key)}"
: "${VAPID_PRIVATE_KEY:?VAPID_PRIVATE_KEY must be set (generate once with: bundle exec rake mastodon:webpush:generate_vapid_key)}"
export SECRET_KEY_BASE OTP_SECRET VAPID_PUBLIC_KEY VAPID_PRIVATE_KEY

sed -e "s/__PORT__/${PORT}/" -e "s/__LOCAL_DOMAIN__/${LOCAL_DOMAIN}/" /etc/nginx/nginx.conf > /etc/nginx/nginx.conf.rendered
mv /etc/nginx/nginx.conf.rendered /etc/nginx/nginx.conf

# Persist uploaded media (avatars) on a named volume. A fresh named volume
# mounts empty and root-owned, so make sure the dir exists and is writable by
# the mastodon user — otherwise avatar attach fails and media vanishes on every
# container recreate.
mkdir -p /opt/mastodon/public/system && chown mastodon:mastodon /opt/mastodon/public/system

cd /opt/mastodon
gosu mastodon bash -lc "bundle exec rails db:migrate" || \
    echo "[entrypoint] db:migrate failed; check Postgres link."

# Seed default UserRoles (Owner/Admin/Moderator) + instance defaults. Idempotent
# (find_or_create); reset.py's wipe preserves user_roles, so once is enough — but
# running every boot is harmless and makes fresh worlds self-provision.
gosu mastodon bash -lc "bundle exec rails db:seed" || \
    echo "[entrypoint] db:seed failed; default roles (Owner/...) may be missing."

# Persist for supervisord children.
{
    echo "SECRET_KEY_BASE=$SECRET_KEY_BASE"
    echo "OTP_SECRET=$OTP_SECRET"
    echo "VAPID_PUBLIC_KEY=$VAPID_PUBLIC_KEY"
    echo "VAPID_PRIVATE_KEY=$VAPID_PRIVATE_KEY"
    echo "DATABASE_URL=$DATABASE_URL"
    echo "REDIS_URL=$REDIS_URL"
    echo "CORDA_RANKER_REDIS_URL=$CORDA_RANKER_REDIS_URL"
    echo "CORDA_RANKER_URL=${CORDA_RANKER_URL}"
    echo "CORDA_RANKER_PORT=${CORDA_RANKER_PORT}"
} > /opt/mastodon/.env.runtime
chown mastodon:mastodon /opt/mastodon/.env.runtime

exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
