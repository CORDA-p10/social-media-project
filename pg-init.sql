-- Per-world databases. mastodon_w1 is created by the postgres image via
-- POSTGRES_DB; this script adds the remaining four. Each world gets its own
-- database so engagement in one condition never leaks into another.
-- Runs only on first boot (empty data volume).
CREATE DATABASE mastodon_w2 OWNER mastodon;
CREATE DATABASE mastodon_w3 OWNER mastodon;
CREATE DATABASE mastodon_w4 OWNER mastodon;
CREATE DATABASE mastodon_w5 OWNER mastodon;
CREATE DATABASE mastodon_w6 OWNER mastodon;
