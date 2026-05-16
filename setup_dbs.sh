#!/usr/bin/env bash
# Creates two independent robopoker databases on the training server.
# Run --cluster on both machines in parallel; whichever finishes first
# wins and its DB_URL is used for --fast (phase 2) on the server.
#
# Usage:
#   ./setup_dbs.sh [postgres_url]
#
# Example:
#   ./setup_dbs.sh postgres://postgres@192.168.1.10:5432
#   ./setup_dbs.sh  # defaults to $DATABASE_URL or localhost

set -euo pipefail

BASE_URL="${1:-${DATABASE_URL:-postgres://postgres@localhost:5432}}"
DB_MAC="robopoker_mac"
DB_SERVER="robopoker_server"

echo "Creating databases on ${BASE_URL}..."
psql "${BASE_URL}/postgres" -c "CREATE DATABASE ${DB_MAC};"
psql "${BASE_URL}/postgres" -c "CREATE DATABASE ${DB_SERVER};"
echo "Done."

# ─── Tuning ──────────────────────────────────────────────────────────────────
# Apply the same performance settings to both databases.
for DB in "$DB_MAC" "$DB_SERVER"; do
    psql "${BASE_URL}/${DB}" <<-SQL
        ALTER SYSTEM SET shared_buffers          = '48GB';
        ALTER SYSTEM SET effective_cache_size    = '144GB';
        ALTER SYSTEM SET maintenance_work_mem    = '4GB';
        ALTER SYSTEM SET work_mem                = '256MB';
        ALTER SYSTEM SET max_wal_size            = '8GB';
        ALTER SYSTEM SET checkpoint_completion_target = '0.9';
        ALTER SYSTEM SET synchronous_commit      = 'off';
        SELECT pg_reload_conf();
SQL
    echo "Tuned ${DB}."
done

# ─── Instructions ────────────────────────────────────────────────────────────
MAC_URL="${BASE_URL}/${DB_MAC}"
SERVER_URL="${BASE_URL}/${DB_SERVER}"

cat <<-INSTRUCTIONS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE 1 — run both in parallel, on separate machines
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mac Studio:
  DB_URL="${MAC_URL}" \\
    cargo run --release --bin trainer -- --cluster

Server (192 GB):
  DB_URL="${SERVER_URL}" \\
    cargo run --release --bin trainer -- --cluster

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE 2 — run on the server using whichever DB won
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If Mac finished first:
  DB_URL="${MAC_URL}" \\
    cargo run --release --bin trainer -- --fast

If Server finished first:
  DB_URL="${SERVER_URL}" \\
    cargo run --release --bin trainer -- --fast

Check clustering status on either DB at any time:
  DB_URL="${MAC_URL}"    cargo run --release --bin trainer -- --status
  DB_URL="${SERVER_URL}" cargo run --release --bin trainer -- --status

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS
