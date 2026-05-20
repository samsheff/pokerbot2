//! Database schema initialization for the trainer binary.
use rbp_cards::Street;
use rbp_clustering::{Future, Lookup, Metric};
use rbp_core;
use rbp_database::Schema;
use rbp_gameplay::Abstraction;
use rbp_nlhe::NlheProfile;
use tokio_postgres::Client;

use crate::EpochMeta;

/// Creates all trainer tables if they don't already exist.
///
/// Safe to call on every startup — all DDL uses CREATE TABLE IF NOT EXISTS.
/// For Street and Abstraction whose creates() includes TRUNCATE, creation is
/// guarded by an existence check so existing data is never wiped.
pub async fn ensure_schema(client: &Client) {
    log::info!("{:<32}", "ensuring schema");
    let safe = [
        <Lookup as Schema>::creates(),
        <Metric as Schema>::creates(),
        <Future as Schema>::creates(),
        <NlheProfile as Schema>::creates(),
        <EpochMeta as Schema>::creates(),
    ]
    .join("\n");
    client.batch_execute(&safe).await.expect("ensure schema");
    ensure_blueprint_players(client).await;
    ensure_epoch_players(client).await;
    ensure_if_absent::<Abstraction>(client).await;
    ensure_if_absent::<Street>(client).await;
}

async fn ensure_blueprint_players(client: &Client) {
    let sql = const_format::concatcp!(
        "ALTER TABLE ",
        rbp_database::BLUEPRINT,
        " ADD COLUMN IF NOT EXISTS players SMALLINT NOT NULL DEFAULT 6;
         ALTER TABLE ",
        rbp_database::BLUEPRINT,
        " DROP CONSTRAINT IF EXISTS blueprint_past_present_choices_edge_key;
         DROP INDEX IF EXISTS idx_blueprint_upsert;
         DROP INDEX IF EXISTS idx_blueprint_bucket;
         CREATE UNIQUE INDEX IF NOT EXISTS idx_blueprint_upsert_players ON ",
        rbp_database::BLUEPRINT,
        " (players, present, past, choices, edge);
         CREATE INDEX IF NOT EXISTS idx_blueprint_bucket_players ON ",
        rbp_database::BLUEPRINT,
        " (players, present, past, choices);"
    );
    client
        .batch_execute(sql)
        .await
        .expect("ensure blueprint players");
}

async fn ensure_epoch_players(client: &Client) {
    for players in rbp_core::MIN_PLAYERS..=rbp_core::MAX_PLAYERS {
        let key = format!("current:{players}");
        client
            .execute(
                const_format::concatcp!(
                    "INSERT INTO ",
                    rbp_database::EPOCH,
                    " (key, value)
                      VALUES ($1, COALESCE((SELECT value FROM ",
                    rbp_database::EPOCH,
                    " WHERE key = 'current' AND $1 = 'current:6'), 0))
                      ON CONFLICT (key) DO NOTHING"
                ),
                &[&key],
            )
            .await
            .expect("ensure player epoch");
    }
}

async fn ensure_if_absent<T: Schema>(client: &Client) {
    let name = T::name();
    let absent = client
        .query(
            &format!("SELECT 1 FROM information_schema.tables WHERE table_name = '{name}'"),
            &[],
        )
        .await
        .map(|rows| rows.is_empty())
        .unwrap_or(true);
    if absent {
        log::info!("{:<32}{:<32}", "creating table", name);
        client
            .batch_execute(T::creates())
            .await
            .expect("ensure schema");
    }
}
