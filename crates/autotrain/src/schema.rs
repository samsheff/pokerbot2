//! Database schema initialization for the trainer binary.
use rbp_cards::Street;
use rbp_clustering::{Future, Lookup, Metric};
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
    ensure_if_absent::<Abstraction>(client).await;
    ensure_if_absent::<Street>(client).await;
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
        client.batch_execute(T::creates()).await.expect("ensure schema");
    }
}
