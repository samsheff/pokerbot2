//! Training mode selection from command line arguments.
use crate::*;
use rbp_database::Check;

/// Training mode parsed from command line arguments
pub enum Mode {
    Status,
    Cluster,
    Fast,
    Slow,
    Reset,
}

impl Mode {
    pub fn from_args() -> Self {
        std::env::args()
            .find_map(|a| match a.as_str() {
                "--cluster" => Some(Self::Cluster),
                "--status" => Some(Self::Status),
                "--fast" => Some(Self::Fast),
                "--slow" => Some(Self::Slow),
                "--reset" => Some(Self::Reset),
                _ => None,
            })
            .unwrap_or_else(|| {
                eprintln!(
                    "Usage: trainer --status | --cluster | --fast | --slow | --reset [--players N]"
                );
                std::process::exit(1);
            })
    }

    pub async fn run() {
        let client = rbp_database::db().await;
        ensure_schema(&client).await;
        let players = Self::players_from_args();
        match (Self::from_args(), players) {
            (Self::Fast, 2) => FastSession::<2>::new(client).await.train().await,
            (Self::Fast, 3) => FastSession::<3>::new(client).await.train().await,
            (Self::Fast, 4) => FastSession::<4>::new(client).await.train().await,
            (Self::Fast, 5) => FastSession::<5>::new(client).await.train().await,
            (Self::Fast, _) => FastSession::<6>::new(client).await.train().await,
            (Self::Slow, 2) => SlowSession::<2>::new(client).await.train().await,
            (Self::Slow, 3) => SlowSession::<3>::new(client).await.train().await,
            (Self::Slow, 4) => SlowSession::<4>::new(client).await.train().await,
            (Self::Slow, 5) => SlowSession::<5>::new(client).await.train().await,
            (Self::Slow, _) => SlowSession::<6>::new(client).await.train().await,
            (Self::Reset, p) => Self::reset(&client, p).await,
            (Self::Status, _) => client.status().await,
            (Self::Cluster, _) => PreTraining::run(&client).await,
        }
    }
    fn players_from_args() -> usize {
        let mut args = std::env::args();
        while let Some(arg) = args.next() {
            if arg == "--players" {
                let raw = args.next().unwrap_or_else(|| {
                    eprintln!("--players requires a value");
                    std::process::exit(1);
                });
                let players = raw.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("--players must be an integer");
                    std::process::exit(1);
                });
                return rbp_core::validate_players(players).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                });
            }
        }
        rbp_core::N
    }
    async fn reset(client: &tokio_postgres::Client, players: usize) {
        log::info!("Truncating blueprint rows for {} players...", players);
        client
            .execute(
                const_format::concatcp!(
                    "DELETE FROM ",
                    rbp_database::BLUEPRINT,
                    " WHERE players = $1"
                ),
                &[&(players as i16)],
            )
            .await
            .expect("truncate blueprint");
        log::info!("Resetting epoch counter...");
        let key = format!("current:{players}");
        client
            .execute(
                const_format::concatcp!(
                    "UPDATE ",
                    rbp_database::EPOCH,
                    " SET value = 0 WHERE key = $1"
                ),
                &[&key],
            )
            .await
            .expect("reset epoch");
        log::info!("Reset complete.");
    }
}
