//! Unified Backend Binary
//!
//! Combines analysis API and live game hosting into a single server.
//! Runs on BIND_ADDR (e.g. 0.0.0.0:8888).

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Maximum table size to load inference blueprints for.
    #[arg(long, default_value_t = rbp_core::MAX_PLAYERS, value_parser = parse_players)]
    max_inference_players: usize,
}

fn parse_players(raw: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|_| "max-inference-players must be an integer".to_string())
        .and_then(rbp_core::validate_players)
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    rbp_core::log();
    rbp_core::kys();
    rbp_core::brb();
    rbp_server::run_with_options(rbp_server::ServerOptions {
        max_inference_players: args.max_inference_players,
    })
    .await
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_max_inference_players() {
        let args = Args::try_parse_from(["backend"]).unwrap();
        assert_eq!(args.max_inference_players, rbp_core::MAX_PLAYERS);
    }

    #[test]
    fn parses_custom_max_inference_players() {
        let args = Args::try_parse_from(["backend", "--max-inference-players", "3"]).unwrap();
        assert_eq!(args.max_inference_players, 3);
    }

    #[test]
    fn rejects_invalid_max_inference_players() {
        let err = Args::try_parse_from(["backend", "--max-inference-players", "1"]).unwrap_err();
        assert!(err.to_string().contains("table_size must be between"));
    }
}
