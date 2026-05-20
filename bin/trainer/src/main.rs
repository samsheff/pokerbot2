//! Autotrain Binary
//!
//! Unified training pipeline with postgres as source of truth.
//!
//! Options: --status, --fast, --slow, --cluster, --reset, --stability

#[tokio::main]
async fn main() {
    load_dotenv();
    rbp_core::log();
    rbp_core::kys();
    rbp_core::brb();
    rbp_autotrain::Mode::run().await;
}

fn load_dotenv() {
    let Ok(contents) = std::fs::read_to_string(".env") else {
        return;
    };
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if std::env::var_os(key).is_some() {
            continue;
        }
        let value = value.trim_matches('"').trim_matches('\'');
        unsafe {
            std::env::set_var(key.trim(), value);
        }
    }
}
