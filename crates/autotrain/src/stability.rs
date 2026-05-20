//! Policy stability snapshots for deciding training stop epochs.
use rbp_database::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use tokio_postgres::Client;

const DEFAULT_SAMPLE_SIZE: usize = 1024;
const DEFAULT_OUTPUT_DIR: &str = "analysis/policy-stability";
const POLICY_MIN: f32 = 1e-6;
const STOP_MEAN_JSD: f32 = 0.01;
const STOP_P95_JSD: f32 = 0.05;
const STOP_MEAN_MAX_DELTA: f32 = 0.025;

/// Policy stability measurement configuration.
pub struct Stability {
    players: usize,
    sample_size: usize,
    output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct InfoKey {
    past: i64,
    present: i16,
    choices: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PolicyEntry {
    info: InfoKey,
    policy: BTreeMap<i64, f32>,
    counts: BTreeMap<i64, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StabilityMetrics {
    previous_epoch: usize,
    compared_infos: usize,
    mean_jsd: f32,
    p95_jsd: f32,
    mean_max_delta: f32,
    stable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StabilitySnapshot {
    players: usize,
    epoch: usize,
    sample_size: usize,
    generated_at_unix: u64,
    stop_thresholds: StopThresholds,
    metrics_vs_previous: Option<StabilityMetrics>,
    entries: Vec<PolicyEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StopThresholds {
    mean_jsd: f32,
    p95_jsd: f32,
    mean_max_delta: f32,
}

impl Default for StopThresholds {
    fn default() -> Self {
        Self {
            mean_jsd: STOP_MEAN_JSD,
            p95_jsd: STOP_P95_JSD,
            mean_max_delta: STOP_MEAN_MAX_DELTA,
        }
    }
}

impl Stability {
    pub fn from_args(players: usize) -> Self {
        let mut args = std::env::args();
        let mut sample_size = DEFAULT_SAMPLE_SIZE;
        let mut output_dir = PathBuf::from(DEFAULT_OUTPUT_DIR);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--sample-size" => {
                    let raw = args.next().unwrap_or_else(|| {
                        eprintln!("--sample-size requires a value");
                        std::process::exit(1);
                    });
                    sample_size = raw.parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("--sample-size must be an integer");
                        std::process::exit(1);
                    });
                }
                "--stability-dir" => {
                    output_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
                        eprintln!("--stability-dir requires a value");
                        std::process::exit(1);
                    });
                }
                _ => {}
            }
        }
        Self {
            players,
            sample_size,
            output_dir,
        }
    }

    pub async fn run(&self, client: &Client) {
        std::fs::create_dir_all(&self.output_dir).expect("create stability output dir");
        let previous = latest_snapshot(&self.output_dir, self.players);
        let keys = previous
            .as_ref()
            .map(|snapshot| {
                snapshot
                    .entries
                    .iter()
                    .map(|entry| entry.info.clone())
                    .collect()
            })
            .unwrap_or_else(Vec::new);
        let entries = if keys.is_empty() {
            self.sample_entries(client).await
        } else {
            self.entries_for_keys(client, &keys).await
        };
        if entries.is_empty() {
            log::warn!(
                "no blueprint policies found for {} players; train this table size first",
                self.players
            );
            return;
        }
        let epoch = client.epochs_for(self.players).await;
        let metrics = previous
            .as_ref()
            .map(|snapshot| compare(snapshot, &entries));
        let snapshot = StabilitySnapshot {
            players: self.players,
            epoch,
            sample_size: entries.len(),
            generated_at_unix: now_unix(),
            stop_thresholds: StopThresholds::default(),
            metrics_vs_previous: metrics.clone(),
            entries,
        };
        let path = self
            .output_dir
            .join(format!("p{}-epoch-{}.json", self.players, epoch));
        let json = serde_json::to_string_pretty(&snapshot).expect("serialize stability snapshot");
        std::fs::write(&path, json).expect("write stability snapshot");
        log::info!(
            "wrote policy stability snapshot for P{} epoch {} to {}",
            self.players,
            epoch,
            path.display()
        );
        match metrics {
            Some(m) => log::info!(
                "compared with epoch {}: infos {}, mean_jsd {:.6}, p95_jsd {:.6}, mean_max_delta {:.6}, stable {}",
                m.previous_epoch,
                m.compared_infos,
                m.mean_jsd,
                m.p95_jsd,
                m.mean_max_delta,
                m.stable
            ),
            None => log::info!(
                "no prior P{} snapshot found; this is the baseline",
                self.players
            ),
        }
    }

    async fn sample_entries(&self, client: &Client) -> Vec<PolicyEntry> {
        let sql = format!(
            "WITH infos AS (
                SELECT past, present, choices
                FROM   {blueprint}
                WHERE  players = $1
                GROUP  BY past, present, choices
                ORDER  BY md5(past::TEXT || ':' || present::TEXT || ':' || choices::TEXT)
                LIMIT  $2
             )
             SELECT b.past, b.present, b.choices, b.edge, b.weight, b.counts
             FROM   {blueprint} b
             JOIN   infos i
             ON     b.past = i.past
             AND    b.present = i.present
             AND    b.choices = i.choices
             WHERE  b.players = $1
             ORDER  BY b.past, b.present, b.choices, b.edge",
            blueprint = BLUEPRINT
        );
        rows_to_entries(
            client
                .query(&sql, &[&(self.players as i16), &(self.sample_size as i64)])
                .await
                .expect("sample policy stability entries"),
        )
    }

    async fn entries_for_keys(&self, client: &Client, keys: &[InfoKey]) -> Vec<PolicyEntry> {
        let past = keys.iter().map(|key| key.past).collect::<Vec<_>>();
        let present = keys.iter().map(|key| key.present).collect::<Vec<_>>();
        let choices = keys.iter().map(|key| key.choices).collect::<Vec<_>>();
        let sql = format!(
            "WITH keys AS (
                SELECT * FROM UNNEST($2::BIGINT[], $3::SMALLINT[], $4::BIGINT[])
                AS t(past, present, choices)
             )
             SELECT b.past, b.present, b.choices, b.edge, b.weight, b.counts
             FROM   {blueprint} b
             JOIN   keys k
             ON     b.past = k.past
             AND    b.present = k.present
             AND    b.choices = k.choices
             WHERE  b.players = $1
             ORDER  BY b.past, b.present, b.choices, b.edge",
            blueprint = BLUEPRINT
        );
        rows_to_entries(
            client
                .query(&sql, &[&(self.players as i16), &past, &present, &choices])
                .await
                .expect("load policy stability entries"),
        )
    }
}

fn rows_to_entries(rows: Vec<tokio_postgres::Row>) -> Vec<PolicyEntry> {
    let mut grouped: BTreeMap<InfoKey, Vec<(i64, f32, u32)>> = BTreeMap::new();
    for row in rows {
        let key = InfoKey {
            past: row.get::<_, i64>(0),
            present: row.get::<_, i16>(1),
            choices: row.get::<_, i64>(2),
        };
        grouped.entry(key).or_default().push((
            row.get::<_, i64>(3),
            row.get::<_, f32>(4),
            row.get::<_, i32>(5) as u32,
        ));
    }
    grouped
        .into_iter()
        .map(|(info, rows)| {
            let denom = rows
                .iter()
                .map(|(_, weight, _)| weight.max(POLICY_MIN))
                .sum::<f32>();
            let policy = rows
                .iter()
                .map(|(edge, weight, _)| (*edge, weight.max(POLICY_MIN) / denom))
                .collect();
            let counts = rows
                .into_iter()
                .map(|(edge, _, counts)| (edge, counts))
                .collect();
            PolicyEntry {
                info,
                policy,
                counts,
            }
        })
        .collect()
}

fn latest_snapshot(dir: &Path, players: usize) -> Option<StabilitySnapshot> {
    let prefix = format!("p{}-epoch-", players);
    let mut snapshots = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if !name.starts_with(&prefix) || !name.ends_with(".json") {
                return None;
            }
            let contents = std::fs::read_to_string(path).ok()?;
            serde_json::from_str::<StabilitySnapshot>(&contents).ok()
        })
        .collect::<Vec<_>>();
    snapshots.sort_by_key(|snapshot| snapshot.epoch);
    snapshots.pop()
}

fn compare(previous: &StabilitySnapshot, current: &[PolicyEntry]) -> StabilityMetrics {
    let previous_epoch = previous.epoch;
    let previous_entries = previous
        .entries
        .iter()
        .map(|entry| (&entry.info, entry))
        .collect::<BTreeMap<_, _>>();
    let mut jsds = Vec::new();
    let mut max_deltas = Vec::new();
    for curr in current {
        let Some(prev) = previous_entries.get(&curr.info) else {
            continue;
        };
        jsds.push(jsd(&prev.policy, &curr.policy));
        max_deltas.push(max_delta(&prev.policy, &curr.policy));
    }
    jsds.sort_by(f32::total_cmp);
    let mean_jsd = mean(&jsds);
    let p95_jsd = percentile(&jsds, 0.95);
    let mean_max_delta = mean(&max_deltas);
    StabilityMetrics {
        previous_epoch,
        compared_infos: jsds.len(),
        mean_jsd,
        p95_jsd,
        mean_max_delta,
        stable: mean_jsd <= STOP_MEAN_JSD
            && p95_jsd <= STOP_P95_JSD
            && mean_max_delta <= STOP_MEAN_MAX_DELTA,
    }
}

fn jsd(left: &BTreeMap<i64, f32>, right: &BTreeMap<i64, f32>) -> f32 {
    support(left, right)
        .iter()
        .map(|edge| {
            let p = left.get(edge).copied().unwrap_or_default();
            let q = right.get(edge).copied().unwrap_or_default();
            let m = (p + q) * 0.5;
            let left_kl = if p <= 0.0 {
                0.0
            } else {
                p * (p / m.max(f32::MIN_POSITIVE)).ln()
            };
            let right_kl = if q <= 0.0 {
                0.0
            } else {
                q * (q / m.max(f32::MIN_POSITIVE)).ln()
            };
            0.5 * (left_kl + right_kl)
        })
        .sum()
}

fn max_delta(left: &BTreeMap<i64, f32>, right: &BTreeMap<i64, f32>) -> f32 {
    support(left, right)
        .iter()
        .map(|edge| {
            let a = left.get(edge).copied().unwrap_or_default();
            let b = right.get(edge).copied().unwrap_or_default();
            (a - b).abs()
        })
        .fold(0.0, f32::max)
}

fn support(left: &BTreeMap<i64, f32>, right: &BTreeMap<i64, f32>) -> BTreeSet<i64> {
    left.keys().chain(right.keys()).copied().collect()
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn percentile(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f32 * p).round() as usize;
    values[idx]
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_secs()
}
