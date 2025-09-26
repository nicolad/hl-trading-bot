use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use futures::{StreamExt, stream};
use hyperliquid_bot::info_snapshot::{
    HyperliquidUserSnapshot, SnapshotRequest, fetch_user_snapshot,
};
use hyperliquid_bot::leaderboard::{InfoApiClient, ReqwestInfoClient, WalletResult};
use serde::{Deserialize, Serialize};

const API_URL: &str = "https://api.hyperliquid.xyz/info";
const DEFAULT_INPUT: &str = "leaderboard.json";
const DEFAULT_OUTPUT: &str = "snapshots.json";
const CONCURRENCY: usize = 2;
const AGGREGATE_FILLS: bool = true;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WalletSnapshot {
    wallet: WalletResult,
    snapshot: HyperliquidUserSnapshot,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FailedWallet {
    wallet: WalletResult,
    error: String,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
struct SnapshotReport {
    snapshots: Vec<WalletSnapshot>,
    failures: Vec<FailedWallet>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let stdin = io::stdin();
    let is_tty = stdin.is_terminal();
    let (bytes, source_label, output_path) = if is_tty {
        let input_path = args
            .next()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_INPUT));
        let output = args
            .next()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT));
        let data = fs::read(&input_path)
            .with_context(|| format!("failed to read {}", input_path.display()))?;
        (data, input_path.display().to_string(), output)
    } else {
        let mut buffer = Vec::new();
        stdin.lock().read_to_end(&mut buffer)?;
        let output = args
            .next()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT));
        (buffer, String::from("stdin"), output)
    };
    let mut input_report: Option<SnapshotReport> = None;
    let wallets: Vec<WalletResult> = match serde_json::from_slice::<Vec<WalletResult>>(&bytes) {
        Ok(list) => list,
        Err(_) => {
            let report: SnapshotReport = serde_json::from_slice(&bytes)
                .with_context(|| format!("failed to parse {}", source_label))?;
            let pending = report
                .failures
                .iter()
                .map(|item| item.wallet.clone())
                .collect();
            input_report = Some(report);
            pending
        }
    };
    let client: Arc<dyn InfoApiClient> = Arc::new(ReqwestInfoClient::new(API_URL)?);
    let mut report = if output_path.exists() {
        let existing_bytes = fs::read(&output_path)
            .with_context(|| format!("failed to read {}", output_path.display()))?;
        match serde_json::from_slice::<SnapshotReport>(&existing_bytes) {
            Ok(report) => report,
            Err(_) => {
                let snapshots: Vec<WalletSnapshot> = serde_json::from_slice(&existing_bytes)
                    .with_context(|| {
                        format!(
                            "failed to parse legacy snapshot format {}",
                            output_path.display()
                        )
                    })?;
                SnapshotReport {
                    snapshots,
                    failures: Vec::new(),
                }
            }
        }
    } else if let Some(report) = input_report.clone() {
        report
    } else {
        SnapshotReport::default()
    };
    let mut existing_snapshots: HashMap<String, WalletSnapshot> = report
        .snapshots
        .drain(..)
        .map(|item| (item.wallet.address.clone(), item))
        .collect();
    let mut outstanding_failures: HashMap<String, FailedWallet> = report
        .failures
        .drain(..)
        .map(|item| (item.wallet.address.clone(), item))
        .collect();
    let mut retained: Vec<WalletSnapshot> = Vec::new();
    let mut pending: Vec<WalletResult> = Vec::new();
    let mut seen_addresses = HashSet::new();
    for wallet in wallets.into_iter() {
        if !seen_addresses.insert(wallet.address.clone()) {
            continue;
        }
        if let Some(mut snapshot) = existing_snapshots.remove(&wallet.address) {
            outstanding_failures.remove(&wallet.address);
            snapshot.wallet = wallet;
            retained.push(snapshot);
        } else {
            pending.push(wallet);
        }
    }
    let total_pending = pending.len();
    if total_pending == 0 {
        let mut remaining_failures: Vec<FailedWallet> =
            outstanding_failures.into_values().collect();
        remaining_failures.sort_by(|a, b| a.wallet.rank.cmp(&b.wallet.rank));
        let final_report = SnapshotReport {
            snapshots: retained,
            failures: remaining_failures.clone(),
        };
        let rendered = serde_json::to_string_pretty(&final_report)?;
        fs::write(&output_path, rendered.as_bytes())
            .with_context(|| format!("failed to write {}", output_path.display()))?;
        println!(
            "wrote {} (0 fresh, {} reused, {} failed)",
            output_path.display(),
            final_report.snapshots.len(),
            final_report.failures.len()
        );
        if final_report.failures.is_empty() {
            return Ok(());
        }
        for item in &final_report.failures {
            eprintln!(
                "snapshot pending for {}: {}",
                item.wallet.address, item.error
            );
        }
        return Err(anyhow!(
            "{} wallet snapshots still pending (no new attempts)",
            final_report.failures.len()
        ));
    }
    let mut processed = 0usize;
    let mut fresh: Vec<WalletSnapshot> = Vec::new();
    let mut failures: Vec<FailedWallet> = Vec::new();
    let mut stream = stream::iter(pending.into_iter())
        .map(|wallet| {
            let client = client.clone();
            async move {
                let request = SnapshotRequest {
                    address: wallet.address.clone(),
                    dex: None,
                    aggregate_fills: AGGREGATE_FILLS,
                    fills_window: None,
                };
                match fetch_user_snapshot(client, request).await {
                    Ok(snapshot) => Ok(WalletSnapshot { wallet, snapshot }),
                    Err(err) => Err((wallet, err)),
                }
            }
        })
        .buffer_unordered(CONCURRENCY.max(1));
    while let Some(result) = stream.next().await {
        processed += 1;
        match result {
            Ok(value) => {
                outstanding_failures.remove(&value.wallet.address);
                println!(
                    "[{}/{}] {} ok",
                    processed, total_pending, value.wallet.address
                );
                fresh.push(value);
            }
            Err((wallet, err)) => {
                println!(
                    "[{}/{}] {} failed: {}",
                    processed, total_pending, wallet.address, err
                );
                failures.push(FailedWallet {
                    wallet,
                    error: err.to_string(),
                });
            }
        }
    }
    let reused_count = retained.len();
    let fresh_count = fresh.len();
    let mut combined: HashMap<String, WalletSnapshot> = HashMap::new();
    for snapshot in retained.into_iter().chain(fresh.into_iter()) {
        combined.insert(snapshot.wallet.address.clone(), snapshot);
    }
    let mut final_snapshots: Vec<WalletSnapshot> = combined.into_values().collect();
    final_snapshots.sort_by(|a, b| a.wallet.rank.cmp(&b.wallet.rank));
    let mut failure_map: HashMap<String, FailedWallet> = outstanding_failures;
    for item in failures.into_iter() {
        failure_map.insert(item.wallet.address.clone(), item);
    }
    let mut final_failures: Vec<FailedWallet> = failure_map.into_values().collect();
    final_failures.sort_by(|a, b| a.wallet.rank.cmp(&b.wallet.rank));
    let final_report = SnapshotReport {
        snapshots: final_snapshots,
        failures: final_failures.clone(),
    };
    let rendered = serde_json::to_string_pretty(&final_report)?;
    fs::write(&output_path, rendered.as_bytes())
        .with_context(|| format!("failed to write {}", output_path.display()))?;
    println!(
        "wrote {} ({} fresh, {} reused, {} failed)",
        output_path.display(),
        fresh_count,
        reused_count,
        final_failures.len()
    );
    if final_failures.is_empty() {
        Ok(())
    } else {
        for item in &final_failures {
            eprintln!(
                "snapshot failed for {}: {}",
                item.wallet.address, item.error
            );
        }
        Err(anyhow!(
            "{} wallet snapshots failed (partial results saved)",
            final_failures.len()
        ))
    }
}
