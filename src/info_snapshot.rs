use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail, ensure};
use futures::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Semaphore;

use ndarray::Array2;

use crate::leaderboard::InfoApiClient;
use linfa::prelude::{Fit, Predict};
use linfa_clustering::KMeans;

/// Configuration for bulk snapshot fetching operations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BulkSnapshotConfig {
    pub concurrency: usize,
    pub retry_count: usize,
    pub retry_delay_ms: u64,
    pub timeout_seconds: u64,
    pub batch_size: usize,
}

impl Default for BulkSnapshotConfig {
    fn default() -> Self {
        Self {
            concurrency: 16,
            retry_count: 3,
            retry_delay_ms: 500,
            timeout_seconds: 30,
            batch_size: 100,
        }
    }
}

/// Output format options for Polars-ready files
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolarsOutputConfig {
    pub ndjson_path: Option<String>,
    pub features_csv_path: Option<String>,
    pub snapshots_json_path: Option<String>,
    pub include_metadata: bool,
}

impl Default for PolarsOutputConfig {
    fn default() -> Self {
        Self {
            ndjson_path: Some("snapshots.jsonl".to_string()),
            features_csv_path: Some("features.csv".to_string()),
            snapshots_json_path: None,
            include_metadata: true,
        }
    }
}

/// Result of bulk snapshot operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BulkSnapshotResult {
    pub successful: usize,
    pub failed: usize,
    pub total_addresses: usize,
    pub failed_addresses: Vec<String>,
    pub elapsed_seconds: f64,
}

/// Leaderboard entry structure for parsing input JSON
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub rank: Option<u64>,
    pub address: String,
    pub realized_pnl: Option<f64>,
    pub net_pnl: Option<f64>,
    #[serde(default)]
    pub breakdown: Option<Value>,
    #[serde(default)]
    pub leaderboard_stat: Option<LeaderboardStat>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeaderboardStat {
    #[serde(rename = "accountValue")]
    pub account_value: Option<String>,
    #[serde(rename = "displayName")]
    pub display_name: Option<String>,
    #[serde(rename = "ethAddress")]
    pub eth_address: Option<String>,
    pub prize: Option<u64>,
    #[serde(rename = "windowPerformances", default)]
    pub window_performances: Vec<(String, WindowPerformance)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowPerformance {
    pub pnl: String,
    pub roi: String,
    pub vlm: String,
}

/// Snapshot with computed features for analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotRecord {
    pub snapshot: HyperliquidUserSnapshot,
    pub features: SnapshotFeatures,
    #[serde(rename = "leaderboard", alias = "leaderboard_data", default)]
    pub leaderboard_data: Option<LeaderboardEntry>,
}

/// Extracted features from a snapshot for ML/analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotFeatures {
    pub address: String,
    pub fills_count: usize,
    pub open_orders_count: usize,
    pub frontend_orders_count: usize,
    pub historical_orders_count: usize,
    pub twap_fills_count: usize,
    pub fills_window_count: Option<usize>,
    pub positions_count: usize,
    pub total_volume: f64,
    pub total_fees: f64,
    pub unique_coins: usize,
    pub avg_fill_size: f64,
    pub largest_fill: f64,
    pub activity_span_hours: f64,
    pub fills_with_timestamp_count: usize,
    pub account_value: Option<f64>,
    pub realized_pnl: Option<f64>,
    pub unrealized_pnl: Option<f64>,
    pub leverage: Option<f64>,
    pub win_rate: Option<f64>,
}

/// Main snapshot fetching function with bulk operations
pub async fn fetch_snapshots_from_leaderboard(
    client: Arc<dyn InfoApiClient>,
    leaderboard_json: &str,
    config: BulkSnapshotConfig,
    output: PolarsOutputConfig,
) -> Result<(BulkSnapshotResult, Vec<SnapshotRecord>)> {
    let start_time = std::time::Instant::now();

    // Parse leaderboard data
    let entries: Vec<LeaderboardEntry> =
        serde_json::from_str(leaderboard_json).context("Failed to parse leaderboard JSON")?;

    if entries.is_empty() {
        bail!("No addresses found in leaderboard data");
    }

    // Extract unique addresses
    let mut addresses: Vec<String> = entries
        .iter()
        .map(|e| e.address.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    addresses.sort();
    let original_total = addresses.len();

    // Load previously fetched snapshots (if any) so we can skip successful ones.
    let mut existing_records = Vec::new();
    let mut completed_addresses: HashSet<String> = HashSet::new();
    if let Some(ref ndjson_path) = output.ndjson_path {
        let path = Path::new(ndjson_path);
        if path.exists() {
            existing_records = load_existing_snapshot_records(ndjson_path)
                .with_context(|| format!("loading existing snapshots from {}", ndjson_path))?;
            for record in &existing_records {
                completed_addresses.insert(record.snapshot.address.clone());
            }
            if !completed_addresses.is_empty() {
                eprintln!(
                    "Skipping {} addresses already fetched",
                    completed_addresses.len()
                );
                addresses.retain(|addr| !completed_addresses.contains(addr));
            }
        }
    }

    let pending_total = addresses.len();
    if pending_total == 0 {
        eprintln!(
            "Found {} unique addresses; nothing to fetch (all cached)",
            original_total
        );
        let elapsed = start_time.elapsed().as_secs_f64();
        write_polars_outputs(&existing_records, &output).await?;
        let result = BulkSnapshotResult {
            successful: existing_records.len(),
            failed: 0,
            total_addresses: original_total,
            failed_addresses: Vec::new(),
            elapsed_seconds: elapsed,
        };
        return Ok((result, existing_records));
    }

    eprintln!(
        "Found {} unique addresses ({} pending fetches)",
        original_total, pending_total
    );

    // Create semaphore for concurrency control
    let semaphore = Arc::new(Semaphore::new(config.concurrency));
    let progress = Arc::new(AtomicUsize::new(0));
    let failures = Arc::new(AtomicUsize::new(0));

    // Build address -> leaderboard entry map for enrichment
    let leaderboard_map: HashMap<String, LeaderboardEntry> = entries
        .into_iter()
        .map(|e| (e.address.clone(), e))
        .collect();

    // Fetch all snapshots concurrently
    let results: Vec<_> = stream::iter(addresses.iter().cloned())
        .map(|address| {
            let client = Arc::clone(&client);
            let semaphore = Arc::clone(&semaphore);
            let leaderboard_entry = leaderboard_map.get(&address).cloned();
            let config = config.clone();
            let progress = Arc::clone(&progress);
            let failures = Arc::clone(&failures);
            let total = pending_total;

            async move {
                let _permit = semaphore.acquire().await.unwrap();

                match fetch_snapshot_with_retries(
                    client,
                    address.clone(),
                    config.retry_count,
                    Duration::from_millis(config.retry_delay_ms),
                )
                .await
                {
                    Ok(snapshot) => {
                        let features = extract_features(&snapshot, leaderboard_entry.as_ref());
                        let record = SnapshotRecord {
                            snapshot,
                            features,
                            leaderboard_data: leaderboard_entry,
                        };
                        let done = progress.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                        let failed = failures.load(AtomicOrdering::Relaxed);
                        eprintln!(
                            "Fetched {}/{}: {} (failures so far: {})",
                            done, total, address, failed
                        );
                        Ok((address, record))
                    }
                    Err(e) => {
                        let done = progress.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                        let failed = failures.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                        eprintln!("Failed {}/{}: {} ({:#})", done, total, address, e);
                        eprintln!("Failures so far: {}", failed);
                        Err((address, e))
                    }
                }
            }
        })
        .buffer_unordered(config.concurrency)
        .collect()
        .await;

    // Separate successes and failures
    let mut snapshots = Vec::new();
    let mut failed_addresses = Vec::new();

    for result in results {
        match result {
            Ok((_address, record)) => {
                snapshots.push(record);
            }
            Err((address, _error)) => {
                failed_addresses.push(address);
            }
        }
    }

    // Merge existing successful snapshots with newly fetched ones
    let mut merged: HashMap<String, SnapshotRecord> = existing_records
        .into_iter()
        .map(|record| (record.snapshot.address.clone(), record))
        .collect();
    for record in snapshots {
        merged.insert(record.snapshot.address.clone(), record);
    }
    let mut combined_records: Vec<SnapshotRecord> = merged.into_values().collect();
    combined_records.sort_by(|a, b| a.snapshot.address.cmp(&b.snapshot.address));

    // Write output files
    write_polars_outputs(&combined_records, &output).await?;

    let elapsed = start_time.elapsed().as_secs_f64();
    let result = BulkSnapshotResult {
        successful: combined_records.len(),
        failed: failed_addresses.len(),
        total_addresses: original_total,
        failed_addresses,
        elapsed_seconds: elapsed,
    };

    eprintln!(
        "Bulk snapshot complete: {} successful, {} failed in {:.1}s",
        result.successful, result.failed, result.elapsed_seconds
    );

    Ok((result, combined_records))
}

/// Fetch a single snapshot with retries and exponential backoff
async fn fetch_snapshot_with_retries(
    client: Arc<dyn InfoApiClient>,
    address: String,
    max_retries: usize,
    base_delay: Duration,
) -> Result<HyperliquidUserSnapshot> {
    let mut last_error = None;

    for attempt in 0..=max_retries {
        let request = SnapshotRequest::new(&address);

        match fetch_user_snapshot(Arc::clone(&client), request).await {
            Ok(snapshot) => return Ok(snapshot),
            Err(e) => {
                last_error = Some(e);

                if attempt < max_retries {
                    let delay = base_delay * (2_u32.pow(attempt as u32));
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("Max retries exceeded for {}", address)))
}

/// Extract analytical features from a snapshot
fn extract_features(
    snapshot: &HyperliquidUserSnapshot,
    leaderboard: Option<&LeaderboardEntry>,
) -> SnapshotFeatures {
    let fills_count = snapshot.fills.len();
    let fills_window_count = snapshot.fills_window.as_ref().map(|w| w.len());

    // Analyze fills for volume, fees, and activity patterns
    let (total_volume, total_fees, unique_coins, largest_fill, activity_span, fills_with_ts) =
        analyze_fills(&snapshot.fills);
    let avg_fill_size = if fills_count > 0 {
        total_volume / fills_count as f64
    } else {
        0.0
    };

    // Count positions from clearinghouse state
    let positions_count = count_positions(&snapshot.clearinghouse_state);

    // Extract account metrics
    let (account_value, unrealized_pnl, leverage) =
        extract_account_metrics(&snapshot.clearinghouse_state);

    // Calculate win rate from fills
    let win_rate = calculate_win_rate(&snapshot.fills);

    // Get realized PnL from leaderboard if available
    let realized_pnl = leaderboard.and_then(|l| l.realized_pnl);

    SnapshotFeatures {
        address: snapshot.address.clone(),
        fills_count,
        open_orders_count: snapshot.open_orders.len(),
        frontend_orders_count: snapshot.frontend_open_orders.len(),
        historical_orders_count: snapshot.historical_orders.len(),
        twap_fills_count: snapshot.twap_slice_fills.len(),
        fills_window_count,
        positions_count,
        total_volume,
        total_fees,
        unique_coins,
        avg_fill_size,
        largest_fill,
        activity_span_hours: activity_span,
        fills_with_timestamp_count: fills_with_ts,
        account_value,
        realized_pnl,
        unrealized_pnl,
        leverage,
        win_rate,
    }
}

/// Analyze fills for volume, fees, and activity metrics
fn analyze_fills(fills: &[Value]) -> (f64, f64, usize, f64, f64, usize) {
    let mut total_volume = 0.0;
    let mut total_fees = 0.0;
    let mut coins = HashSet::new();
    let mut largest_fill = 0.0;
    let mut timestamps = Vec::new();
    let mut fills_with_ts = 0;

    for fill in fills {
        // Volume calculation (px * sz)
        if let (Some(px), Some(sz)) = (
            fill.get("px")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok()),
            fill.get("sz")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok()),
        ) {
            let volume = px * sz;
            total_volume += volume;
            if volume > largest_fill {
                largest_fill = volume;
            }
        }

        // Fees
        if let Some(fee) = fill
            .get("fee")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
        {
            total_fees += fee;
        }

        // Unique coins
        if let Some(coin) = fill.get("coin").and_then(|v| v.as_str()) {
            coins.insert(coin.to_string());
        }

        // Timestamps for activity span
        if let Some(ts) = extract_timestamp_ms(fill) {
            timestamps.push(ts);
            fills_with_ts += 1;
        }
    }

    let activity_span_hours = if timestamps.len() >= 2 {
        timestamps.sort_unstable();
        let span_ms = timestamps.last().unwrap() - timestamps.first().unwrap();
        span_ms as f64 / (1000.0 * 60.0 * 60.0) // Convert to hours
    } else {
        0.0
    };

    (
        total_volume,
        total_fees,
        coins.len(),
        largest_fill,
        activity_span_hours,
        fills_with_ts,
    )
}

/// Extract timestamp from various possible fields in a fill record
fn extract_timestamp_ms(fill: &Value) -> Option<u64> {
    for field in ["time", "timestamp", "ts", "closedPnl"] {
        if let Some(val) = fill.get(field) {
            if let Some(num) = val.as_u64() {
                // Assume milliseconds if > 1e12, seconds otherwise
                return Some(if num > 1_000_000_000_000 {
                    num
                } else {
                    num * 1000
                });
            }
            if let Some(s) = val.as_str() {
                if let Ok(num) = s.parse::<u64>() {
                    return Some(if num > 1_000_000_000_000 {
                        num
                    } else {
                        num * 1000
                    });
                }
            }
        }
    }
    None
}

/// Count positions from clearinghouse state
fn count_positions(clearinghouse_state: &Value) -> usize {
    if let Some(positions) = clearinghouse_state
        .get("assetPositions")
        .and_then(|v| v.as_array())
    {
        positions.len()
    } else if let Some(margin) = clearinghouse_state.get("marginSummary") {
        // Try alternative structure
        margin
            .get("positions")
            .and_then(|v| v.as_array())
            .map(|p| p.len())
            .unwrap_or(0)
    } else {
        0
    }
}

/// Extract account value, unrealized PnL, and leverage from clearinghouse state
fn extract_account_metrics(clearinghouse_state: &Value) -> (Option<f64>, Option<f64>, Option<f64>) {
    let margin_summary = clearinghouse_state.get("marginSummary");

    let account_value = margin_summary
        .and_then(|m| m.get("accountValue"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok());

    let unrealized_pnl = margin_summary
        .and_then(|m| m.get("totalNtlPos"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok());

    let leverage = margin_summary
        .and_then(|m| m.get("totalMarginUsed"))
        .and_then(|used| used.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .and_then(|used| account_value.map(|av| if av > 0.0 { used / av } else { 0.0 }));

    (account_value, unrealized_pnl, leverage)
}

/// Calculate win rate from fills (simple heuristic based on PnL)
fn calculate_win_rate(fills: &[Value]) -> Option<f64> {
    let mut total_trades = 0;
    let mut winning_trades = 0;

    for fill in fills {
        if let Some(pnl) = fill
            .get("closedPnl")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
        {
            total_trades += 1;
            if pnl > 0.0 {
                winning_trades += 1;
            }
        }
    }

    if total_trades > 0 {
        Some(winning_trades as f64 / total_trades as f64)
    } else {
        None
    }
}

/// Write output files in Polars-ready formats
async fn write_polars_outputs(
    snapshots: &[SnapshotRecord],
    config: &PolarsOutputConfig,
) -> Result<()> {
    // Write NDJSON format (best for Polars)
    if let Some(ref ndjson_path) = config.ndjson_path {
        write_snapshots_ndjson(snapshots, ndjson_path, config.include_metadata)?;
        eprintln!(
            "Wrote {} snapshots to NDJSON: {}",
            snapshots.len(),
            ndjson_path
        );
    }

    // Write features CSV (flat table for analysis)
    if let Some(ref csv_path) = config.features_csv_path {
        write_features_csv(snapshots, csv_path)?;
        eprintln!(
            "Wrote {} feature rows to CSV: {}",
            snapshots.len(),
            csv_path
        );
    }

    // Write complete snapshots as JSON array (compatibility)
    if let Some(ref json_path) = config.snapshots_json_path {
        write_snapshots_json(snapshots, json_path)?;
        eprintln!("Wrote {} snapshots to JSON: {}", snapshots.len(), json_path);
    }

    Ok(())
}

/// Write snapshots as NDJSON (one snapshot per line) - optimal for Polars
fn write_snapshots_ndjson(
    snapshots: &[SnapshotRecord],
    path: &str,
    include_metadata: bool,
) -> Result<()> {
    let file = File::create(path).context("Failed to create NDJSON file")?;
    let mut writer = BufWriter::new(file);

    for record in snapshots {
        let line = if include_metadata {
            json!({
                "address": record.snapshot.address,
                "fetched_at_ms": record.snapshot.fetched_at_ms,
                "features": record.features,
                "leaderboard": record.leaderboard_data,
                "snapshot": record.snapshot
            })
        } else {
            json!({
                "address": record.snapshot.address,
                "snapshot": record.snapshot
            })
        };

        serde_json::to_writer(&mut writer, &line)?;
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

/// Write features as CSV for easy analysis
fn write_features_csv(snapshots: &[SnapshotRecord], path: &str) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path).context("Failed to create CSV file")?;

    // Write header
    wtr.write_record(&[
        "address",
        "fills_count",
        "open_orders_count",
        "frontend_orders_count",
        "historical_orders_count",
        "twap_fills_count",
        "fills_window_count",
        "positions_count",
        "total_volume",
        "total_fees",
        "unique_coins",
        "avg_fill_size",
        "largest_fill",
        "activity_span_hours",
        "fills_with_timestamp_count",
        "account_value",
        "realized_pnl",
        "unrealized_pnl",
        "leverage",
        "win_rate",
    ])?;

    // Write data rows
    for record in snapshots {
        let f = &record.features;
        wtr.write_record(&[
            f.address.as_str(),
            &f.fills_count.to_string(),
            &f.open_orders_count.to_string(),
            &f.frontend_orders_count.to_string(),
            &f.historical_orders_count.to_string(),
            &f.twap_fills_count.to_string(),
            &f.fills_window_count
                .map(|c| c.to_string())
                .unwrap_or_default(),
            &f.positions_count.to_string(),
            &f.total_volume.to_string(),
            &f.total_fees.to_string(),
            &f.unique_coins.to_string(),
            &f.avg_fill_size.to_string(),
            &f.largest_fill.to_string(),
            &f.activity_span_hours.to_string(),
            &f.fills_with_timestamp_count.to_string(),
            &f.account_value.map(|v| v.to_string()).unwrap_or_default(),
            &f.realized_pnl.map(|v| v.to_string()).unwrap_or_default(),
            &f.unrealized_pnl.map(|v| v.to_string()).unwrap_or_default(),
            &f.leverage.map(|v| v.to_string()).unwrap_or_default(),
            &f.win_rate.map(|v| v.to_string()).unwrap_or_default(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

/// Write snapshots as JSON array (compatibility with existing loaders)
fn write_snapshots_json(snapshots: &[SnapshotRecord], path: &str) -> Result<()> {
    let file = File::create(path).context("Failed to create JSON file")?;
    let mut writer = BufWriter::new(file);

    let snapshot_objects: Vec<_> = snapshots.iter().map(|e| &e.snapshot).collect();

    serde_json::to_writer_pretty(&mut writer, &snapshot_objects)?;
    writer.flush()?;
    Ok(())
}

fn load_existing_snapshot_records(path: &str) -> Result<Vec<SnapshotRecord>> {
    let file = File::open(path).context("opening existing snapshots NDJSON")?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: SnapshotRecord = serde_json::from_str(&line)
            .with_context(|| "parsing snapshot record from NDJSON line")?;
        records.push(record);
    }
    Ok(records)
}

/// Original fetch_user_snapshot function (preserved for compatibility)
pub async fn fetch_user_snapshot(
    client: Arc<dyn InfoApiClient>,
    request: SnapshotRequest,
) -> Result<HyperliquidUserSnapshot> {
    ensure!(!request.address.is_empty(), "address required");
    if let Some((start, end)) = request.fills_window {
        ensure!(start <= end, "fills window start must be <= end");
    }
    let SnapshotRequest {
        address,
        dex,
        aggregate_fills,
        fills_window,
    } = request;
    let dex_payload = dex.clone().unwrap_or_default();
    let open_orders = expect_array(
        "openOrders",
        client
            .post(json!({
                "type": "openOrders",
                "user": address.clone(),
                "dex": dex_payload,
            }))
            .await?,
    )?;
    let frontend_open_orders = expect_array(
        "frontendOpenOrders",
        client
            .post(json!({
                "type": "frontendOpenOrders",
                "user": address.clone(),
                "dex": dex.clone().unwrap_or_default(),
            }))
            .await?,
    )?;
    let historical_orders = expect_array(
        "historicalOrders",
        client
            .post(json!({"type": "historicalOrders", "user": address.clone()}))
            .await?,
    )?;
    let fills = expect_array(
        "userFills",
        client
            .post(json!({
                "type": "userFills",
                "user": address.clone(),
                "aggregateByTime": aggregate_fills,
            }))
            .await?,
    )?;
    let twap_slice_fills = expect_array(
        "userTwapSliceFills",
        client
            .post(json!({"type": "userTwapSliceFills", "user": address.clone()}))
            .await?,
    )?;
    let clearinghouse_state = expect_object(
        "clearinghouseState",
        client
            .post(json!({
                "type": "clearinghouseState",
                "user": address.clone(),
                "dex": dex.clone().unwrap_or_default(),
            }))
            .await?,
    )?;
    let fills_window_values = match fills_window {
        Some((start, end)) => Some(expect_array(
            "userFillsByTime",
            client
                .post(json!({
                    "type": "userFillsByTime",
                    "user": address.clone(),
                    "startTime": start,
                    "endTime": end,
                    "aggregateByTime": aggregate_fills,
                }))
                .await?,
        )?),
        None => None,
    };
    Ok(HyperliquidUserSnapshot {
        address,
        dex,
        open_orders,
        frontend_open_orders,
        historical_orders,
        fills,
        fills_window: fills_window_values,
        twap_slice_fills,
        clearinghouse_state,
        fetched_at_ms: current_millis(),
    })
}

fn expect_array(name: &str, value: Value) -> Result<Vec<Value>> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => Ok(items),
        other => bail!("{name} expected array, got {other}"),
    }
}

fn expect_object(name: &str, value: Value) -> Result<Value> {
    match value {
        Value::Null => Ok(json!({})),
        Value::Object(_) => Ok(value),
        other => bail!("{name} expected object, got {other}"),
    }
}

fn current_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HyperliquidUserSnapshot {
    pub address: String,
    pub dex: Option<String>,
    pub open_orders: Vec<Value>,
    pub frontend_open_orders: Vec<Value>,
    pub historical_orders: Vec<Value>,
    pub fills: Vec<Value>,
    pub fills_window: Option<Vec<Value>>,
    pub twap_slice_fills: Vec<Value>,
    pub clearinghouse_state: Value,
    pub fetched_at_ms: u64,
}

impl HyperliquidUserSnapshot {
    /// Save the snapshot as NDJSON format to a file
    /// Each array field (fills, orders, etc.) will be saved as separate NDJSON lines
    pub fn save_to_ndjson<P: AsRef<Path>>(&self, file_path: P) -> Result<()> {
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);

        // Save fills as NDJSON lines
        for fill in &self.fills {
            let mut fill_record = fill.clone();
            // Add metadata to each record
            if let Value::Object(ref mut map) = fill_record {
                map.insert("record_type".to_string(), Value::String("fill".to_string()));
                map.insert("address".to_string(), Value::String(self.address.clone()));
                if let Some(ref dex) = self.dex {
                    map.insert("dex".to_string(), Value::String(dex.clone()));
                }
                map.insert(
                    "fetched_at_ms".to_string(),
                    Value::Number(self.fetched_at_ms.into()),
                );
            }
            writeln!(writer, "{}", serde_json::to_string(&fill_record)?)?;
        }

        // Save fills from window if available
        if let Some(ref window_fills) = self.fills_window {
            for fill in window_fills {
                let mut fill_record = fill.clone();
                if let Value::Object(ref mut map) = fill_record {
                    map.insert(
                        "record_type".to_string(),
                        Value::String("window_fill".to_string()),
                    );
                    map.insert("address".to_string(), Value::String(self.address.clone()));
                    if let Some(ref dex) = self.dex {
                        map.insert("dex".to_string(), Value::String(dex.clone()));
                    }
                    map.insert(
                        "fetched_at_ms".to_string(),
                        Value::Number(self.fetched_at_ms.into()),
                    );
                }
                writeln!(writer, "{}", serde_json::to_string(&fill_record)?)?;
            }
        }

        // Save TWAP slice fills
        for fill in &self.twap_slice_fills {
            let mut fill_record = fill.clone();
            if let Value::Object(ref mut map) = fill_record {
                map.insert(
                    "record_type".to_string(),
                    Value::String("twap_fill".to_string()),
                );
                map.insert("address".to_string(), Value::String(self.address.clone()));
                if let Some(ref dex) = self.dex {
                    map.insert("dex".to_string(), Value::String(dex.clone()));
                }
                map.insert(
                    "fetched_at_ms".to_string(),
                    Value::Number(self.fetched_at_ms.into()),
                );
            }
            writeln!(writer, "{}", serde_json::to_string(&fill_record)?)?;
        }

        // Save open orders
        for order in &self.open_orders {
            let mut order_record = order.clone();
            if let Value::Object(ref mut map) = order_record {
                map.insert(
                    "record_type".to_string(),
                    Value::String("open_order".to_string()),
                );
                map.insert("address".to_string(), Value::String(self.address.clone()));
                if let Some(ref dex) = self.dex {
                    map.insert("dex".to_string(), Value::String(dex.clone()));
                }
                map.insert(
                    "fetched_at_ms".to_string(),
                    Value::Number(self.fetched_at_ms.into()),
                );
            }
            writeln!(writer, "{}", serde_json::to_string(&order_record)?)?;
        }

        // Save frontend open orders
        for order in &self.frontend_open_orders {
            let mut order_record = order.clone();
            if let Value::Object(ref mut map) = order_record {
                map.insert(
                    "record_type".to_string(),
                    Value::String("frontend_order".to_string()),
                );
                map.insert("address".to_string(), Value::String(self.address.clone()));
                if let Some(ref dex) = self.dex {
                    map.insert("dex".to_string(), Value::String(dex.clone()));
                }
                map.insert(
                    "fetched_at_ms".to_string(),
                    Value::Number(self.fetched_at_ms.into()),
                );
            }
            writeln!(writer, "{}", serde_json::to_string(&order_record)?)?;
        }

        // Save historical orders
        for order in &self.historical_orders {
            let mut order_record = order.clone();
            if let Value::Object(ref mut map) = order_record {
                map.insert(
                    "record_type".to_string(),
                    Value::String("historical_order".to_string()),
                );
                map.insert("address".to_string(), Value::String(self.address.clone()));
                if let Some(ref dex) = self.dex {
                    map.insert("dex".to_string(), Value::String(dex.clone()));
                }
                map.insert(
                    "fetched_at_ms".to_string(),
                    Value::Number(self.fetched_at_ms.into()),
                );
            }
            writeln!(writer, "{}", serde_json::to_string(&order_record)?)?;
        }

        // Save clearinghouse state as a single record
        let mut clearinghouse_record = self.clearinghouse_state.clone();
        if let Value::Object(ref mut map) = clearinghouse_record {
            map.insert(
                "record_type".to_string(),
                Value::String("clearinghouse_state".to_string()),
            );
            map.insert("address".to_string(), Value::String(self.address.clone()));
            if let Some(ref dex) = self.dex {
                map.insert("dex".to_string(), Value::String(dex.clone()));
            }
            map.insert(
                "fetched_at_ms".to_string(),
                Value::Number(self.fetched_at_ms.into()),
            );
        }
        writeln!(writer, "{}", serde_json::to_string(&clearinghouse_record)?)?;

        writer.flush()?;
        Ok(())
    }

    /// Save only fills to NDJSON format
    pub fn save_fills_to_ndjson<P: AsRef<Path>>(&self, file_path: P) -> Result<()> {
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);

        // Save all fills including window fills and TWAP fills
        let all_fills = [("fill", &self.fills), ("twap_fill", &self.twap_slice_fills)];

        for (fill_type, fills) in all_fills {
            for fill in fills {
                let mut fill_record = fill.clone();
                if let Value::Object(ref mut map) = fill_record {
                    map.insert(
                        "record_type".to_string(),
                        Value::String(fill_type.to_string()),
                    );
                    map.insert("address".to_string(), Value::String(self.address.clone()));
                    if let Some(ref dex) = self.dex {
                        map.insert("dex".to_string(), Value::String(dex.clone()));
                    }
                    map.insert(
                        "fetched_at_ms".to_string(),
                        Value::Number(self.fetched_at_ms.into()),
                    );
                }
                writeln!(writer, "{}", serde_json::to_string(&fill_record)?)?;
            }
        }

        // Add window fills if available
        if let Some(ref window_fills) = self.fills_window {
            for fill in window_fills {
                let mut fill_record = fill.clone();
                if let Value::Object(ref mut map) = fill_record {
                    map.insert(
                        "record_type".to_string(),
                        Value::String("window_fill".to_string()),
                    );
                    map.insert("address".to_string(), Value::String(self.address.clone()));
                    if let Some(ref dex) = self.dex {
                        map.insert("dex".to_string(), Value::String(dex.clone()));
                    }
                    map.insert(
                        "fetched_at_ms".to_string(),
                        Value::Number(self.fetched_at_ms.into()),
                    );
                }
                writeln!(writer, "{}", serde_json::to_string(&fill_record)?)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Save only orders to NDJSON format
    pub fn save_orders_to_ndjson<P: AsRef<Path>>(&self, file_path: P) -> Result<()> {
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);

        let all_orders = [
            ("open_order", &self.open_orders),
            ("frontend_order", &self.frontend_open_orders),
            ("historical_order", &self.historical_orders),
        ];

        for (order_type, orders) in all_orders {
            for order in orders {
                let mut order_record = order.clone();
                if let Value::Object(ref mut map) = order_record {
                    map.insert(
                        "record_type".to_string(),
                        Value::String(order_type.to_string()),
                    );
                    map.insert("address".to_string(), Value::String(self.address.clone()));
                    if let Some(ref dex) = self.dex {
                        map.insert("dex".to_string(), Value::String(dex.clone()));
                    }
                    map.insert(
                        "fetched_at_ms".to_string(),
                        Value::Number(self.fetched_at_ms.into()),
                    );
                }
                writeln!(writer, "{}", serde_json::to_string(&order_record)?)?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SnapshotRequest {
    pub address: String,
    pub dex: Option<String>,
    pub aggregate_fills: bool,
    pub fills_window: Option<(u64, u64)>,
}
impl SnapshotRequest {
    pub fn new(address: impl Into<String>) -> Self {
        Self {
            address: address.into(),
            dex: None,
            aggregate_fills: false,
            fills_window: None,
        }
    }
}

/// Save a user snapshot directly to NDJSON after fetching
pub async fn fetch_and_save_snapshot_ndjson<P: AsRef<Path>>(
    client: Arc<dyn InfoApiClient>,
    request: SnapshotRequest,
    file_path: P,
) -> Result<HyperliquidUserSnapshot> {
    let snapshot = fetch_user_snapshot(client, request).await?;
    snapshot.save_to_ndjson(file_path)?;
    Ok(snapshot)
}

/// Load NDJSON records from file and filter by record type
pub fn load_ndjson_records<P: AsRef<Path>>(
    file_path: P,
    record_type: Option<&str>,
) -> Result<Vec<Value>> {
    use std::io::{BufRead, BufReader};

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let value: Value = serde_json::from_str(&line)?;

        // Filter by record type if specified
        if let Some(expected_type) = record_type {
            if let Some(actual_type) = value.get("record_type").and_then(|v| v.as_str()) {
                if actual_type == expected_type {
                    records.push(value);
                }
            }
        } else {
            records.push(value);
        }
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_ndjson_save_and_load() {
        let snapshot = HyperliquidUserSnapshot {
            address: "0x123".to_string(),
            dex: Some("perp0".to_string()),
            open_orders: vec![
                json!({"oid": 1, "coin": "BTC", "side": "B", "limitPx": "50000"}),
                json!({"oid": 2, "coin": "ETH", "side": "A", "limitPx": "3000"}),
            ],
            frontend_open_orders: vec![],
            historical_orders: vec![
                json!({"oid": 3, "coin": "BTC", "side": "A", "limitPx": "49000", "status": "filled"}),
            ],
            fills: vec![
                json!({"tid": 101, "coin": "BTC", "side": "B", "px": "49500", "sz": "0.1"}),
                json!({"tid": 102, "coin": "ETH", "side": "A", "px": "2950", "sz": "1.0"}),
            ],
            fills_window: None,
            twap_slice_fills: vec![],
            clearinghouse_state: json!({"marginSummary": {"accountValue": "10000"}}),
            fetched_at_ms: 1234567890,
        };

        // Test saving to NDJSON
        let temp_file = NamedTempFile::new().unwrap();
        snapshot.save_to_ndjson(temp_file.path()).unwrap();

        // Test loading back
        let records = load_ndjson_records(temp_file.path(), None).unwrap();
        assert!(!records.is_empty());

        // Check that records have correct metadata
        let fill_records: Vec<_> = records
            .iter()
            .filter(|r| r.get("record_type").and_then(|v| v.as_str()) == Some("fill"))
            .collect();
        assert_eq!(fill_records.len(), 2);

        // Verify first fill has correct metadata
        let first_fill = &fill_records[0];
        assert_eq!(
            first_fill.get("address").and_then(|v| v.as_str()),
            Some("0x123")
        );
        assert_eq!(
            first_fill.get("dex").and_then(|v| v.as_str()),
            Some("perp0")
        );
        assert_eq!(
            first_fill.get("fetched_at_ms").and_then(|v| v.as_u64()),
            Some(1234567890)
        );
    }

    #[test]
    fn test_fills_only_ndjson() {
        let snapshot = HyperliquidUserSnapshot {
            address: "0x456".to_string(),
            dex: None,
            open_orders: vec![json!({"oid": 1})],
            frontend_open_orders: vec![],
            historical_orders: vec![],
            fills: vec![json!({"tid": 201, "coin": "BTC", "px": "50000"})],
            fills_window: Some(vec![json!({"tid": 202, "coin": "ETH", "px": "3000"})]),
            twap_slice_fills: vec![json!({"tid": 203, "coin": "SOL", "px": "100"})],
            clearinghouse_state: json!({}),
            fetched_at_ms: 1234567890,
        };

        let temp_file = NamedTempFile::new().unwrap();
        snapshot.save_fills_to_ndjson(temp_file.path()).unwrap();

        let records = load_ndjson_records(temp_file.path(), None).unwrap();
        assert_eq!(records.len(), 3); // 1 fill + 1 window_fill + 1 twap_fill

        // Verify we only have fill-related records
        let record_types: std::collections::HashSet<_> = records
            .iter()
            .filter_map(|r| r.get("record_type").and_then(|v| v.as_str()))
            .collect();

        let expected_types: std::collections::HashSet<_> = ["fill", "window_fill", "twap_fill"]
            .iter()
            .cloned()
            .collect();

        assert_eq!(record_types, expected_types);
    }

    #[test]
    fn test_load_filtered_records() {
        let snapshot = HyperliquidUserSnapshot {
            address: "0x789".to_string(),
            dex: Some("spot".to_string()),
            open_orders: vec![json!({"oid": 1})],
            frontend_open_orders: vec![],
            historical_orders: vec![],
            fills: vec![json!({"tid": 301})],
            fills_window: None,
            twap_slice_fills: vec![],
            clearinghouse_state: json!({}),
            fetched_at_ms: 1234567890,
        };

        let temp_file = NamedTempFile::new().unwrap();
        snapshot.save_to_ndjson(temp_file.path()).unwrap();

        // Load only fill records
        let fill_records = load_ndjson_records(temp_file.path(), Some("fill")).unwrap();
        assert_eq!(fill_records.len(), 1);

        // Load only order records
        let order_records = load_ndjson_records(temp_file.path(), Some("open_order")).unwrap();
        assert_eq!(order_records.len(), 1);

        // Load non-existent type
        let empty_records = load_ndjson_records(temp_file.path(), Some("nonexistent")).unwrap();
        assert_eq!(empty_records.len(), 0);
    }
}

/// Strategy Mining Integration
/// Integrated pattern analysis and trading signal generation capabilities

/// Configuration for strategy mining operations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyMiningConfig {
    pub bar_interval_ms: i64,
    pub window_bars: usize,
    pub horizon_bars: usize,
    pub k_clusters: usize,
    pub min_fills_per_coin: usize,
    pub min_edge_threshold: f64,
    pub min_win_rate: f64,
    pub output_patterns_json: Option<String>,
    pub output_rules_csv: Option<String>,
}

impl Default for StrategyMiningConfig {
    fn default() -> Self {
        Self {
            bar_interval_ms: 60_000, // 1 minute bars
            window_bars: 20,
            horizon_bars: 5,
            k_clusters: 6,
            min_fills_per_coin: 30,
            min_edge_threshold: 0.001, // 0.1% edge
            min_win_rate: 0.52,
            output_patterns_json: None,
            output_rules_csv: None,
        }
    }
}

/// Trading pattern extracted from clustering analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradingPattern {
    pub pattern_id: usize,
    pub coin: String,
    pub cluster_id: usize,
    pub sample_count: usize,
    pub mean_forward_return: f64,
    pub median_forward_return: f64,
    pub win_rate: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub confidence_score: f64,
}

/// Generated trading signal/rule
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradingSignal {
    pub signal_id: String,
    pub coin: String,
    pub pattern_id: usize,
    pub action: String, // "LONG" or "SHORT"
    pub confidence: f64,
    pub expected_return: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub hold_bars: usize,
    pub min_volume_filter: f64,
    pub created_at: u64,
}

/// Result of strategy mining operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyMiningResult {
    pub patterns: Vec<TradingPattern>,
    pub signals: Vec<TradingSignal>,
    pub total_snapshots: usize,
    pub analyzed_coins: Vec<String>,
    pub mining_duration_seconds: f64,
    pub performance_summary: PerformanceSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_signals: usize,
    pub high_confidence_signals: usize,
    pub avg_expected_return: f64,
    pub avg_confidence: f64,
    pub top_coins_by_signal_count: Vec<(String, usize)>,
}

pub fn perform_strategy_mining(
    snapshots: &[SnapshotRecord],
    cfg: &StrategyMiningConfig,
) -> Result<StrategyMiningResult> {
    let start = std::time::Instant::now();

    if snapshots.is_empty() {
        return Ok(empty_mining_result(0, 0.0));
    }

    let candidates: Vec<&SnapshotRecord> = snapshots
        .iter()
        .filter(|s| s.features.fills_count >= cfg.min_fills_per_coin)
        .collect();

    if candidates.len() < 2 {
        return Ok(empty_mining_result(
            snapshots.len(),
            start.elapsed().as_secs_f64(),
        ));
    }

    let feature_rows: Vec<Vec<f64>> = candidates
        .iter()
        .map(|s| build_feature_vector(&s.features))
        .collect();
    let mut matrix = array_from_rows(&feature_rows)?;
    standardize_columns(&mut matrix);

    let n_rows = matrix.nrows();
    let k = cfg.k_clusters.clamp(2, n_rows.max(2));
    let dataset = linfa::DatasetBase::from(matrix.clone());
    let model = match KMeans::params(k)
        .max_n_iterations(300)
        .tolerance(1e-4)
        .fit(&dataset)
    {
        Ok(m) => m,
        Err(err) => {
            eprintln!(
                "⚠️ KMeans fit failed ({} samples, k={}): {:#}",
                matrix.nrows(),
                k,
                err
            );
            return Ok(empty_mining_result(
                snapshots.len(),
                start.elapsed().as_secs_f64(),
            ));
        }
    };

    let assignments = model.predict(&matrix);

    let mut by_cluster: HashMap<usize, Vec<&SnapshotRecord>> = HashMap::new();
    for (idx, label) in assignments.iter().enumerate() {
        by_cluster.entry(*label).or_default().push(candidates[idx]);
    }

    let mut patterns = Vec::new();
    let mut aggregates = Vec::new();
    let mut pattern_id = 1usize;

    for (cluster_id, samples) in by_cluster.into_iter() {
        if samples.len() < 2 {
            continue;
        }

        let returns: Vec<f64> = samples
            .iter()
            .map(|s| calc_expected_return(&s.features))
            .collect();

        let mean_return = mean(&returns);
        let median_return = median(&returns);
        let volatility = std_dev(&returns);
        let sharpe = if volatility > 1e-9 {
            mean_return / volatility
        } else {
            0.0
        };
        let max_drawdown = max_drawdown_from_returns(&returns);

        let win_values: Vec<f64> = samples.iter().filter_map(|s| s.features.win_rate).collect();
        let win_rate = if win_values.is_empty() {
            0.0
        } else {
            mean(&win_values)
        };

        let mut coin_counts: HashMap<String, usize> = HashMap::new();
        for sample in &samples {
            if let Some(coin) = dominant_coin(&sample.snapshot) {
                *coin_counts.entry(coin).or_insert(0) += 1;
            }
        }
        let coin = coin_counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1))
            .map(|(c, _)| c)
            .unwrap_or_else(|| "UNKNOWN".to_string());

        let confidence_score = win_rate * (samples.len() as f64).ln_1p();
        let avg_volume =
            samples.iter().map(|s| s.features.total_volume).sum::<f64>() / samples.len() as f64;

        let pattern = TradingPattern {
            pattern_id,
            coin: coin.clone(),
            cluster_id,
            sample_count: samples.len(),
            mean_forward_return: mean_return,
            median_forward_return: median_return,
            win_rate,
            volatility,
            sharpe_ratio: sharpe,
            max_drawdown,
            confidence_score,
        };

        aggregates.push((pattern.clone(), mean_return, win_rate, avg_volume));
        patterns.push(pattern);
        pattern_id += 1;
    }

    let mut signals = Vec::new();
    let now_ms = current_millis();

    for (pattern, mean_return, win_rate, avg_volume) in &aggregates {
        let action = if *mean_return >= cfg.min_edge_threshold {
            Some(("LONG", *mean_return, *win_rate))
        } else if *mean_return <= -cfg.min_edge_threshold {
            Some(("SHORT", -*mean_return, 1.0 - *win_rate))
        } else {
            None
        };

        if let Some((action, edge, win_metric)) = action {
            if win_metric < cfg.min_win_rate {
                continue;
            }

            let confidence = win_metric * (pattern.sample_count as f64).ln_1p();
            let stop_loss = -edge.max(cfg.min_edge_threshold);
            let take_profit = edge * 2.0;

            signals.push(TradingSignal {
                signal_id: format!("SIG-{:03}", pattern.pattern_id),
                coin: pattern.coin.clone(),
                pattern_id: pattern.pattern_id,
                action: action.to_string(),
                confidence,
                expected_return: edge,
                stop_loss,
                take_profit,
                hold_bars: cfg.horizon_bars,
                min_volume_filter: *avg_volume,
                created_at: now_ms,
            });
        }
    }

    let total_snapshots = snapshots.len();
    let mining_duration_seconds = start.elapsed().as_secs_f64();
    let analyzed_coins = patterns
        .iter()
        .map(|p| p.coin.clone())
        .filter(|c| c != "UNKNOWN")
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let total_signals = signals.len();
    let high_confidence_signals = signals
        .iter()
        .filter(|s| s.confidence >= cfg.min_win_rate)
        .count();
    let avg_expected_return = if total_signals > 0 {
        signals.iter().map(|s| s.expected_return).sum::<f64>() / total_signals as f64
    } else {
        0.0
    };
    let avg_confidence = if total_signals > 0 {
        signals.iter().map(|s| s.confidence).sum::<f64>() / total_signals as f64
    } else {
        0.0
    };

    let mut coin_counts: HashMap<String, usize> = HashMap::new();
    for sig in &signals {
        *coin_counts.entry(sig.coin.clone()).or_insert(0) += 1;
    }
    let mut top_coins_by_signal_count: Vec<(String, usize)> = coin_counts.into_iter().collect();
    top_coins_by_signal_count.sort_by(|a, b| b.1.cmp(&a.1));
    top_coins_by_signal_count.truncate(5);

    Ok(StrategyMiningResult {
        patterns,
        signals,
        total_snapshots,
        analyzed_coins,
        mining_duration_seconds,
        performance_summary: PerformanceSummary {
            total_signals,
            high_confidence_signals,
            avg_expected_return,
            avg_confidence,
            top_coins_by_signal_count,
        },
    })
}

fn empty_mining_result(total_snapshots: usize, duration: f64) -> StrategyMiningResult {
    StrategyMiningResult {
        patterns: Vec::new(),
        signals: Vec::new(),
        total_snapshots,
        analyzed_coins: Vec::new(),
        mining_duration_seconds: duration,
        performance_summary: PerformanceSummary {
            total_signals: 0,
            high_confidence_signals: 0,
            avg_expected_return: 0.0,
            avg_confidence: 0.0,
            top_coins_by_signal_count: Vec::new(),
        },
    }
}

fn build_feature_vector(features: &SnapshotFeatures) -> Vec<f64> {
    vec![
        (features.fills_count as f64).ln_1p(),
        (features.open_orders_count as f64).ln_1p(),
        (features.frontend_orders_count as f64).ln_1p(),
        (features.historical_orders_count as f64).ln_1p(),
        (features.twap_fills_count as f64).ln_1p(),
        features
            .fills_window_count
            .map(|c| (c as f64).ln_1p())
            .unwrap_or(0.0),
        (features.positions_count as f64).ln_1p(),
        features.total_volume.ln_1p(),
        features.total_fees.ln_1p(),
        (features.unique_coins as f64).ln_1p(),
        features.avg_fill_size.ln_1p(),
        features.largest_fill.ln_1p(),
        features.activity_span_hours.ln_1p(),
        (features.fills_with_timestamp_count as f64).ln_1p(),
        features.account_value.map(signed_log1p).unwrap_or(0.0),
        features.realized_pnl.map(signed_log1p).unwrap_or(0.0),
        features.unrealized_pnl.map(signed_log1p).unwrap_or(0.0),
        features.leverage.unwrap_or(0.0),
        features.win_rate.unwrap_or(0.0),
    ]
}

fn signed_log1p(value: f64) -> f64 {
    if value >= 0.0 {
        value.ln_1p()
    } else {
        -(-value).ln_1p()
    }
}

fn array_from_rows(rows: &[Vec<f64>]) -> Result<Array2<f64>> {
    if rows.is_empty() {
        bail!("no rows to build matrix");
    }
    let cols = rows[0].len();
    let mut mat = Array2::<f64>::zeros((rows.len(), cols));
    for (i, row) in rows.iter().enumerate() {
        if row.len() != cols {
            bail!("inconsistent feature length");
        }
        for (j, value) in row.iter().enumerate() {
            mat[[i, j]] = *value;
        }
    }
    Ok(mat)
}

fn standardize_columns(mat: &mut Array2<f64>) {
    let (rows, cols) = mat.dim();
    if rows < 2 {
        return;
    }
    for j in 0..cols {
        let mut mean = 0.0;
        for i in 0..rows {
            mean += mat[[i, j]];
        }
        mean /= rows as f64;

        let mut var = 0.0;
        for i in 0..rows {
            let d = mat[[i, j]] - mean;
            var += d * d;
        }
        let denom = (rows as f64 - 1.0).max(1.0);
        let std = (var / denom).sqrt().max(1e-9);

        for i in 0..rows {
            mat[[i, j]] = (mat[[i, j]] - mean) / std;
        }
    }
}

fn calc_expected_return(features: &SnapshotFeatures) -> f64 {
    match (features.realized_pnl, features.account_value) {
        (Some(pnl), Some(account)) if account.abs() > 1e-6 => pnl / account,
        (Some(pnl), _) => pnl,
        _ => 0.0,
    }
}

fn dominant_coin(snapshot: &HyperliquidUserSnapshot) -> Option<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for fill in &snapshot.fills {
        if let Some(coin) = fill.get("coin").and_then(|v| v.as_str()) {
            *counts.entry(coin.to_string()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(coin, _)| coin)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mu = mean(values);
    let var = values
        .iter()
        .map(|v| {
            let d = v - mu;
            d * d
        })
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    var.sqrt()
}

fn max_drawdown_from_returns(returns: &[f64]) -> f64 {
    let mut cumulative = 0.0;
    let mut peak = 0.0;
    let mut max_dd = 0.0;
    for &r in returns {
        cumulative += r;
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = peak - cumulative;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }
    max_dd
}

/// Convenience function to run complete snapshot and strategy mining pipeline
pub async fn run_complete_analysis_pipeline(
    client: Arc<dyn InfoApiClient>,
    leaderboard_json: &str,
    snapshot_config: BulkSnapshotConfig,
    output_config: PolarsOutputConfig,
    mining_config: StrategyMiningConfig,
) -> Result<(BulkSnapshotResult, StrategyMiningResult)> {
    // Step 1: Fetch all snapshots
    eprintln!("Starting bulk snapshot fetching...");
    let (bulk_result, snapshots) = fetch_snapshots_from_leaderboard(
        Arc::clone(&client),
        leaderboard_json,
        snapshot_config,
        output_config,
    )
    .await?;

    eprintln!(
        "Snapshot fetching complete: {} successful, {} failed",
        bulk_result.successful, bulk_result.failed
    );

    // Step 2: Run strategy mining on the collected snapshots
    let mining_result = perform_strategy_mining(&snapshots, &mining_config)?;

    eprintln!("Analysis pipeline complete!");

    Ok((bulk_result, mining_result))
}
