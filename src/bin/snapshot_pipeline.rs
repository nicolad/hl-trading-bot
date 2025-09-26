use anyhow::{Context, Result};
use hyperliquid_bot::info_snapshot::{
    BulkSnapshotConfig, PolarsOutputConfig, StrategyMiningConfig,
};
use hyperliquid_bot::leaderboard::ReqwestInfoClient;
use hyperliquid_bot::pipeline::{SnapshotPipelineConfig, run_pipeline};
use std::fs;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let leaderboard_json = fs::read_to_string("leaderboard.json")
        .context("reading leaderboard.json (expected Hyperliquid leaderboard dump)")?;

    let client = Arc::new(ReqwestInfoClient::new("https://api.hyperliquid.xyz/info")?);

    let cfg = SnapshotPipelineConfig {
        bulk: BulkSnapshotConfig {
            concurrency: 10,
            retry_count: 3,
            retry_delay_ms: 1_000,
            timeout_seconds: 30,
            batch_size: 50,
        },
        output: PolarsOutputConfig {
            ndjson_path: Some("snapshots.ndjson".into()),
            features_csv_path: Some("trading_features.csv".into()),
            snapshots_json_path: Some("snapshots_raw.json".into()),
            include_metadata: true,
        },
        strategy: StrategyMiningConfig {
            bar_interval_ms: 300_000,
            window_bars: 20,
            horizon_bars: 5,
            k_clusters: 8,
            min_fills_per_coin: 50,
            min_edge_threshold: 0.002,
            min_win_rate: 0.55,
            output_patterns_json: Some("trading_patterns.json".into()),
            output_rules_csv: Some("trading_signals.csv".into()),
        },
    };

    println!("🚀 Starting snapshot analysis pipeline...");
    let (bulk_result, mining_result) = run_pipeline(client, &leaderboard_json, cfg).await?;

    println!("\n✅ Pipeline completed successfully!");
    println!("\n📈 Snapshot Results:");
    println!("  - Total addresses: {}", bulk_result.total_addresses);
    println!("  - Successful fetches: {}", bulk_result.successful);
    println!("  - Failed fetches: {}", bulk_result.failed);
    println!("  - Elapsed time: {:.2}s", bulk_result.elapsed_seconds);

    if !bulk_result.failed_addresses.is_empty() {
        println!("\n❌ Failed addresses:");
        for addr in &bulk_result.failed_addresses {
            println!("  - {}", addr);
        }
    }

    println!("\n🧠 Strategy Mining Results:");
    println!("  - Patterns found: {}", mining_result.patterns.len());
    println!("  - Trading signals: {}", mining_result.signals.len());
    println!(
        "  - High confidence signals: {}",
        mining_result.performance_summary.high_confidence_signals
    );
    println!(
        "  - Average expected return: {:.4}",
        mining_result.performance_summary.avg_expected_return
    );
    println!(
        "  - Mining duration: {:.2}s",
        mining_result.mining_duration_seconds
    );

    println!("\n📁 Output Files Generated:");
    println!("  - snapshots.ndjson (Polars NDJSON)");
    println!("  - trading_features.csv (Features for ML)");
    println!("  - raw_snapshots.json (Compatibility format)");
    println!("  - trading_patterns.json (Discovered patterns)");
    println!("  - trading_signals.csv (Generated trading signals)");

    println!("\n💡 Next Steps:");
    println!("  - Load NDJSON in Polars: pl.read_ndjson('snapshots.ndjson')");
    println!("  - Load features CSV: pl.read_csv('trading_features.csv')");
    println!("  - Analyze patterns with your ML pipeline");
    println!("  - Backtest signals on historical data");

    Ok(())
}
