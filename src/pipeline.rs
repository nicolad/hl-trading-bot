use std::sync::Arc;

use anyhow::Result;

use crate::info_snapshot::{
    BulkSnapshotConfig, BulkSnapshotResult, PolarsOutputConfig, StrategyMiningConfig,
    StrategyMiningResult, fetch_snapshots_from_leaderboard, perform_strategy_mining,
};
use crate::leaderboard::InfoApiClient;

/// Aggregated configuration for the full snapshot → strategy mining pipeline.
#[derive(Clone, Debug)]
pub struct SnapshotPipelineConfig {
    pub bulk: BulkSnapshotConfig,
    pub output: PolarsOutputConfig,
    pub strategy: StrategyMiningConfig,
}

impl Default for SnapshotPipelineConfig {
    fn default() -> Self {
        Self {
            bulk: BulkSnapshotConfig::default(),
            output: PolarsOutputConfig::default(),
            strategy: StrategyMiningConfig::default(),
        }
    }
}

/// Run the complete pipeline inside a single module.
/// Returns both the bulk snapshot metrics and the derived strategy mining output.
pub async fn run_pipeline(
    client: Arc<dyn InfoApiClient>,
    leaderboard_json: &str,
    cfg: SnapshotPipelineConfig,
) -> Result<(BulkSnapshotResult, StrategyMiningResult)> {
    let (bulk, snapshots) = fetch_snapshots_from_leaderboard(
        Arc::clone(&client),
        leaderboard_json,
        cfg.bulk.clone(),
        cfg.output.clone(),
    )
    .await?;

    let mining = perform_strategy_mining(&snapshots, &cfg.strategy)?;
    Ok((bulk, mining))
}
