use std::path::PathBuf;

use anyhow::Result;
use tracing_subscriber::EnvFilter;

use hyperliquid_bot::{config::BotConfig, engine::TradingEngine};

fn init_tracing(level: &str) -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(level))
        .unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("bots/btc_conservative.yaml"));
    let config = BotConfig::load(&config_path)?;
    init_tracing(config.monitoring.log_level.as_str())?;
    tracing::info!(bot = %config.name, "starting hyperliquid bot");
    let engine = TradingEngine::new(config)?;
    engine.initialize().await?;
    engine.start().await?;
    tracing::info!("engine running, press Ctrl+C to stop");
    tokio::signal::ctrl_c().await?;
    tracing::info!("shutdown signal received");
    engine.stop().await?;
    Ok(())
}
