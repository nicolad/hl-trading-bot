pub mod ema;

use anyhow::Result;

use crate::{
    config::{BotConfig, StrategyKind},
    interfaces::TradingStrategy,
};

pub use ema::EmaStrategy;

pub fn create_strategy(config: &BotConfig) -> Result<Box<dyn TradingStrategy>> {
    let strategy: Box<dyn TradingStrategy> = match config.strategy.kind {
        StrategyKind::Grid => {
            return Err(anyhow::anyhow!("Grid strategy not implemented yet"));
        }
        StrategyKind::Ema => Box::new(EmaStrategy::new(config)?),
    };
    Ok(strategy)
}
