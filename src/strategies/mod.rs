mod grid;

use anyhow::Result;

use crate::{config::BotConfig, interfaces::TradingStrategy};

pub use grid::BasicGridStrategy;

pub fn create_strategy(config: &BotConfig) -> Result<Box<dyn TradingStrategy>> {
    let strategy = BasicGridStrategy::new(config);
    Ok(Box::new(strategy))
}
