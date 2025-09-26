use anyhow::Result;

use crate::backtest::{BacktestResult, PriceSample};
use crate::config::BotConfig;

pub fn run_backtest(
    config: &BotConfig,
    initial_cash: f64,
    samples: &[PriceSample],
) -> Result<BacktestResult> {
    // Use EMA backtest specifically - you can modify this to switch strategy types
    let prices: Vec<_> = samples
        .iter()
        .map(|s| (s.timestamp, s.price))
        .collect();
    
    crate::backtest::run_ema_backtest(config, initial_cash, &prices)
}
