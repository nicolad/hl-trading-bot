use anyhow::Result;
use chrono::{DateTime, Utc};

pub mod ema;

pub use nautilus_backtest::engine::BacktestEngine;
pub use nautilus_backtest::config::BacktestEngineConfig;
pub use nautilus_model::data::Bar;

use crate::config::BotConfig;

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub initial_balance: f64,
    pub final_balance: f64,
    pub total_pnl: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub trades: Vec<TradeExecution>,
}

#[derive(Debug, Clone)]
pub struct TradeExecution {
    pub timestamp: DateTime<Utc>,
    pub asset: String,
    pub side: String,
    pub quantity: f64,
    pub price: f64,
    pub fee: f64,
    pub pnl: f64,
}

#[derive(Debug, Clone)]
pub struct PriceSample {
    pub timestamp: DateTime<Utc>,
    pub price: f64,
}

impl PriceSample {
    pub fn new(timestamp: DateTime<Utc>, price: f64) -> Self {
        Self { timestamp, price }
    }
}

pub fn run_ema_backtest(
    config: &BotConfig,
    initial_cash: f64,
    prices: &[(DateTime<Utc>, f64)],
) -> Result<BacktestResult> {
    ema::run_backtest(config, initial_cash, prices)
}
