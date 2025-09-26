use anyhow::Result;

mod core;
pub mod engines;

pub use core::{BacktestResult, PriceSample, TradeExecution};
use core::execute_backtest;

use crate::config::BotConfig;

#[derive(Clone, Copy)]
pub struct BacktestRequest<'a> {
    pub config: &'a BotConfig,
    pub initial_cash: f64,
    pub samples: &'a [PriceSample],
}

impl<'a> BacktestRequest<'a> {
    pub fn new(config: &'a BotConfig, initial_cash: f64, samples: &'a [PriceSample]) -> Self {
        Self {
            config,
            initial_cash,
            samples,
        }
    }
}

pub trait BacktestEngine {
    fn run(&self, request: BacktestRequest<'_>) -> Result<BacktestResult>;
}

pub(crate) fn run_core_backtest(
    config: &BotConfig,
    initial_cash: f64,
    samples: &[PriceSample],
) -> Result<BacktestResult> {
    execute_backtest(config, initial_cash, samples)
}
