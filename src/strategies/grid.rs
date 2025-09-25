use std::f64;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::json;
use serde_yaml::to_value;

use crate::{
    config::{BotConfig, RangeMode},
    interfaces::{MarketData, Position, SignalType, TradingSignal, TradingStrategy},
};

#[derive(Clone, Copy, PartialEq)]
enum GridState {
    Initializing,
    Active,
    Rebalancing,
    Stopped,
}

#[derive(Clone)]
struct GridLevel {
    price: f64,
    size: f64,
    index: usize,
    is_buy: bool,
    filled: bool,
}

pub struct BasicGridStrategy {
    symbol: String,
    levels: usize,
    range_pct: f64,
    total_allocation: f64,
    manual_min: Option<f64>,
    manual_max: Option<f64>,
    rebalance_threshold_pct: f64,
    state: GridState,
    center_price: Option<f64>,
    grid_levels: Vec<GridLevel>,
    last_rebalance: Option<DateTime<Utc>>,
    total_trades: usize,
    total_profit: f64,
    active: bool,
}

impl BasicGridStrategy {
    pub fn new(config: &BotConfig) -> Self {
        let grid = &config.grid;
        let range_pct = match grid.price_range.mode {
            RangeMode::Auto => grid.price_range.auto.range_pct,
            RangeMode::Manual => {
                let min = grid.price_range.manual.min;
                let max = grid.price_range.manual.max;
                ((max - min) / ((max + min) / 2.0)) * 100.0
            }
        };
        let manual_min = if grid.price_range.mode == RangeMode::Manual {
            Some(grid.price_range.manual.min)
        } else {
            None
        };
        let manual_max = if grid.price_range.mode == RangeMode::Manual {
            Some(grid.price_range.manual.max)
        } else {
            None
        };
        let allocation = match grid.position_sizing.mode {
            crate::config::PositionSizingMode::Auto => {
                let levels = grid.levels as f64;
                grid.position_sizing.auto.min_position_size_usd * levels
            }
            crate::config::PositionSizingMode::Manual => {
                grid.position_sizing.manual.size_per_level * grid.levels as f64
            }
        };
        Self {
            symbol: grid.symbol.clone(),
            levels: grid.levels as usize,
            range_pct,
            total_allocation: allocation,
            manual_min,
            manual_max,
            rebalance_threshold_pct: config.risk_management.rebalance.price_move_threshold_pct,
            state: GridState::Initializing,
            center_price: None,
            grid_levels: Vec::new(),
            last_rebalance: None,
            total_trades: 0,
            total_profit: 0.0,
            active: true,
        }
    }

    fn build_metadata(&self, level: &GridLevel, label: &str) -> serde_yaml::Value {
        to_value(json!({
            "level_index": level.index,
            "grid_type": label,
        }))
        .unwrap_or(serde_yaml::Value::Null)
    }

    fn build_signal(
        &self,
        signal_type: SignalType,
        level: &GridLevel,
        label: &str,
    ) -> TradingSignal {
        TradingSignal {
            signal_type,
            asset: self.symbol.clone(),
            size: level.size,
            price: Some(level.price),
            reason: Some(format!("grid level {} at {:.2}", level.index, level.price)),
            metadata: self.build_metadata(level, label),
        }
    }

    fn initialize_grid(&mut self, current_price: f64) -> Vec<TradingSignal> {
        self.center_price = Some(current_price);
        let (min_price, max_price) = match (self.manual_min, self.manual_max) {
            (Some(min), Some(max)) => (min, max),
            _ => {
                let range = current_price * (self.range_pct / 100.0);
                (current_price - range, current_price + range)
            }
        };
        self.grid_levels = self.create_levels(min_price, max_price, current_price);
        let mut signals = Vec::new();
        for level in &self.grid_levels {
            if level.is_buy && level.price < current_price {
                signals.push(self.build_signal(SignalType::Buy, level, "initial"));
            } else if !level.is_buy && level.price > current_price {
                signals.push(self.build_signal(SignalType::Sell, level, "initial"));
            }
        }
        self.state = GridState::Active;
        signals
    }

    fn create_levels(&self, min_price: f64, max_price: f64, current_price: f64) -> Vec<GridLevel> {
        let mut levels = Vec::with_capacity(self.levels);
        if self.levels <= 1 {
            return levels;
        }
        let size_per_level = self.total_allocation / self.levels as f64;
        let ratio = (max_price / min_price).powf(1.0 / (self.levels as f64 - 1.0));
        for index in 0..self.levels {
            let price = min_price * ratio.powf(index as f64);
            let size = size_per_level / price.max(f64::EPSILON);
            levels.push(GridLevel {
                price,
                size,
                index,
                is_buy: price < current_price,
                filled: false,
            });
        }
        levels
    }

    fn should_rebalance(&self, current_price: f64) -> bool {
        match self.center_price {
            Some(center) => {
                ((current_price - center).abs() / center) * 100.0 > self.rebalance_threshold_pct
            }
            None => false,
        }
    }

    fn rebalance_grid(&mut self, current_price: f64) -> Vec<TradingSignal> {
        self.state = GridState::Rebalancing;
        let mut signals = Vec::new();
        let cancel = TradingSignal {
            signal_type: SignalType::Close,
            asset: self.symbol.clone(),
            size: 0.0,
            price: None,
            reason: Some("rebalance".into()),
            metadata: to_value(json!({ "action": "cancel_all" }))
                .unwrap_or(serde_yaml::Value::Null),
        };
        signals.push(cancel);
        self.state = GridState::Initializing;
        signals.extend(self.initialize_grid(current_price));
        self.last_rebalance = Some(Utc::now());
        signals
    }
}

impl TradingStrategy for BasicGridStrategy {
    fn generate_signals(
        &mut self,
        market_data: &MarketData,
        _positions: &[Position],
        _balance: f64,
    ) -> Result<Vec<TradingSignal>> {
        if !self.active {
            return Ok(Vec::new());
        }
        match self.state {
            GridState::Initializing => Ok(self.initialize_grid(market_data.price)),
            GridState::Active => {
                if self.should_rebalance(market_data.price) {
                    Ok(self.rebalance_grid(market_data.price))
                } else {
                    Ok(Vec::new())
                }
            }
            _ => Ok(Vec::new()),
        }
    }

    fn on_trade_executed(
        &mut self,
        signal: &TradingSignal,
        executed_price: f64,
        executed_size: f64,
    ) -> Result<()> {
        self.total_trades += 1;
        if let Some(index) = signal.metadata.get("level_index").and_then(|v| v.as_u64())
            && let Some(level) = self
                .grid_levels
                .iter_mut()
                .find(|lvl| lvl.index as u64 == index)
        {
            level.filled = true;
            if signal.signal_type == SignalType::Sell {
                let buy_price = executed_price * 0.99;
                let profit = (executed_price - buy_price) * executed_size;
                self.total_profit += profit;
            }
        }
        Ok(())
    }

    fn on_error(&mut self, _error: &anyhow::Error) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        &self.symbol
    }

    fn start(&mut self) {
        self.active = true;
    }

    fn stop(&mut self) {
        self.active = false;
        self.state = GridState::Stopped;
    }

    fn get_status(&self) -> serde_yaml::Value {
        let active_levels = self.grid_levels.iter().filter(|lvl| !lvl.filled).count();
        let filled_levels = self.grid_levels.len().saturating_sub(active_levels);
        to_value(json!({
            "name": self.symbol,
            "active": self.active,
            "state": match self.state {
                GridState::Initializing => "initializing",
                GridState::Active => "active",
                GridState::Rebalancing => "rebalancing",
                GridState::Stopped => "stopped",
            },
            "center_price": self.center_price,
            "total_levels": self.grid_levels.len(),
            "active_levels": active_levels,
            "filled_levels": filled_levels,
            "total_trades": self.total_trades,
            "total_profit": self.total_profit,
            "last_rebalance": self.last_rebalance.map(|dt| dt.to_rfc3339()),
        }))
        .unwrap_or(serde_yaml::Value::Null)
    }
}
