use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::json;

use crate::config::{BotConfig, RangeMode};
use crate::interfaces::{MarketData, Position};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RiskAction {
    None,
    ClosePosition,
    ReducePosition,
    CancelOrders,
    PauseTrading,
    EmergencyExit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RiskEvent {
    pub rule_name: String,
    pub asset: String,
    pub action: RiskAction,
    pub reason: String,
    pub severity: Severity,
    pub metadata: serde_yaml::Value,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AccountMetrics {
    pub total_value: f64,
    pub total_pnl: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub drawdown_pct: f64,
    pub positions_count: usize,
    pub largest_position_pct: f64,
}

impl AccountMetrics {
    pub fn zero() -> Self {
        Self {
            total_value: 0.0,
            total_pnl: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            drawdown_pct: 0.0,
            positions_count: 0,
            largest_position_pct: 0.0,
        }
    }
}

pub struct RiskManager {
    config: crate::config::RiskManagementConfig,
    grid: crate::config::GridConfig,
    last_rebalance: Option<Instant>,
    rebalance_count: u32,
    trading_paused: bool,
}

impl RiskManager {
    pub fn new(config: &BotConfig) -> Self {
        Self {
            config: config.risk_management.clone(),
            grid: config.grid.clone(),
            last_rebalance: None,
            rebalance_count: 0,
            trading_paused: false,
        }
    }

    pub fn evaluate(
        &mut self,
        positions: &[Position],
        market_data: &MarketData,
        metrics: &AccountMetrics,
    ) -> Vec<RiskEvent> {
        let mut events = Vec::new();
        events.extend(self.check_stop_loss(positions));
        events.extend(self.check_take_profit(positions));
        events.extend(self.check_drawdown(metrics));
        events.extend(self.check_position_size(metrics));
        events.extend(self.check_rebalance(market_data));
        if events.iter().any(|e| e.action == RiskAction::PauseTrading) {
            self.trading_paused = true;
        }
        events
    }

    pub fn trading_paused(&self) -> bool {
        self.trading_paused
    }

    fn check_stop_loss(&self, positions: &[Position]) -> Vec<RiskEvent> {
        if !self.config.stop_loss_enabled {
            return Vec::new();
        }
        let mut events = Vec::new();
        for position in positions {
            if position.entry_price <= 0.0 {
                continue;
            }
            if position.unrealized_pnl >= 0.0 {
                continue;
            }
            let basis = position.entry_price * position.size.abs();
            if basis <= 0.0 {
                continue;
            }
            let loss_pct = (position.unrealized_pnl.abs() / basis) * 100.0;
            if loss_pct >= self.config.stop_loss_pct {
                events.push(self.event(
                    "stop_loss",
                    &position.asset,
                    RiskAction::ClosePosition,
                    Severity::High,
                    format!(
                        "loss {:.2}% exceeds {:.2}%",
                        loss_pct, self.config.stop_loss_pct
                    ),
                    &position,
                ));
            }
        }
        events
    }

    fn check_take_profit(&self, positions: &[Position]) -> Vec<RiskEvent> {
        if !self.config.take_profit_enabled {
            return Vec::new();
        }
        let mut events = Vec::new();
        for position in positions {
            if position.entry_price <= 0.0 || position.unrealized_pnl <= 0.0 {
                continue;
            }
            let basis = position.entry_price * position.size.abs();
            if basis <= 0.0 {
                continue;
            }
            let profit_pct = (position.unrealized_pnl / basis) * 100.0;
            if profit_pct >= self.config.take_profit_pct {
                events.push(self.event(
                    "take_profit",
                    &position.asset,
                    RiskAction::ClosePosition,
                    Severity::Medium,
                    format!(
                        "profit {:.2}% exceeds {:.2}%",
                        profit_pct, self.config.take_profit_pct
                    ),
                    &position,
                ));
            }
        }
        events
    }

    fn check_drawdown(&self, metrics: &AccountMetrics) -> Vec<RiskEvent> {
        if metrics.drawdown_pct < self.config.max_drawdown_pct {
            return Vec::new();
        }
        vec![self.event(
            "drawdown",
            &self.grid.symbol,
            RiskAction::PauseTrading,
            Severity::Critical,
            format!(
                "drawdown {:.2}% exceeds {:.2}%",
                metrics.drawdown_pct, self.config.max_drawdown_pct
            ),
            &metrics.drawdown_pct,
        )]
    }

    fn check_position_size(&self, metrics: &AccountMetrics) -> Vec<RiskEvent> {
        if metrics.largest_position_pct <= self.config.max_position_size_pct {
            return Vec::new();
        }
        vec![self.event(
            "position_size",
            &self.grid.symbol,
            RiskAction::ReducePosition,
            Severity::High,
            format!(
                "position {:.2}% exceeds {:.2}%",
                metrics.largest_position_pct, self.config.max_position_size_pct
            ),
            &metrics.largest_position_pct,
        )]
    }

    fn check_rebalance(&mut self, market_data: &MarketData) -> Vec<RiskEvent> {
        let threshold = self.config.rebalance.price_move_threshold_pct;
        let (lower_bound, upper_bound) = match self.grid.price_range.mode {
            RangeMode::Auto => {
                let center = market_data.price;
                let span = center * (self.grid.price_range.auto.range_pct / 100.0);
                (center - span, center + span)
            }
            RangeMode::Manual => {
                let min = self.grid.price_range.manual.min;
                let max = self.grid.price_range.manual.max;
                if min <= 0.0 || max <= min {
                    return Vec::new();
                }
                (min, max)
            }
        };
        let upper_trigger = upper_bound * (1.0 + threshold / 100.0);
        let lower_trigger = lower_bound * (1.0 - threshold / 100.0);
        if (market_data.price >= upper_trigger || market_data.price <= lower_trigger)
            && self.can_rebalance()
        {
            self.last_rebalance = Some(Instant::now());
            self.rebalance_count += 1;
            return vec![self.event(
                "rebalance",
                &market_data.asset,
                RiskAction::CancelOrders,
                Severity::Medium,
                format!(
                    "price {:.2} outside [{:.2}, {:.2}] with threshold {:.2}%",
                    market_data.price, lower_bound, upper_bound, threshold
                ),
                &json!({
                    "price": market_data.price,
                    "lower": lower_bound,
                    "upper": upper_bound,
                }),
            )];
        }
        Vec::new()
    }

    fn can_rebalance(&self) -> bool {
        if let Some(last) = self.last_rebalance {
            let cooldown_secs = (self.config.rebalance.cooldown_minutes as u64) * 60;
            if last.elapsed() < Duration::from_secs(cooldown_secs) {
                return false;
            }
        }
        if self.rebalance_count >= self.config.rebalance.max_rebalances_per_day {
            return false;
        }
        true
    }

    fn event<T: Serialize>(
        &self,
        rule: &str,
        asset: &str,
        action: RiskAction,
        severity: Severity,
        reason: String,
        payload: &T,
    ) -> RiskEvent {
        RiskEvent {
            rule_name: rule.into(),
            asset: asset.into(),
            action,
            reason,
            severity,
            metadata: serde_yaml::to_value(payload).unwrap_or_default(),
            timestamp: Utc::now(),
        }
    }
}

pub struct RiskEvaluator<'a> {
    manager: &'a mut RiskManager,
}

impl<'a> RiskEvaluator<'a> {
    pub fn new(manager: &'a mut RiskManager) -> Self {
        Self { manager }
    }

    pub fn evaluate(
        &mut self,
        positions: &[Position],
        market_data: &MarketData,
        metrics: &AccountMetrics,
    ) -> Vec<RiskEvent> {
        self.manager.evaluate(positions, market_data, metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BotConfig, RangeMode};

    fn sample_config() -> BotConfig {
        let yaml = include_str!("../bots/btc_conservative.yaml");
        BotConfig::load_from_str(yaml).unwrap()
    }

    fn make_position(price: f64, pnl: f64) -> Position {
        Position {
            asset: "BTC".into(),
            size: 0.1,
            entry_price: 100_000.0,
            current_value: price * 0.1,
            unrealized_pnl: pnl,
            timestamp: Utc::now(),
        }
    }

    fn market(price: f64) -> MarketData {
        MarketData {
            asset: "BTC".into(),
            price,
            volume_24h: 0.0,
            timestamp: Utc::now(),
            bid: None,
            ask: None,
            volatility: None,
        }
    }

    #[test]
    fn triggers_stop_loss() {
        let mut cfg = sample_config();
        cfg.risk_management.stop_loss_enabled = true;
        cfg.risk_management.stop_loss_pct = 5.0;
        let mut manager = RiskManager::new(&cfg);
        let position = make_position(90_000.0, -1_500.0);
        let events = manager.evaluate(&[position], &market(90_000.0), &AccountMetrics::zero());
        assert!(events.iter().any(|e| e.rule_name == "stop_loss"));
    }

    #[test]
    fn pauses_on_drawdown() {
        let cfg = sample_config();
        let mut manager = RiskManager::new(&cfg);
        let metrics = AccountMetrics {
            total_value: 10_000.0,
            total_pnl: -2_000.0,
            unrealized_pnl: -1_000.0,
            realized_pnl: -1_000.0,
            drawdown_pct: 20.0,
            positions_count: 1,
            largest_position_pct: 5.0,
        };
        let events = manager.evaluate(&[], &market(100_000.0), &metrics);
        assert!(events.iter().any(|e| e.action == RiskAction::PauseTrading));
        assert!(manager.trading_paused());
    }

    #[test]
    fn respects_manual_price_range_for_rebalance() {
        let mut cfg = sample_config();
        cfg.grid.price_range.mode = RangeMode::Manual;
        cfg.grid.price_range.manual.min = 95_000.0;
        cfg.grid.price_range.manual.max = 105_000.0;
        cfg.risk_management.rebalance.price_move_threshold_pct = 10.0;
        let mut manager = RiskManager::new(&cfg);

        // Price within manual bounds should not trigger rebalance
        let events = manager.evaluate(&[], &market(100_000.0), &AccountMetrics::zero());
        assert!(events.iter().all(|e| e.rule_name != "rebalance"));

        // Price far outside manual bounds should trigger
        let events = manager.evaluate(&[], &market(140_000.0), &AccountMetrics::zero());
        assert!(events.iter().any(|e| e.rule_name == "rebalance"));
    }
}
