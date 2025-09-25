use std::fmt;
use std::fs;
use std::path::Path;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("configuration file missing")]
    Missing,
    #[error("failed to read configuration: {0}")]
    Read(std::io::Error),
    #[error("failed to parse configuration: {0}")]
    Parse(serde_yaml::Error),
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    Conservative,
    Moderate,
    Aggressive,
}

impl Default for RiskLevel {
    fn default() -> Self {
        Self::Moderate
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct AccountConfig {
    pub max_allocation_pct: f64,
    #[serde(default)]
    pub risk_level: RiskLevel,
}

impl Default for AccountConfig {
    fn default() -> Self {
        Self {
            max_allocation_pct: 20.0,
            risk_level: RiskLevel::default(),
        }
    }
}

impl AccountConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.max_allocation_pct < 1.0 || self.max_allocation_pct > 100.0 {
            return Err(ConfigError::Invalid(
                "max_allocation_pct must be between 1 and 100".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct AutoPriceRangeConfig {
    pub range_pct: f64,
    #[serde(default = "default_true")]
    pub volatility_adjustment: bool,
    #[serde(default = "default_auto_min_range_pct")]
    pub min_range_pct: f64,
    #[serde(default = "default_auto_max_range_pct")]
    pub max_range_pct: f64,
    #[serde(default = "default_volatility_multiplier")]
    pub volatility_multiplier: f64,
}

fn default_true() -> bool {
    true
}

fn default_auto_min_range_pct() -> f64 {
    5.0
}

fn default_auto_max_range_pct() -> f64 {
    25.0
}

fn default_volatility_multiplier() -> f64 {
    2.0
}

impl AutoPriceRangeConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if !(1.0..=50.0).contains(&self.range_pct) {
            return Err(ConfigError::Invalid(
                "range_pct must be within 1 and 50".into(),
            ));
        }
        if !(1.0..=50.0).contains(&self.min_range_pct) {
            return Err(ConfigError::Invalid(
                "min_range_pct must be within 1 and 50".into(),
            ));
        }
        if !(1.0..=50.0).contains(&self.max_range_pct) {
            return Err(ConfigError::Invalid(
                "max_range_pct must be within 1 and 50".into(),
            ));
        }
        if self.min_range_pct > self.max_range_pct {
            return Err(ConfigError::Invalid(
                "min_range_pct must not exceed max_range_pct".into(),
            ));
        }
        if self.range_pct < self.min_range_pct || self.range_pct > self.max_range_pct {
            return Err(ConfigError::Invalid(
                "range_pct must fall between min_range_pct and max_range_pct".into(),
            ));
        }
        Ok(())
    }
}

impl Default for AutoPriceRangeConfig {
    fn default() -> Self {
        Self {
            range_pct: 10.0,
            volatility_adjustment: true,
            min_range_pct: 5.0,
            max_range_pct: 25.0,
            volatility_multiplier: 2.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ManualPriceRangeConfig {
    pub min: f64,
    pub max: f64,
}

impl Default for ManualPriceRangeConfig {
    fn default() -> Self {
        Self {
            min: 90_000.0,
            max: 120_000.0,
        }
    }
}

impl ManualPriceRangeConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.min <= 0.0 || self.max <= 0.0 {
            return Err(ConfigError::Invalid(
                "manual price bounds must be positive".into(),
            ));
        }
        if self.min >= self.max {
            return Err(ConfigError::Invalid(
                "manual min must be less than max".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RangeMode {
    Auto,
    Manual,
}

impl Default for RangeMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
pub struct PriceRangeConfig {
    #[serde(default)]
    pub mode: RangeMode,
    #[serde(default)]
    pub auto: AutoPriceRangeConfig,
    #[serde(default)]
    pub manual: ManualPriceRangeConfig,
}

impl PriceRangeConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        self.auto.validate()?;
        self.manual.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PositionSizingMode {
    Auto,
    Manual,
}

impl Default for PositionSizingMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct AutoPositionSizingConfig {
    #[serde(default = "default_balance_reserve_pct")]
    pub balance_reserve_pct: f64,
    #[serde(default = "default_max_single_position_pct")]
    pub max_single_position_pct: f64,
    #[serde(default = "default_grid_spacing_strategy")]
    pub grid_spacing_strategy: GridSpacingStrategy,
    #[serde(default = "default_true")]
    pub volatility_position_adjustment: bool,
    #[serde(default = "default_min_position_size_usd")]
    pub min_position_size_usd: f64,
}

fn default_balance_reserve_pct() -> f64 {
    50.0
}

fn default_max_single_position_pct() -> f64 {
    10.0
}

fn default_min_position_size_usd() -> f64 {
    10.0
}

impl Default for AutoPositionSizingConfig {
    fn default() -> Self {
        Self {
            balance_reserve_pct: default_balance_reserve_pct(),
            max_single_position_pct: default_max_single_position_pct(),
            grid_spacing_strategy: default_grid_spacing_strategy(),
            volatility_position_adjustment: true,
            min_position_size_usd: default_min_position_size_usd(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum GridSpacingStrategy {
    Percentage,
    Fixed,
}

fn default_grid_spacing_strategy() -> GridSpacingStrategy {
    GridSpacingStrategy::Percentage
}

impl AutoPositionSizingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if !(10.0..=90.0).contains(&self.balance_reserve_pct) {
            return Err(ConfigError::Invalid(
                "balance_reserve_pct must be between 10 and 90".into(),
            ));
        }
        if !(1.0..=50.0).contains(&self.max_single_position_pct) {
            return Err(ConfigError::Invalid(
                "max_single_position_pct must be between 1 and 50".into(),
            ));
        }
        if self.min_position_size_usd <= 0.0 {
            return Err(ConfigError::Invalid(
                "min_position_size_usd must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ManualPositionSizingConfig {
    #[serde(default = "default_manual_size_per_level")]
    pub size_per_level: f64,
}

fn default_manual_size_per_level() -> f64 {
    0.0001
}

impl ManualPositionSizingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.size_per_level <= 0.0 {
            return Err(ConfigError::Invalid(
                "size_per_level must be positive".into(),
            ));
        }
        Ok(())
    }
}

impl Default for ManualPositionSizingConfig {
    fn default() -> Self {
        Self {
            size_per_level: default_manual_size_per_level(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
pub struct PositionSizingConfig {
    #[serde(default)]
    pub mode: PositionSizingMode,
    #[serde(default)]
    pub auto: AutoPositionSizingConfig,
    #[serde(default)]
    pub manual: ManualPositionSizingConfig,
}

impl PositionSizingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        self.auto.validate()?;
        self.manual.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GridConfig {
    pub symbol: String,
    pub levels: u32,
    #[serde(default)]
    pub price_range: PriceRangeConfig,
    #[serde(default)]
    pub position_sizing: PositionSizingConfig,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            symbol: "BTC".into(),
            levels: 15,
            price_range: PriceRangeConfig::default(),
            position_sizing: PositionSizingConfig::default(),
        }
    }
}

impl GridConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.symbol.is_empty() {
            return Err(ConfigError::Invalid("symbol must be provided".into()));
        }
        if self.levels < 3 || self.levels > 50 {
            return Err(ConfigError::Invalid(
                "levels must be between 3 and 50".into(),
            ));
        }
        self.price_range.validate()?;
        self.position_sizing.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RebalanceConfig {
    #[serde(default = "default_rebalance_threshold")]
    pub price_move_threshold_pct: f64,
    #[serde(default)]
    pub time_based: bool,
    #[serde(default = "default_rebalance_cooldown")]
    pub cooldown_minutes: u32,
    #[serde(default = "default_max_rebalances_per_day")]
    pub max_rebalances_per_day: u32,
}

fn default_rebalance_threshold() -> f64 {
    15.0
}

fn default_rebalance_cooldown() -> u32 {
    30
}

fn default_max_rebalances_per_day() -> u32 {
    10
}

impl RebalanceConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if !(5.0..=50.0).contains(&self.price_move_threshold_pct) {
            return Err(ConfigError::Invalid(
                "price_move_threshold_pct must be between 5 and 50".into(),
            ));
        }
        if self.cooldown_minutes < 1 {
            return Err(ConfigError::Invalid(
                "cooldown_minutes must be at least 1".into(),
            ));
        }
        if self.max_rebalances_per_day < 1 {
            return Err(ConfigError::Invalid(
                "max_rebalances_per_day must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

impl Default for RebalanceConfig {
    fn default() -> Self {
        Self {
            price_move_threshold_pct: default_rebalance_threshold(),
            time_based: false,
            cooldown_minutes: default_rebalance_cooldown(),
            max_rebalances_per_day: default_max_rebalances_per_day(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RiskManagementConfig {
    #[serde(default = "default_max_drawdown_pct")]
    pub max_drawdown_pct: f64,
    #[serde(default = "default_max_position_size_pct")]
    pub max_position_size_pct: f64,
    #[serde(default)]
    pub stop_loss_enabled: bool,
    #[serde(default = "default_stop_loss_pct")]
    pub stop_loss_pct: f64,
    #[serde(default)]
    pub take_profit_enabled: bool,
    #[serde(default = "default_take_profit_pct")]
    pub take_profit_pct: f64,
    #[serde(default)]
    pub rebalance: RebalanceConfig,
}

fn default_max_drawdown_pct() -> f64 {
    15.0
}

fn default_max_position_size_pct() -> f64 {
    30.0
}

fn default_stop_loss_pct() -> f64 {
    5.0
}

fn default_take_profit_pct() -> f64 {
    20.0
}

impl RiskManagementConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if !(5.0..=50.0).contains(&self.max_drawdown_pct) {
            return Err(ConfigError::Invalid(
                "max_drawdown_pct must be between 5 and 50".into(),
            ));
        }
        if !(10.0..=100.0).contains(&self.max_position_size_pct) {
            return Err(ConfigError::Invalid(
                "max_position_size_pct must be between 10 and 100".into(),
            ));
        }
        if self.stop_loss_enabled && !(1.0..=20.0).contains(&self.stop_loss_pct) {
            return Err(ConfigError::Invalid(
                "stop_loss_pct must be between 1 and 20".into(),
            ));
        }
        if self.take_profit_enabled && !(5.0..=100.0).contains(&self.take_profit_pct) {
            return Err(ConfigError::Invalid(
                "take_profit_pct must be between 5 and 100".into(),
            ));
        }
        self.rebalance.validate()?;
        Ok(())
    }
}

impl Default for RiskManagementConfig {
    fn default() -> Self {
        Self {
            max_drawdown_pct: default_max_drawdown_pct(),
            max_position_size_pct: default_max_position_size_pct(),
            stop_loss_enabled: false,
            stop_loss_pct: default_stop_loss_pct(),
            take_profit_enabled: false,
            take_profit_pct: default_take_profit_pct(),
            rebalance: RebalanceConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct MonitoringConfig {
    #[serde(default = "default_log_level")]
    pub log_level: LogLevel,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warning => "warn",
            LogLevel::Error => "error",
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

fn default_log_level() -> LogLevel {
    LogLevel::Info
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
        }
    }
}

impl MonitoringConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        match self.log_level {
            LogLevel::Debug | LogLevel::Info | LogLevel::Warning | LogLevel::Error => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ExchangeConfig {
    #[serde(rename = "type")]
    pub exchange_type: String,
    #[serde(default = "default_true")]
    pub testnet: bool,
}

impl Default for ExchangeConfig {
    fn default() -> Self {
        Self {
            exchange_type: "hyperliquid".into(),
            testnet: true,
        }
    }
}

impl ExchangeConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.exchange_type.trim().is_empty() {
            return Err(ConfigError::Invalid(
                "exchange type must be provided".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct BotConfig {
    pub name: String,
    pub active: bool,
    #[serde(default)]
    pub exchange: ExchangeConfig,
    #[serde(default)]
    pub account: AccountConfig,
    #[serde(default)]
    pub grid: GridConfig,
    #[serde(default)]
    pub risk_management: RiskManagementConfig,
    #[serde(default)]
    pub monitoring: MonitoringConfig,
    #[serde(skip)]
    pub loaded_at: Option<DateTime<Utc>>,
}

impl BotConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(ConfigError::Read)?;
        Self::load_from_str(&content)
    }

    pub fn load_from_str(content: &str) -> Result<Self, ConfigError> {
        let mut config: BotConfig = serde_yaml::from_str(content).map_err(ConfigError::Parse)?;
        config.validate()?;
        config.loaded_at = Some(Utc::now());
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.name.trim().is_empty() {
            return Err(ConfigError::Invalid("name must be provided".into()));
        }
        self.exchange.validate()?;
        self.account.validate()?;
        self.grid.validate()?;
        self.risk_management.validate()?;
        self.monitoring.validate()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_config_from_yaml() {
        let yaml = r#"
name: Test Bot
active: true
exchange:
  type: hyperliquid
  testnet: true
account:
  max_allocation_pct: 15.0
  risk_level: conservative
grid:
  symbol: BTC
  levels: 12
risk_management:
  max_drawdown_pct: 20.0
monitoring:
  log_level: INFO
"#;
        let config = BotConfig::load_from_str(yaml).unwrap();
        assert_eq!(config.name, "Test Bot");
        assert!(config.active);
        assert_eq!(config.account.max_allocation_pct, 15.0);
        assert_eq!(config.grid.levels, 12);
        assert!(config.loaded_at.is_some());
    }

    #[test]
    fn rejects_invalid_allocation() {
        let yaml = r#"
name: Bad Bot
active: true
account:
  max_allocation_pct: 0.5
"#;
        let result = BotConfig::load_from_str(yaml);
        assert!(matches!(result, Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn rejects_invalid_manual_range() {
        let yaml = r#"
name: Manual Bot
active: true
grid:
  symbol: BTC
  levels: 10
  price_range:
    mode: manual
    manual:
      min: 120000.0
      max: 110000.0
"#;
        let result = BotConfig::load_from_str(yaml);
        assert!(
            matches!(result, Err(ConfigError::Invalid(message)) if message.contains("manual min"))
        );
    }

    #[test]
    fn log_level_display_matches_env_strings() {
        assert_eq!(LogLevel::Debug.as_str(), "debug");
        assert_eq!(LogLevel::Info.as_str(), "info");
        assert_eq!(LogLevel::Warning.as_str(), "warn");
        assert_eq!(LogLevel::Error.as_str(), "error");
        assert_eq!(LogLevel::Info.to_string(), "info");
    }
}
