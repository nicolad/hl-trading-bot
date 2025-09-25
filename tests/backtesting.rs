use chrono::{Duration, TimeZone, Utc};

use hyperliquid_bot::backtesting::{PriceSample, run_backtest};
use hyperliquid_bot::config::BotConfig;

fn sample_config() -> BotConfig {
    let yaml = r#"
name: Backtest Bot
active: true
exchange:
  type: hyperliquid
  testnet: true
grid:
  symbol: BTC
  levels: 3
  price_range:
    mode: manual
    manual:
      min: 90.0
      max: 110.0
  position_sizing:
    mode: manual
    manual:
      size_per_level: 1000.0
risk_management:
  rebalance:
    price_move_threshold_pct: 12.0
monitoring:
  log_level: INFO
"#;
    BotConfig::load_from_str(yaml).expect("config should load")
}

fn build_series(prices: &[f64]) -> Vec<PriceSample> {
    let start = Utc.timestamp_opt(0, 0).single().unwrap();
    prices
        .iter()
        .enumerate()
        .map(|(index, price)| PriceSample::new(start + Duration::minutes(index as i64), *price))
        .collect()
}

#[test]
fn grid_strategy_realizes_partial_cycle_profit() {
    let config = sample_config();
    let series = build_series(&[100.0, 95.0, 90.0, 100.0, 110.0]);
    let result = run_backtest(&config, 5000.0, &series).expect("backtest success");
    assert_eq!(result.trades.len(), 3);
    assert!((result.final_value - 5327.763819007356).abs() < 1e-6);
    assert!((result.cash - 4000.0).abs() < 1e-6);
    assert!((result.position - 12.070580172794143).abs() < 1e-6);
}

#[test]
fn grid_strategy_respects_cash_constraints() {
    let config = sample_config();
    let series = build_series(&[100.0, 95.0, 90.0, 100.0, 110.0]);
    let result = run_backtest(&config, 1000.0, &series).expect("backtest success");
    assert_eq!(result.trades.len(), 2);
    assert!((result.final_value - 1105.5415967851334).abs() < 1e-6);
    assert!((result.cash - 1000.0).abs() < 1e-6);
    assert!((result.position - 0.9594690616830306).abs() < 1e-6);
}
