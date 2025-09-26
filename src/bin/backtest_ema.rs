use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use tracing_subscriber::EnvFilter;

use hyperliquid_bot::{
    config::BotConfig,
    backtest::run_ema_backtest,
};

fn init_tracing() -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
    Ok(())
}

fn generate_sample_prices() -> Vec<(DateTime<Utc>, f64)> {
    let start_time = Utc::now() - Duration::days(30);
    let mut prices = Vec::new();
    let base_price = 100000.0;
    
    // Generate 30 days of hourly price data with some trend and noise
    for i in 0..(30 * 24) {
        let timestamp = start_time + Duration::hours(i);
        
        // Create a trending price pattern with some noise
        let trend = (i as f64 / (30.0 * 24.0)) * 5000.0; // Upward trend
        let cycle = (i as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin() * 2000.0; // Daily cycle
        let noise = (i as f64 * 0.1).sin() * (i as f64 * 0.17).cos() * 500.0; // Random-ish noise
        
        let price = base_price + trend + cycle + noise;
        prices.push((timestamp, price));
    }
    
    prices
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;
    
    // Create a sample EMA configuration
    let config_yaml = r#"
name: "ema_backtest"
active: true

exchange:
  type: "hyperliquid"
  testnet: true

account:
  max_allocation_pct: 10.0

strategy:
  kind: "ema"
  ema:
    fast_period: 12
    slow_period: 26
    order_size: 0.01

grid:
  symbol: "BTC"
  levels: 10
  price_range:
    mode: "auto"
    auto:
      range_pct: 5.0

risk_management:
  stop_loss_enabled: false
  take_profit_enabled: false
  max_drawdown_pct: 15.0
  max_position_size_pct: 40.0
  rebalance:
    price_move_threshold_pct: 12.0

monitoring:
  log_level: "INFO"
"#;

    let config = BotConfig::load_from_str(config_yaml)?;
    let initial_cash = 10000.0;
    let prices = generate_sample_prices();
    
    println!("Running EMA backtest with {} price samples", prices.len());
    println!("Initial cash: ${:.2}", initial_cash);
    println!("Strategy: EMA({})/EMA({})", config.strategy.ema.fast_period, config.strategy.ema.slow_period);
    println!("Order size: {} BTC", config.strategy.ema.order_size);
    
    let result = run_ema_backtest(&config, initial_cash, &prices)?;
    
    println!("\n=== BACKTEST RESULTS ===");
    println!("Initial Balance: ${:.2}", result.initial_balance);
    println!("Final Balance: ${:.2}", result.final_balance);
    println!("Total P&L: ${:.2}", result.total_pnl);
    println!("Total Return: {:.2}%", result.total_return * 100.0);
    println!("Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
    println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("Total Trades: {}", result.total_trades);
    println!("Winning Trades: {}", result.winning_trades);
    println!("Losing Trades: {}", result.losing_trades);
    
    if !result.trades.is_empty() {
        println!("\n=== FIRST 5 TRADES ===");
        for (i, trade) in result.trades.iter().take(5).enumerate() {
            println!("{}. {} {} BTC @ ${:.2} on {}", 
                i + 1, trade.side, trade.quantity, trade.price, trade.timestamp);
        }
    }
    
    Ok(())
}