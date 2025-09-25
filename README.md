## Extensible grid trading bot for [Hyperliquid DEX](https://hyperliquid.xyz)

> ⚠️ This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Never trade with funds you cannot afford to lose. Always validate strategies on testnet before live deployment.

The project now ships as a pure Rust codebase focused on production-grade automation and reproducible research workflows.

## Quick start

### Prerequisites
- Rust toolchain (install with [rustup](https://rustup.rs/))
- Hyperliquid testnet credentials and funded test account

### Install

```bash
git clone https://github.com/chainstacklabs/hyperliquid-trading-bot
cd hyperliquid-trading-bot
cargo build
```

### Configure the environment
Create a `.env` file with your credentials:

```bash
HYPERLIQUID_TESTNET_PRIVATE_KEY=0xYOUR_PRIVATE_KEY
HYPERLIQUID_TESTNET=true
```

### Run the live engine

```bash
# Run the default conservative BTC configuration
cargo run

# Run a specific bot configuration
cargo run -- bots/your_config.yaml
```

`cargo run` blocks until Ctrl+C and performs graceful shutdown (disconnects market data, cancels outstanding orders, and stops strategies).

## Configuration

Bot configurations use YAML and follow the same schema across live trading and backtests.

```yaml
name: "btc_conservative_clean"
active: true

account:
  max_allocation_pct: 10.0

grid:
  symbol: "BTC"
  levels: 10
  price_range:
    mode: "auto"
    auto:
      range_pct: 5.0

risk_management:
  stop_loss_enabled: false
  stop_loss_pct: 8.0
  take_profit_enabled: false
  take_profit_pct: 25.0
  max_drawdown_pct: 15.0
  max_position_size_pct: 40.0
  rebalance:
    price_move_threshold_pct: 12.0

monitoring:
  log_level: "INFO"
```

## Nautilus-style backtesting

The Rust-native backtesting harness reuses the live grid strategy, exchanges, and `nautilus-backtest` data iterator so signals behave exactly as they do in production while keeping Python dependencies out of the toolchain.

```rust
use chrono::{Duration, TimeZone, Utc};
use hyperliquid_bot::backtesting::{run_backtest, PriceSample};
use hyperliquid_bot::config::BotConfig;

let config = BotConfig::load_from_str(include_str!("bots/btc_conservative.yaml"))?;
let series = (0..5)
    .map(|i| PriceSample::new(Utc.timestamp_opt(i * 60, 0).single().unwrap(), 100.0 + i as f64))
    .collect::<Vec<_>>();
let result = run_backtest(&config, 5000.0, &series)?;
println!("final account value: {:.2}", result.final_value);
```

To execute the built-in regression scenarios:

```bash
cargo test --test backtesting
```

`run_backtest` returns execution history, cash balance, residual position, and marked-to-market equity so custom analytics can be layered on top.

## Development workflow

```bash
cargo fmt        # Format the workspace
cargo clippy     # Lint with additional checks
cargo test       # Execute unit and integration tests
```

The repository follows SOLID design principles with dependency injection and async I/O throughout the trading engine. Backtests should mirror live configurations; avoid duplicating strategy parameters in code.
# hl-trading-bot
