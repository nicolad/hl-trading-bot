## Hyperliquid strategy toolkit (Rust)

> ⚠️ This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Never trade with funds you cannot afford to lose. Always validate strategies on testnet before live deployment.

This repository is centred on a single loop: capture on-chain behaviour, mine repeatable edges, and deploy bots on Hyperliquid. The sections below focus on the binaries and configuration you need to move from raw data to trading strategies.

## Setup

```bash
git clone https://github.com/chainstacklabs/hyperliquid-trading-bot
cd hyperliquid-trading-bot
cargo build
```

Dependencies:

- Recent stable Rust toolchain (install with [rustup](https://rustup.rs/)).
- Hyperliquid testnet credentials stored in `.env`:

  ```bash
  HYPERLIQUID_TESTNET_PRIVATE_KEY=0xYOUR_PRIVATE_KEY
  HYPERLIQUID_TESTNET=true
  ```

## 1. Capture leaderboard snapshots

`snapshot_pipeline` reads `leaderboard.json`, pulls detailed account snapshots for every address, and writes Polars-friendly datasets.

```bash
# leaderboard.json must exist in the repo root
cargo run --bin snapshot_pipeline
```

Artifacts (written to the repository root):

- `snapshots.ndjson` — streamable snapshots for Polars (`pl.read_ndjson`).
- `trading_features.csv` — flattened feature matrix that powers clustering.
- `snapshots_raw.json` — raw JSON array for manual inspection or replay tooling.

Adjust API endpoints, rate limits, or output paths in `src/bin/snapshot_pipeline.rs` if required.

## 2. Mine account-level patterns

`patterns` consumes `trading_features.csv`, learns behavioural clusters, and emits basic trade recommendations.

```bash
cargo run --bin patterns
```

Runtime output shows each cluster with sample counts, mean returns, and whether it triggered a trade bias. Two files accompany the console log:

- `strategy_patterns.json` — per-cluster statistics (mean/median returns, win rate, leverage proxies, confidence).
- `strategy_signals.csv` — actionable addresses with expected return and confidence for clusters that pass the filters.

Bias thresholds and output paths live in `src/patterns.rs` (`Config::default`). Tune them before running if you want tighter confidence gates or custom locations.

## 3. Deploy a bot

When a configuration looks promising, launch it via the standard entry point:

```bash
# Discover the first active bot configuration
cargo run

# Pin a specific configuration
cargo run -- bots/btc_conservative.yaml
```

The process handles graceful shutdown (cancel orders, disconnect feeds) on Ctrl+C.

## Configuration files

YAML files in `bots/` define allocation limits, grid parameters, and risk rules. The same schema is used for both live execution and backtests.

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

## Development checklist

```bash
cargo fmt
cargo clippy
cargo test
```

Run every change against the Hyperliquid testnet sandbox before deploying capital.

# hl-trading-bot
