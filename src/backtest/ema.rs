use anyhow::Result;
use chrono::{DateTime, Utc};

use crate::backtest::{BacktestResult, TradeExecution};
use crate::config::BotConfig;
use crate::interfaces::{MarketData, Position, SignalType, TradingStrategy};
use crate::strategies::ema::EmaStrategy;

pub fn run_backtest(
    config: &BotConfig,
    initial_cash: f64,
    prices: &[(DateTime<Utc>, f64)],
) -> Result<BacktestResult> {
    let mut strategy = EmaStrategy::new(config)?;
    strategy.start();

    let mut cash = initial_cash;
    let mut position = 0.0;
    let mut trades = Vec::new();
    let mut equity_peak = initial_cash;
    let mut max_drawdown = 0.0;

    for (timestamp, price) in prices {
        let market_data = MarketData {
            asset: config.grid.symbol.clone(),
            price: *price,
            volume_24h: 1000.0,
            timestamp: *timestamp,
            bid: Some(price * 0.999),
            ask: Some(price * 1.001),
            volatility: None,
        };

        let positions = if position != 0.0 {
            vec![Position {
                asset: config.grid.symbol.clone(),
                size: position,
                entry_price: *price,
                current_value: position * price,
                unrealized_pnl: 0.0,
                timestamp: *timestamp,
            }]
        } else {
            vec![]
        };

        let signals = strategy.generate_signals(&market_data, &positions, cash)?;

        for signal in signals {
            let (new_position, new_cash, executed_size) = match signal.signal_type {
                SignalType::Buy => {
                    let trade_value = signal.size * price;
                    if cash >= trade_value {
                        (position + signal.size, cash - trade_value, signal.size)
                    } else {
                        (position, cash, 0.0)
                    }
                }
                SignalType::Sell => {
                    let sell_size = if position >= signal.size {
                        signal.size
                    } else {
                        position
                    };
                    if sell_size > 0.0 {
                        (position - sell_size, cash + sell_size * price, sell_size)
                    } else {
                        (position, cash, 0.0)
                    }
                }
                _ => (position, cash, 0.0),
            };

            if executed_size > 0.0 {
                trades.push(TradeExecution {
                    timestamp: *timestamp,
                    asset: config.grid.symbol.clone(),
                    side: format!("{:?}", signal.signal_type),
                    quantity: executed_size,
                    price: *price,
                    fee: 0.0,
                    pnl: 0.0,
                });

                strategy.on_trade_executed(&signal, *price, executed_size)?;
                position = new_position;
                cash = new_cash;
            }
        }

        let portfolio_value = cash + position * price;
        if portfolio_value > equity_peak {
            equity_peak = portfolio_value;
        } else {
            let drawdown = (equity_peak - portfolio_value) / equity_peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
    }

    let final_price = prices.last().map(|(_, p)| *p).unwrap_or(0.0);
    let final_value = cash + position * final_price;
    let total_return = (final_value - initial_cash) / initial_cash;
    let sharpe_ratio = if max_drawdown > 0.0 {
        total_return / max_drawdown.sqrt()
    } else {
        0.0
    };

    Ok(BacktestResult {
        initial_balance: initial_cash,
        final_balance: final_value,
        total_pnl: final_value - initial_cash,
        total_return,
        max_drawdown,
        sharpe_ratio,
        total_trades: trades.len(),
        winning_trades: trades.len() / 2,
        losing_trades: trades.len() / 2,
        trades,
    })
}