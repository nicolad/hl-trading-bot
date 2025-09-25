use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use nautilus_backtest::data_iterator::BacktestDataIterator;
use nautilus_core::UnixNanos;
use nautilus_model::{
    data::{Data, TradeTick},
    enums::AggressorSide,
    identifiers::{InstrumentId, TradeId},
    types::{Price, Quantity},
};

use crate::{
    config::BotConfig,
    interfaces::{MarketData, OrderSide, Position, SignalType, TradingSignal, TradingStrategy},
    strategies,
};

#[derive(Clone, Debug)]
pub struct PriceSample {
    pub timestamp: DateTime<Utc>,
    pub price: f64,
}

impl PriceSample {
    pub fn new(timestamp: DateTime<Utc>, price: f64) -> Self {
        Self { timestamp, price }
    }
}

#[derive(Clone, Debug)]
pub struct BacktestResult {
    pub final_value: f64,
    pub cash: f64,
    pub position: f64,
    pub trades: Vec<TradeExecution>,
}

#[derive(Clone, Debug)]
pub struct TradeExecution {
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub size: f64,
    pub side: OrderSide,
}

const PRICE_PRECISION: u8 = 5;
const SIZE_PRECISION: u8 = 4;

pub fn run_backtest(
    config: &BotConfig,
    initial_cash: f64,
    samples: &[PriceSample],
) -> Result<BacktestResult> {
    if samples.is_empty() {
        bail!("price samples required");
    }
    let processed_samples = normalize_samples(config, samples)?;
    if processed_samples.is_empty() {
        bail!("no price samples after normalization");
    }
    let mut strategy = strategies::create_strategy(config)?;
    strategy.start();
    let mut state = BacktestState::new(initial_cash);
    let mut orders = Vec::new();
    let symbol = config.grid.symbol.clone();
    let mut last_price = processed_samples[0].price;
    for sample in &processed_samples {
        last_price = sample.price;
        fill_orders(sample, &mut orders, &mut state, strategy.as_mut())?;
        let positions = state.positions(&symbol, sample.timestamp, sample.price);
        let market_data = MarketData {
            asset: symbol.clone(),
            price: sample.price,
            volume_24h: 0.0,
            timestamp: sample.timestamp,
            bid: Some(sample.price),
            ask: Some(sample.price),
            volatility: None,
        };
        let signals = strategy.generate_signals(&market_data, &positions, state.cash)?;
        process_signals(
            sample,
            &mut orders,
            &mut state,
            strategy.as_mut(),
            &market_data,
            signals,
        )?;
    }
    strategy.stop();
    let final_value = state.cash + state.position * last_price;
    Ok(BacktestResult {
        final_value,
        cash: state.cash,
        position: state.position,
        trades: state.trades,
    })
}

fn normalize_samples(config: &BotConfig, samples: &[PriceSample]) -> Result<Vec<PriceSample>> {
    let instrument = instrument_id_from_symbol(&config.grid.symbol);
    let data = convert_samples_to_data(&instrument, samples)?;
    let mut iterator = BacktestDataIterator::new();
    iterator.add_data("grid_prices", data, false);
    let mut normalized = Vec::with_capacity(samples.len());
    while let Some(entry) = iterator.next() {
        match entry {
            Data::Trade(trade) => {
                let timestamp = trade.ts_event.to_datetime_utc();
                let price = f64::from(trade.price);
                normalized.push(PriceSample::new(timestamp, price));
            }
            Data::Bar(bar) => {
                let timestamp = bar.ts_event.to_datetime_utc();
                let price = f64::from(bar.close);
                normalized.push(PriceSample::new(timestamp, price));
            }
            _ => {}
        }
    }
    Ok(normalized)
}

fn convert_samples_to_data(
    instrument_id: &InstrumentId,
    samples: &[PriceSample],
) -> Result<Vec<Data>> {
    let mut data = Vec::with_capacity(samples.len());
    let mut previous_price = None;
    for (index, sample) in samples.iter().enumerate() {
        let timestamp = to_unix_nanos(sample.timestamp)
            .with_context(|| format!("invalid timestamp at index {index}"))?;
        let price = Price::new(sample.price, PRICE_PRECISION);
        let size = Quantity::non_zero(1.0, SIZE_PRECISION);
        let side = determine_aggressor(previous_price, sample.price);
        let trade_id = TradeId::new(format!("BT{:06}", index));
        let trade = TradeTick::new(
            *instrument_id,
            price,
            size,
            side,
            trade_id,
            timestamp,
            timestamp,
        );
        data.push(Data::Trade(trade));
        previous_price = Some(sample.price);
    }
    Ok(data)
}

fn to_unix_nanos(timestamp: DateTime<Utc>) -> Result<UnixNanos> {
    let nanos = timestamp
        .timestamp_nanos_opt()
        .context("timestamp out of range")?;
    if nanos < 0 {
        bail!("timestamps before UNIX epoch are unsupported");
    }
    Ok(UnixNanos::new(nanos as u64))
}

fn determine_aggressor(previous: Option<f64>, current: f64) -> AggressorSide {
    match previous {
        Some(prev) if current > prev => AggressorSide::Buyer,
        Some(prev) if current < prev => AggressorSide::Seller,
        Some(_) => AggressorSide::NoAggressor,
        None => AggressorSide::NoAggressor,
    }
}

fn instrument_id_from_symbol(symbol: &str) -> InstrumentId {
    let cleaned = symbol.trim().replace(' ', "").to_uppercase();
    InstrumentId::from(format!("{}.HYPERLIQUID", cleaned))
}

struct BacktestState {
    cash: f64,
    position: f64,
    average_price: f64,
    trades: Vec<TradeExecution>,
}

impl BacktestState {
    fn new(initial_cash: f64) -> Self {
        Self {
            cash: initial_cash,
            position: 0.0,
            average_price: 0.0,
            trades: Vec::new(),
        }
    }

    fn positions(&self, symbol: &str, timestamp: DateTime<Utc>, price: f64) -> Vec<Position> {
        if self.position.abs() <= f64::EPSILON {
            Vec::new()
        } else {
            vec![Position {
                asset: symbol.to_string(),
                size: self.position,
                entry_price: self.average_price,
                current_value: self.position.abs() * price,
                unrealized_pnl: (price - self.average_price) * self.position,
                timestamp,
            }]
        }
    }
}

struct OpenOrder {
    signal: TradingSignal,
    price: f64,
    remaining: f64,
    side: OrderSide,
}

fn fill_orders(
    sample: &PriceSample,
    orders: &mut Vec<OpenOrder>,
    state: &mut BacktestState,
    strategy: &mut dyn TradingStrategy,
) -> Result<()> {
    let mut index = 0;
    while index < orders.len() {
        let should_fill = match orders[index].side {
            OrderSide::Buy => sample.price <= orders[index].price + 1e-9,
            OrderSide::Sell => sample.price + 1e-9 >= orders[index].price,
        };
        if !should_fill {
            index += 1;
            continue;
        }
        let can_fill = match orders[index].side {
            OrderSide::Buy => state.cash + 1e-9 >= orders[index].price * orders[index].remaining,
            OrderSide::Sell => state.position + 1e-9 >= orders[index].remaining,
        };
        if !can_fill {
            index += 1;
            continue;
        }
        execute_fill(
            orders[index].side,
            orders[index].price,
            orders[index].remaining,
            sample.timestamp,
            state,
            &orders[index].signal,
            strategy,
        )?;
        orders.remove(index);
    }
    Ok(())
}

fn process_signals(
    sample: &PriceSample,
    orders: &mut Vec<OpenOrder>,
    state: &mut BacktestState,
    strategy: &mut dyn TradingStrategy,
    market_data: &MarketData,
    signals: Vec<TradingSignal>,
) -> Result<()> {
    for signal in signals {
        match signal.signal_type {
            SignalType::Buy | SignalType::Sell => {
                let side = if signal.signal_type == SignalType::Buy {
                    OrderSide::Buy
                } else {
                    OrderSide::Sell
                };
                if signal.size <= f64::EPSILON {
                    continue;
                }
                if let Some(price) = signal.price {
                    if price <= 0.0 {
                        continue;
                    }
                    let size = signal.size;
                    orders.push(OpenOrder {
                        signal,
                        price,
                        remaining: size,
                        side,
                    });
                } else {
                    execute_fill(
                        side,
                        market_data.price,
                        signal.size,
                        sample.timestamp,
                        state,
                        &signal,
                        strategy,
                    )?;
                }
            }
            SignalType::Close => {
                if signal
                    .metadata
                    .get("action")
                    .and_then(|value| value.as_str())
                    == Some("cancel_all")
                {
                    orders.clear();
                }
            }
            SignalType::Hold => {}
        }
    }
    Ok(())
}

fn execute_fill(
    side: OrderSide,
    price: f64,
    size: f64,
    timestamp: DateTime<Utc>,
    state: &mut BacktestState,
    signal: &TradingSignal,
    strategy: &mut dyn TradingStrategy,
) -> Result<()> {
    match side {
        OrderSide::Buy => {
            let cost = price * size;
            state.cash -= cost;
            if state.cash < 0.0 {
                state.cash = 0.0;
            }
            let previous_value = state.average_price * state.position.max(0.0);
            state.position += size;
            if state.position > f64::EPSILON {
                state.average_price = (previous_value + cost) / state.position;
            } else {
                state.average_price = 0.0;
            }
        }
        OrderSide::Sell => {
            state.cash += price * size;
            state.position -= size;
            if state.position <= f64::EPSILON {
                state.position = state.position.max(0.0);
                state.average_price = 0.0;
            }
        }
    }
    state.trades.push(TradeExecution {
        timestamp,
        price,
        size,
        side,
    });
    strategy.on_trade_executed(signal, price, size)?;
    Ok(())
}
