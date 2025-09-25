use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::Result;
use chrono::Utc;
use futures::executor::block_on;
use serde_json::json;
use tokio::sync::Mutex;

use crate::{
    config::BotConfig,
    events::{Event, EventBus, EventType},
    exchange::HyperliquidPublicExchange,
    interfaces::{
        ExchangeAdapter, MarketData, MarketDataProvider, Order, OrderSide, OrderType, SignalType,
        TradingSignal, TradingStrategy,
    },
    market_data::HyperliquidMarketData,
    risk::{RiskAction, RiskEvent, RiskManager},
    strategies::create_strategy,
};

struct EngineStats {
    executed_trades: usize,
    total_pnl: f64,
    pending_orders: HashMap<String, Order>,
}

impl Default for EngineStats {
    fn default() -> Self {
        Self {
            executed_trades: 0,
            total_pnl: 0.0,
            pending_orders: HashMap::new(),
        }
    }
}

pub struct TradingEngine {
    config: BotConfig,
    strategy: Arc<Mutex<Box<dyn TradingStrategy>>>,
    exchange: Arc<HyperliquidPublicExchange>,
    market_data: Arc<Mutex<HyperliquidMarketData>>,
    risk_manager: Arc<Mutex<RiskManager>>,
    stats: Arc<Mutex<EngineStats>>,
    running: Arc<AtomicBool>,
    event_bus: EventBus,
}

impl TradingEngine {
    pub fn new(config: BotConfig) -> Result<Self> {
        let strategy = create_strategy(&config)?;
        let exchange = Arc::new(HyperliquidPublicExchange::new(&config)?);
        let market_data = Arc::new(Mutex::new(HyperliquidMarketData::new(&config)));
        let risk_manager = Arc::new(Mutex::new(RiskManager::new(&config)));
        Ok(Self {
            config,
            strategy: Arc::new(Mutex::new(strategy)),
            exchange,
            market_data,
            risk_manager,
            stats: Arc::new(Mutex::new(EngineStats::default())),
            running: Arc::new(AtomicBool::new(false)),
            event_bus: EventBus::default(),
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        self.exchange.connect().await?;
        self.market_data.lock().await.connect().await?;
        Ok(())
    }

    pub async fn start(&self) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.running.store(true, Ordering::SeqCst);
        self.strategy.lock().await.start();
        self.subscribe_market_data().await?;
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        self.strategy.lock().await.stop();
        self.market_data.lock().await.disconnect().await?;
        self.exchange.disconnect().await?;
        Ok(())
    }

    async fn subscribe_market_data(&self) -> Result<()> {
        let symbol = self.config.grid.symbol.clone();
        let running = self.running.clone();
        let strategy = self.strategy.clone();
        let exchange = self.exchange.clone();
        let risk_manager = self.risk_manager.clone();
        let stats = self.stats.clone();
        let event_bus = self.event_bus.clone();
        let mut market_data = self.market_data.lock().await;
        market_data
            .subscribe_price_updates(&symbol, move |data| {
                let running = running.clone();
                let strategy = strategy.clone();
                let exchange = exchange.clone();
                let risk_manager = risk_manager.clone();
                let stats = stats.clone();
                let event_bus = event_bus.clone();
                Box::pin(async move {
                    if !running.load(Ordering::SeqCst) {
                        return Ok(());
                    }
                    Self::handle_price_update(
                        strategy,
                        exchange,
                        risk_manager,
                        stats,
                        event_bus,
                        data,
                    )
                    .await
                })
            })
            .await?;
        Ok(())
    }

    async fn handle_price_update(
        strategy: Arc<Mutex<Box<dyn TradingStrategy>>>,
        exchange: Arc<HyperliquidPublicExchange>,
        risk_manager: Arc<Mutex<RiskManager>>,
        stats: Arc<Mutex<EngineStats>>,
        event_bus: EventBus,
        market_data: MarketData,
    ) -> Result<()> {
        let positions = exchange.get_positions().await?;
        let balance = exchange.get_balance("USD").await?.available;
        let signals = {
            let mut guard = strategy.lock().await;
            guard.generate_signals(&market_data, &positions, balance)?
        };
        let metrics_value = exchange.get_account_metrics().await?;
        let metrics = engine_utils::AccountMetricsConverter::from_json(&metrics_value);
        let risk_events = {
            let mut guard = risk_manager.lock().await;
            guard.evaluate(&positions, &market_data, &metrics)
        };
        for event in risk_events {
            Self::execute_risk_action(&exchange, &event_bus, &event).await?;
        }
        for signal in signals {
            Self::execute_signal(&strategy, &exchange, &stats, &event_bus, signal).await?;
        }
        Ok(())
    }

    async fn execute_risk_action(
        exchange: &Arc<HyperliquidPublicExchange>,
        event_bus: &EventBus,
        event: &RiskEvent,
    ) -> Result<()> {
        match event.action {
            RiskAction::ClosePosition => {
                exchange.close_position(&event.asset, None).await?;
            }
            RiskAction::ReducePosition => {
                exchange.close_position(&event.asset, Some(0.5)).await?;
            }
            RiskAction::CancelOrders => {
                exchange.cancel_all_orders().await?;
            }
            RiskAction::PauseTrading => {}
            RiskAction::EmergencyExit => {
                exchange.cancel_all_orders().await?;
                let positions = exchange.get_positions().await?;
                for position in positions {
                    exchange.close_position(&position.asset, None).await?;
                }
            }
            RiskAction::None => {}
        }
        event_bus.emit(Event {
            event_type: EventType::System,
            timestamp: Utc::now(),
            data: serde_json::json!({
                "rule": event.rule_name,
                "action": format!("{:?}", event.action),
                "reason": event.reason,
            }),
            source: Some("risk".into()),
        });
        Ok(())
    }

    async fn execute_signal(
        strategy: &Arc<Mutex<Box<dyn TradingStrategy>>>,
        exchange: &Arc<HyperliquidPublicExchange>,
        stats: &Arc<Mutex<EngineStats>>,
        event_bus: &EventBus,
        signal: TradingSignal,
    ) -> Result<()> {
        match signal.signal_type {
            SignalType::Buy | SignalType::Sell => {
                let order_side = if signal.signal_type == SignalType::Buy {
                    OrderSide::Buy
                } else {
                    OrderSide::Sell
                };
                let order = Order::new_local(
                    signal.asset.clone(),
                    order_side,
                    signal.size,
                    if signal.price.is_some() {
                        OrderType::Limit
                    } else {
                        OrderType::Market
                    },
                    signal.price,
                );
                let placed = exchange.place_order(order).await?;
                {
                    let mut guard = strategy.lock().await;
                    guard.on_trade_executed(
                        &signal,
                        placed.average_fill_price,
                        placed.filled_size,
                    )?;
                }
                {
                    let mut guard = stats.lock().await;
                    guard.executed_trades += 1;
                    guard
                        .pending_orders
                        .insert(placed.id.to_string(), placed.clone());
                }
                event_bus.emit(Event {
                    event_type: EventType::OrderPlaced,
                    timestamp: Utc::now(),
                    data: serde_json::to_value(&placed).unwrap_or(serde_json::Value::Null),
                    source: Some("engine".into()),
                });
            }
            SignalType::Close => {
                if signal.metadata.get("action").and_then(|v| v.as_str()) == Some("cancel_all") {
                    exchange.cancel_all_orders().await?;
                }
            }
            SignalType::Hold => {}
        }
        Ok(())
    }

    pub fn get_status(&self) -> Result<serde_json::Value> {
        let stats = block_on(self.stats.lock());
        let strategy_yaml = block_on(self.strategy.lock()).get_status();
        let strategy_status =
            serde_json::to_value(strategy_yaml).unwrap_or(serde_json::Value::Null);
        Ok(json!({
            "running": self.running.load(Ordering::SeqCst),
            "strategy": strategy_status,
            "executed_trades": stats.executed_trades,
            "pending_orders": stats.pending_orders.len(),
            "total_pnl": stats.total_pnl,
        }))
    }

    pub fn event_bus(&self) -> EventBus {
        self.event_bus.clone()
    }
}

mod engine_utils {
    use crate::risk::AccountMetrics;

    pub struct AccountMetricsConverter;

    impl AccountMetricsConverter {
        pub fn from_json(value: &serde_json::Value) -> AccountMetrics {
            AccountMetrics {
                total_value: value
                    .get("total_value")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                total_pnl: value
                    .get("total_pnl")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                unrealized_pnl: value
                    .get("unrealized_pnl")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                realized_pnl: value
                    .get("realized_pnl")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                drawdown_pct: value
                    .get("drawdown_pct")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                positions_count: value
                    .get("positions_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize,
                largest_position_pct: value
                    .get("largest_position_pct")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
            }
        }
    }
}
