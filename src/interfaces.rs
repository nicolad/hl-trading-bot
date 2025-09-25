use std::fmt;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
    Close,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct TradingSignal {
    pub signal_type: SignalType,
    pub asset: String,
    pub size: f64,
    pub price: Option<f64>,
    pub reason: Option<String>,
    #[serde(default)]
    pub metadata: serde_yaml::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct MarketData {
    pub asset: String,
    pub price: f64,
    pub volume_24h: f64,
    pub timestamp: DateTime<Utc>,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub volatility: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Position {
    pub asset: String,
    pub size: f64,
    pub entry_price: f64,
    pub current_value: f64,
    pub unrealized_pnl: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Balance {
    pub asset: String,
    pub available: f64,
    pub locked: f64,
    pub total: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct MarketInfo {
    pub symbol: String,
    pub base_asset: String,
    pub quote_asset: String,
    pub min_order_size: f64,
    pub price_precision: u32,
    pub size_precision: u32,
    pub is_active: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide {
    Buy,
    Sell,
}

impl fmt::Display for OrderSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buy => write!(f, "buy"),
            Self::Sell => write!(f, "sell"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OrderType {
    Market,
    Limit,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OrderStatus {
    Pending,
    Submitted,
    Filled,
    PartiallyFilled,
    Cancelled,
    Rejected,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Order {
    pub id: Uuid,
    pub asset: String,
    pub side: OrderSide,
    pub size: f64,
    pub order_type: OrderType,
    pub price: Option<f64>,
    pub status: OrderStatus,
    pub filled_size: f64,
    pub average_fill_price: f64,
    pub exchange_order_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl Order {
    pub fn new_local(
        asset: String,
        side: OrderSide,
        size: f64,
        order_type: OrderType,
        price: Option<f64>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            asset,
            side,
            size,
            order_type,
            price,
            status: OrderStatus::Pending,
            filled_size: 0.0,
            average_fill_price: 0.0,
            exchange_order_id: None,
            created_at: Utc::now(),
        }
    }
}

#[async_trait]
pub trait TradingStrategy: Send + Sync {
    fn generate_signals(
        &mut self,
        market_data: &MarketData,
        positions: &[Position],
        balance: f64,
    ) -> Result<Vec<TradingSignal>>;

    fn on_trade_executed(
        &mut self,
        _signal: &TradingSignal,
        _executed_price: f64,
        _executed_size: f64,
    ) -> Result<()> {
        Ok(())
    }

    fn on_error(&mut self, _error: &anyhow::Error) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str;

    fn start(&mut self) {}

    fn stop(&mut self) {}

    fn get_status(&self) -> serde_yaml::Value {
        serde_yaml::Value::Null
    }
}

#[async_trait]
pub trait ExchangeAdapter: Send + Sync {
    async fn connect(&self) -> Result<()>;
    async fn disconnect(&self) -> Result<()>;
    async fn get_balance(&self, asset: &str) -> Result<Balance>;
    async fn get_market_price(&self, asset: &str) -> Result<f64>;
    async fn place_order(&self, order: Order) -> Result<Order>;
    async fn cancel_order(&self, order_id: &str) -> Result<bool>;
    async fn get_order_status(&self, order_id: &str) -> Result<Order>;
    async fn get_market_info(&self, asset: &str) -> Result<MarketInfo>;
    async fn get_positions(&self) -> Result<Vec<Position>>;
    async fn close_position(&self, asset: &str, size: Option<f64>) -> Result<bool>;
    async fn get_account_metrics(&self) -> Result<serde_json::Value>;
    async fn get_open_orders(&self) -> Result<Vec<Order>>;
    async fn cancel_all_orders(&self) -> Result<usize>;
    fn get_status(&self) -> serde_json::Value;
    async fn health_check(&self) -> Result<bool>;
}

#[async_trait]
pub trait MarketDataProvider: Send + Sync {
    async fn connect(&mut self) -> Result<()>;
    async fn disconnect(&mut self) -> Result<()>;
    async fn subscribe_price_updates<F, Fut>(&mut self, asset: &str, handler: F) -> Result<()>
    where
        F: Fn(MarketData) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static;
    async fn unsubscribe(&mut self, asset: &str) -> Result<()>;
    fn get_status(&self) -> serde_json::Value;
}
