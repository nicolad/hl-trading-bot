use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use chrono::Utc;
use nautilus_hyperliquid::{
    common::credential::{EvmPrivateKey, Secrets},
    http::{
        client::HyperliquidHttpClient,
        models::{HyperliquidAssetInfo, HyperliquidMeta},
        query::InfoRequest,
    },
};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;

use crate::{
    config::BotConfig,
    interfaces::{Balance, ExchangeAdapter, MarketInfo, Order, OrderSide, OrderStatus, Position},
};

#[derive(Default)]
struct ExchangeState {
    balances: HashMap<String, Balance>,
    positions: HashMap<String, Position>,
    open_orders: HashMap<String, Order>,
    meta_cache: Vec<HyperliquidAssetInfo>,
}

pub struct HyperliquidPublicExchange {
    http_client: HyperliquidHttpClient,
    testnet: bool,
    connected: AtomicBool,
    state: Mutex<ExchangeState>,
}

impl HyperliquidPublicExchange {
    pub fn new(config: &BotConfig) -> Result<Self> {
        let testnet = config.exchange.testnet;
        let http_client = Self::build_client(testnet)?;
        Ok(Self {
            http_client,
            testnet,
            connected: AtomicBool::new(false),
            state: Mutex::new(ExchangeState::default()),
        })
    }

    fn build_client(testnet: bool) -> Result<HyperliquidHttpClient> {
        let env_key = std::env::var(if testnet {
            "HYPERLIQUID_TESTNET_PRIVATE_KEY"
        } else {
            "HYPERLIQUID_MAINNET_PRIVATE_KEY"
        })
        .ok();
        if let Some(key) = env_key {
            if key.trim().is_empty() {
                Ok(HyperliquidHttpClient::new(testnet, None))
            } else {
                let secrets = Secrets {
                    private_key: EvmPrivateKey::new(key)?,
                    vault_address: None,
                    is_testnet: testnet,
                };
                Ok(HyperliquidHttpClient::with_credentials(&secrets, None))
            }
        } else {
            Ok(HyperliquidHttpClient::new(testnet, None))
        }
    }

    async fn ensure_meta(&self) -> Result<()> {
        let mut guard = self.state.lock().await;
        if guard.meta_cache.is_empty() {
            let HyperliquidMeta { universe } = self.http_client.info_meta().await?;
            guard.meta_cache = universe;
        }
        Ok(())
    }

    async fn all_mids(&self) -> Result<HashMap<String, f64>> {
        let request = InfoRequest {
            request_type: "allMids".to_string(),
            params: json!({}),
        };
        let response = self.http_client.send_info_request_raw(&request).await?;
        let mut mids = HashMap::new();
        if let Some(map) = response.get("mids").and_then(|v| v.as_object()) {
            for (asset, value) in map {
                if let Some(price) = value.as_str()
                    && let Ok(parsed) = price.parse::<f64>()
                {
                    mids.insert(asset.clone(), parsed);
                }
            }
        }
        Ok(mids)
    }

    fn update_balance_on_fill(
        state: &mut ExchangeState,
        side: OrderSide,
        price: f64,
        size: f64,
        asset: &str,
    ) {
        let usd = state.balances.entry("USD".into()).or_insert(Balance {
            asset: "USD".into(),
            available: 10000.0,
            locked: 0.0,
            total: 10000.0,
        });
        match side {
            OrderSide::Buy => {
                let cost = price * size;
                usd.available -= cost;
                usd.total -= cost;
            }
            OrderSide::Sell => {
                let proceeds = price * size;
                usd.available += proceeds;
                usd.total += proceeds;
            }
        }
        let position = state
            .positions
            .entry(asset.to_string())
            .or_insert(Position {
                asset: asset.to_string(),
                size: 0.0,
                entry_price: price,
                current_value: 0.0,
                unrealized_pnl: 0.0,
                timestamp: Utc::now(),
            });
        match side {
            OrderSide::Buy => {
                let total_size = position.size + size;
                let cost_basis = position.entry_price * position.size.abs() + price * size;
                position.size = total_size;
                if total_size.abs() > f64::EPSILON {
                    position.entry_price = cost_basis / total_size.abs();
                } else {
                    position.entry_price = price;
                }
            }
            OrderSide::Sell => {
                position.size -= size;
                if position.size.abs() <= f64::EPSILON {
                    state.positions.remove(asset);
                }
            }
        }
    }

    fn refresh_position_values(state: &mut ExchangeState, price_map: &HashMap<String, f64>) {
        for position in state.positions.values_mut() {
            if let Some(price) = price_map.get(&position.asset) {
                position.current_value = position.size.abs() * price;
                position.unrealized_pnl = 0.0;
                position.timestamp = Utc::now();
            }
        }
    }

    fn compute_metrics(state: &ExchangeState) -> serde_json::Value {
        let total_value: f64 = state
            .balances
            .values()
            .map(|b| b.available + b.locked)
            .sum();
        let positions_count = state.positions.len();
        let largest_position_pct = if total_value > 0.0 {
            state
                .positions
                .values()
                .map(|p| (p.current_value.abs() / total_value) * 100.0)
                .fold(0.0, f64::max)
        } else {
            0.0
        };
        json!({
            "total_value": total_value,
            "total_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "drawdown_pct": 0.0,
            "positions_count": positions_count,
            "largest_position_pct": largest_position_pct,
        })
    }
}

#[async_trait]
impl ExchangeAdapter for HyperliquidPublicExchange {
    async fn connect(&self) -> Result<()> {
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> Result<()> {
        self.connected.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn get_balance(&self, asset: &str) -> Result<Balance> {
        let mut guard = self.state.lock().await;
        Ok(guard
            .balances
            .entry(asset.to_string())
            .or_insert(Balance {
                asset: asset.into(),
                available: 0.0,
                locked: 0.0,
                total: 0.0,
            })
            .clone())
    }

    async fn get_market_price(&self, asset: &str) -> Result<f64> {
        let mids = self.all_mids().await?;
        mids.get(asset)
            .copied()
            .ok_or_else(|| anyhow!("price for {} not found", asset))
    }

    async fn place_order(&self, mut order: Order) -> Result<Order> {
        let price = match order.price {
            Some(price) => price,
            None => self
                .all_mids()
                .await?
                .get(&order.asset)
                .copied()
                .context("missing market price")?,
        };
        let mut guard = self.state.lock().await;
        Self::update_balance_on_fill(
            &mut guard,
            order.side,
            price,
            order.size.abs(),
            &order.asset,
        );
        let mids = self.all_mids().await.unwrap_or_default();
        Self::refresh_position_values(&mut guard, &mids);
        order.status = OrderStatus::Filled;
        order.filled_size = order.size;
        order.average_fill_price = price;
        order.exchange_order_id = Some(order.id.to_string());
        guard
            .open_orders
            .insert(order.id.to_string(), order.clone());
        Ok(order)
    }

    async fn cancel_order(&self, order_id: &str) -> Result<bool> {
        let mut guard = self.state.lock().await;
        Ok(guard.open_orders.remove(order_id).is_some())
    }

    async fn get_order_status(&self, order_id: &str) -> Result<Order> {
        let guard = self.state.lock().await;
        guard
            .open_orders
            .get(order_id)
            .cloned()
            .ok_or_else(|| anyhow!("order {} not found", order_id))
    }

    async fn get_market_info(&self, asset: &str) -> Result<MarketInfo> {
        self.ensure_meta().await?;
        let guard = self.state.lock().await;
        let info = guard
            .meta_cache
            .iter()
            .find(|item| item.name.as_str() == asset)
            .cloned()
            .ok_or_else(|| anyhow!("asset {} not found", asset))?;
        Ok(MarketInfo {
            symbol: info.name.to_string(),
            base_asset: info.name.to_string(),
            quote_asset: "USD".into(),
            min_order_size: 10f64.powi(-(info.sz_decimals as i32)),
            price_precision: 2,
            size_precision: info.sz_decimals,
            is_active: true,
        })
    }

    async fn get_positions(&self) -> Result<Vec<Position>> {
        let guard = self.state.lock().await;
        Ok(guard.positions.values().cloned().collect())
    }

    async fn close_position(&self, asset: &str, size: Option<f64>) -> Result<bool> {
        let mut guard = self.state.lock().await;
        if let Some(position) = guard.positions.get_mut(asset) {
            let close_size = size.unwrap_or_else(|| position.size.abs());
            position.size -= close_size.copysign(position.size);
            if position.size.abs() <= f64::EPSILON {
                guard.positions.remove(asset);
            }
            return Ok(true);
        }
        Ok(false)
    }

    async fn get_account_metrics(&self) -> Result<serde_json::Value> {
        let guard = self.state.lock().await;
        Ok(Self::compute_metrics(&guard))
    }

    async fn get_open_orders(&self) -> Result<Vec<Order>> {
        let guard = self.state.lock().await;
        Ok(guard.open_orders.values().cloned().collect())
    }

    async fn cancel_all_orders(&self) -> Result<usize> {
        let mut guard = self.state.lock().await;
        let count = guard.open_orders.len();
        guard.open_orders.clear();
        Ok(count)
    }

    fn get_status(&self) -> serde_json::Value {
        json!({
            "connected": self.connected.load(Ordering::SeqCst),
            "testnet": self.testnet,
        })
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(self.http_client.info_meta().await.is_ok())
    }
}
