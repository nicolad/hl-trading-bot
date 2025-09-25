use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;

use crate::interfaces::OrderSide;

#[derive(Clone, Debug, PartialEq)]
pub struct SpotPrice {
    pub price: f64,
    pub size_decimals: u32,
}

impl SpotPrice {
    pub fn new(price: f64, size_decimals: u32) -> Self {
        Self {
            price,
            size_decimals,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MirrorOrderRequest {
    pub coin: String,
    pub side: OrderSide,
    pub size: f64,
    pub price: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OrderPlacement {
    Resting { order_id: u64 },
    Filled,
}

#[async_trait]
pub trait SpotMirrorExchange: Send + Sync {
    async fn get_spot_price(&self, coin: &str) -> Result<SpotPrice>;
    async fn place_spot_order(&self, request: MirrorOrderRequest) -> Result<OrderPlacement>;
    async fn cancel_spot_order(&self, order_id: u64, coin: &str) -> Result<bool>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct LeaderOrder {
    pub id: u64,
    pub coin: String,
    pub side: OrderSide,
    pub price: f64,
    pub size: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LeaderOrderStatus {
    Open,
    Canceled,
    Filled,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LeaderOrderUpdate {
    pub order: LeaderOrder,
    pub status: LeaderOrderStatus,
}

pub struct SpotOrderMirror {
    exchange: Arc<dyn SpotMirrorExchange>,
    fixed_value_usdc: f64,
    leader_address: String,
    order_mappings: HashMap<u64, u64>,
}

impl SpotOrderMirror {
    pub fn new(
        exchange: Arc<dyn SpotMirrorExchange>,
        is_testnet: bool,
        leader_address: impl Into<String>,
        fixed_value_usdc: f64,
    ) -> Result<Self> {
        ensure!(!is_testnet, "spot mirroring is available only on mainnet");
        let leader_address = leader_address.into();
        ensure!(!leader_address.trim().is_empty(), "leader address required");
        ensure!(fixed_value_usdc > 0.0, "fixed order value must be positive");
        Ok(Self {
            exchange,
            fixed_value_usdc,
            leader_address,
            order_mappings: HashMap::new(),
        })
    }

    pub async fn handle_order_update(&mut self, update: LeaderOrderUpdate) -> Result<()> {
        if !is_spot_coin(&update.order.coin) {
            self.order_mappings.remove(&update.order.id);
            return Ok(());
        }
        match update.status {
            LeaderOrderStatus::Open => self.handle_open(&update.order).await,
            LeaderOrderStatus::Canceled => self.handle_cancel(&update.order).await,
            LeaderOrderStatus::Filled => {
                self.order_mappings.remove(&update.order.id);
                Ok(())
            }
            LeaderOrderStatus::Unknown => Ok(()),
        }
    }

    pub fn follower_order_for(&self, leader_order_id: u64) -> Option<u64> {
        self.order_mappings.get(&leader_order_id).copied()
    }

    pub fn leader_address(&self) -> &str {
        &self.leader_address
    }

    async fn handle_open(&mut self, order: &LeaderOrder) -> Result<()> {
        ensure!(order.price > 0.0, "leader order price must be positive");
        let coin = order.coin.clone();
        let price_info = {
            let exchange = Arc::clone(&self.exchange);
            exchange.get_spot_price(&coin).await?
        };
        let size = calculate_order_size(self.fixed_value_usdc, &price_info)?;
        let request = MirrorOrderRequest {
            coin: coin.clone(),
            side: order.side,
            size,
            price: order.price,
        };
        let placement = {
            let exchange = Arc::clone(&self.exchange);
            exchange.place_spot_order(request).await?
        };
        match placement {
            OrderPlacement::Resting { order_id } => {
                self.order_mappings.insert(order.id, order_id);
            }
            OrderPlacement::Filled => {
                self.order_mappings.remove(&order.id);
            }
        }
        Ok(())
    }

    async fn handle_cancel(&mut self, order: &LeaderOrder) -> Result<()> {
        if let Some(follower_id) = self.order_mappings.remove(&order.id) {
            let coin = order.coin.clone();
            let exchange = Arc::clone(&self.exchange);
            exchange.cancel_spot_order(follower_id, &coin).await?;
        }
        Ok(())
    }
}

fn calculate_order_size(fixed_value_usdc: f64, price_info: &SpotPrice) -> Result<f64> {
    ensure!(price_info.price > 0.0, "spot price must be positive");
    let raw = fixed_value_usdc / price_info.price;
    let decimals = price_info.size_decimals.min(15);
    let factor = 10_f64.powi(decimals as i32);
    let rounded = (raw * factor).round() / factor;
    if rounded <= 0.0 {
        return Err(anyhow!("calculated order size is zero"));
    }
    Ok(rounded)
}

fn is_spot_coin(coin: &str) -> bool {
    if coin.is_empty() || coin == "N/A" {
        return false;
    }
    if coin.starts_with('@') {
        if coin.len() == 1 {
            return false;
        }
        return coin[1..]
            .parse::<i64>()
            .map(|value| value >= 0)
            .unwrap_or(false);
    }
    if coin.contains('/') {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_spot_coins() {
        assert!(is_spot_coin("@1"));
        assert!(is_spot_coin("ETH/USDC"));
        assert!(!is_spot_coin("BTC"));
        assert!(!is_spot_coin("@-1"));
    }

    #[test]
    fn calculates_order_size_with_rounding() {
        let info = SpotPrice::new(25.0, 3);
        let size = calculate_order_size(10.0, &info).unwrap();
        assert!((size - 0.4).abs() < 1e-9);
    }
}
