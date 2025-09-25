use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use hyperliquid_bot::interfaces::OrderSide;
use hyperliquid_bot::mirroring::{
    LeaderOrder, LeaderOrderStatus, LeaderOrderUpdate, MirrorOrderRequest, OrderPlacement,
    SpotMirrorExchange, SpotOrderMirror, SpotPrice,
};
use tokio::sync::Mutex;

struct StubExchange {
    prices: Mutex<VecDeque<SpotPrice>>,
    placement_responses: Mutex<VecDeque<OrderPlacement>>,
    placed_orders: Mutex<Vec<MirrorOrderRequest>>,
    cancellations: Mutex<Vec<(u64, String)>>,
}

impl StubExchange {
    fn new(prices: Vec<SpotPrice>, responses: Vec<OrderPlacement>) -> Self {
        Self {
            prices: Mutex::new(VecDeque::from(prices)),
            placement_responses: Mutex::new(VecDeque::from(responses)),
            placed_orders: Mutex::new(Vec::new()),
            cancellations: Mutex::new(Vec::new()),
        }
    }

    async fn placed(&self) -> Vec<MirrorOrderRequest> {
        self.placed_orders.lock().await.clone()
    }

    async fn cancelled(&self) -> Vec<(u64, String)> {
        self.cancellations.lock().await.clone()
    }
}

#[async_trait]
impl SpotMirrorExchange for StubExchange {
    async fn get_spot_price(&self, _coin: &str) -> Result<SpotPrice> {
        let mut guard = self.prices.lock().await;
        guard
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("missing spot price"))
    }

    async fn place_spot_order(&self, request: MirrorOrderRequest) -> Result<OrderPlacement> {
        self.placed_orders.lock().await.push(request);
        let mut guard = self.placement_responses.lock().await;
        guard
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("missing placement response"))
    }

    async fn cancel_spot_order(&self, order_id: u64, coin: &str) -> Result<bool> {
        self.cancellations
            .lock()
            .await
            .push((order_id, coin.to_string()));
        Ok(true)
    }
}

fn open_order(id: u64, coin: &str, side: OrderSide, price: f64) -> LeaderOrderUpdate {
    LeaderOrderUpdate {
        order: LeaderOrder {
            id,
            coin: coin.to_string(),
            side,
            price,
            size: 1.0,
        },
        status: LeaderOrderStatus::Open,
    }
}

#[tokio::test]
async fn rejects_testnet_configs() {
    let stub = Arc::new(StubExchange::new(vec![], vec![]));
    let result = SpotOrderMirror::new(stub, true, "0xleader", 10.0);
    assert!(result.is_err());
}

#[tokio::test]
async fn mirrors_spot_order_with_fixed_value() {
    let stub = Arc::new(StubExchange::new(
        vec![SpotPrice::new(25.0, 3)],
        vec![OrderPlacement::Resting { order_id: 55 }],
    ));
    let exchange = stub.clone() as Arc<dyn SpotMirrorExchange>;
    let mut mirror = SpotOrderMirror::new(exchange, false, "0xleader", 10.0).unwrap();

    mirror
        .handle_order_update(open_order(1, "@5", OrderSide::Buy, 24.5))
        .await
        .unwrap();

    let placed = stub.placed().await;
    assert_eq!(placed.len(), 1);
    let order = &placed[0];
    assert!((order.size - 0.4).abs() < 1e-9);
    assert_eq!(order.coin, "@5");
    assert_eq!(order.side, OrderSide::Buy);
    assert!((order.price - 24.5).abs() < 1e-9);
    assert_eq!(mirror.follower_order_for(1), Some(55));
}

#[tokio::test]
async fn ignores_non_spot_orders() {
    let stub = Arc::new(StubExchange::new(vec![SpotPrice::new(20.0, 3)], vec![]));
    let exchange = stub.clone() as Arc<dyn SpotMirrorExchange>;
    let mut mirror = SpotOrderMirror::new(exchange, false, "0xleader", 10.0).unwrap();

    mirror
        .handle_order_update(open_order(7, "BTC", OrderSide::Buy, 19.0))
        .await
        .unwrap();

    assert!(stub.placed().await.is_empty());
    assert_eq!(mirror.follower_order_for(7), None);
}

#[tokio::test]
async fn cancels_follower_on_leader_cancel() {
    let stub = Arc::new(StubExchange::new(
        vec![SpotPrice::new(30.0, 4)],
        vec![OrderPlacement::Resting { order_id: 88 }],
    ));
    let exchange = stub.clone() as Arc<dyn SpotMirrorExchange>;
    let mut mirror = SpotOrderMirror::new(exchange, false, "0xleader", 10.0).unwrap();

    mirror
        .handle_order_update(open_order(9, "@2", OrderSide::Sell, 29.5))
        .await
        .unwrap();

    mirror
        .handle_order_update(LeaderOrderUpdate {
            order: LeaderOrder {
                id: 9,
                coin: "@2".into(),
                side: OrderSide::Sell,
                price: 29.5,
                size: 1.0,
            },
            status: LeaderOrderStatus::Canceled,
        })
        .await
        .unwrap();

    let cancelled = stub.cancelled().await;
    assert_eq!(cancelled, vec![(88, "@2".to_string())]);
    assert_eq!(mirror.follower_order_for(9), None);
}

#[tokio::test]
async fn does_not_map_immediate_fills() {
    let stub = Arc::new(StubExchange::new(
        vec![SpotPrice::new(50.0, 2)],
        vec![OrderPlacement::Filled],
    ));
    let exchange = stub.clone() as Arc<dyn SpotMirrorExchange>;
    let mut mirror = SpotOrderMirror::new(exchange, false, "0xleader", 10.0).unwrap();

    mirror
        .handle_order_update(open_order(42, "@7", OrderSide::Buy, 49.0))
        .await
        .unwrap();

    assert!(stub.cancelled().await.is_empty());
    assert_eq!(mirror.follower_order_for(42), None);

    mirror
        .handle_order_update(LeaderOrderUpdate {
            order: LeaderOrder {
                id: 42,
                coin: "@7".into(),
                side: OrderSide::Buy,
                price: 49.0,
                size: 1.0,
            },
            status: LeaderOrderStatus::Canceled,
        })
        .await
        .unwrap();

    assert!(stub.cancelled().await.is_empty());
}
