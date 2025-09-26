use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use hyperliquid_bot::info_snapshot::{SnapshotRequest, fetch_user_snapshot};
use hyperliquid_bot::leaderboard::InfoApiClient;

struct StubClient {
    responses: Mutex<HashMap<String, VecDeque<Value>>>,
    requests: Mutex<Vec<Value>>,
}

impl StubClient {
    fn new() -> Self {
        Self {
            responses: Mutex::new(HashMap::new()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn push(&self, key: &str, value: Value) {
        let mut map = self.responses.lock().unwrap();
        map.entry(key.to_string()).or_default().push_back(value);
    }

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl InfoApiClient for StubClient {
    async fn post(&self, body: Value) -> Result<Value> {
        self.requests.lock().unwrap().push(body.clone());
        let key = body
            .get("type")
            .and_then(Value::as_str)
            .map(|s| s.to_string())
            .unwrap();
        let mut map = self.responses.lock().unwrap();
        let queue = map.get_mut(&key).unwrap();
        Ok(queue.pop_front().unwrap())
    }
}

#[tokio::test]
async fn snapshot_gathers_expected_payloads() {
    let client = Arc::new(StubClient::new());
    client.push("openOrders", json!([{ "oid": 1 }]));
    client.push(
        "frontendOpenOrders",
        json!([{ "oid": 2, "reduceOnly": true }]),
    );
    client.push(
        "historicalOrders",
        json!([{ "oid": 3, "status": "filled" }]),
    );
    client.push("userFills", json!([{ "oid": 4, "time": 12 }]));
    client.push("userTwapSliceFills", json!([{ "oid": 5 }]));
    client.push("clearinghouseState", json!({ "assetPositions": [] }));
    client.push("userFillsByTime", json!([{ "oid": 6, "time": 15 }]));

    let request = SnapshotRequest {
        address: "0x0000000000000000000000000000000000000001".into(),
        dex: Some("perp0".into()),
        aggregate_fills: true,
        fills_window: Some((10, 20)),
    };

    let snapshot = fetch_user_snapshot(client.clone(), request.clone())
        .await
        .expect("snapshot");

    assert_eq!(snapshot.address, request.address);
    assert_eq!(snapshot.dex, request.dex);
    assert_eq!(
        snapshot.open_orders[0].get("oid").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        snapshot.frontend_open_orders[0].get("reduceOnly"),
        Some(&Value::Bool(true))
    );
    assert_eq!(snapshot.historical_orders.len(), 1);
    assert_eq!(snapshot.fills.len(), 1);
    assert_eq!(snapshot.twap_slice_fills.len(), 1);
    assert!(snapshot.clearinghouse_state.get("assetPositions").is_some());
    assert_eq!(snapshot.fills_window.as_ref().unwrap().len(), 1);
    assert!(snapshot.fetched_at_ms > 0);

    let logged = client.requests();
    assert!(
        logged
            .iter()
            .any(|body| body.get("aggregateByTime") == Some(&Value::Bool(true)))
    );
    let window_call = logged
        .iter()
        .find(|body| body.get("type") == Some(&Value::String("userFillsByTime".into())))
        .unwrap();
    assert_eq!(
        window_call.get("startTime").and_then(Value::as_u64),
        Some(10)
    );
    assert_eq!(window_call.get("endTime").and_then(Value::as_u64), Some(20));
}

#[tokio::test]
async fn fails_when_open_orders_shape_invalid() {
    let client = Arc::new(StubClient::new());
    client.push("openOrders", json!({ "unexpected": true }));

    let request = SnapshotRequest {
        address: "0x0000000000000000000000000000000000000002".into(),
        dex: None,
        aggregate_fills: false,
        fills_window: None,
    };

    let error = fetch_user_snapshot(client, request).await.unwrap_err();
    assert!(error.to_string().contains("openOrders"));
}
