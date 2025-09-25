use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use hyperliquid_bot::leaderboard::{InfoApiClient, LeaderboardParams, fetch_top_wallets};
use serde_json::{Value, json};
use tokio::sync::Mutex;

struct StubInfoClient {
    leaderboard: Vec<Value>,
    fills: HashMap<String, Vec<Vec<Value>>>,
    funding: HashMap<String, Vec<Vec<Value>>>,
    fill_index: Mutex<HashMap<String, usize>>,
    funding_index: Mutex<HashMap<String, usize>>,
}

impl StubInfoClient {
    fn new(
        leaderboard: Vec<Value>,
        fills: HashMap<String, Vec<Vec<Value>>>,
        funding: HashMap<String, Vec<Vec<Value>>>,
    ) -> Self {
        Self {
            leaderboard,
            fills,
            funding,
            fill_index: Mutex::new(HashMap::new()),
            funding_index: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl InfoApiClient for StubInfoClient {
    async fn post(&self, body: Value) -> Result<Value> {
        let request_type = body
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        match request_type {
            "leaderboard" => Ok(Value::Array(self.leaderboard.clone())),
            "userFillsByTime" => {
                let user = body.get("user").and_then(|v| v.as_str()).unwrap();
                let mut index_guard = self.fill_index.lock().await;
                let entry = index_guard.entry(user.to_string()).or_insert(0);
                let pages = self.fills.get(user).cloned().unwrap_or_default();
                let page = pages.get(*entry).cloned().unwrap_or_else(|| Vec::new());
                if !page.is_empty() {
                    *entry += 1;
                }
                Ok(Value::Array(page))
            }
            "userFunding" => {
                let user = body.get("user").and_then(|v| v.as_str()).unwrap();
                let mut index_guard = self.funding_index.lock().await;
                let entry = index_guard.entry(user.to_string()).or_insert(0);
                let pages = self.funding.get(user).cloned().unwrap_or_default();
                let page = pages.get(*entry).cloned().unwrap_or_else(|| Vec::new());
                if !page.is_empty() {
                    *entry += 1;
                }
                Ok(Value::Array(page))
            }
            _ => Ok(Value::Null),
        }
    }
}

#[tokio::test]
async fn rejects_testnet_configs() {
    let client = Arc::new(StubInfoClient::new(
        Vec::new(),
        HashMap::new(),
        HashMap::new(),
    ));
    let mut params = LeaderboardParams::default();
    params.is_testnet = true;
    let result = fetch_top_wallets(client, params).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn aggregates_and_sorts_net_pnl() {
    let leaderboard = vec![json!({"address": "0xAAA"}), json!({"address": "0xBBB"})];
    let mut fills: HashMap<String, Vec<Vec<Value>>> = HashMap::new();
    fills.insert(
        "0xAAA".into(),
        vec![vec![
            json!({"closedPnl": "120.5", "fee": "5.5"}),
            json!({"closedPnl": "-20.0", "fee": "1.0", "builderFee": "0.5"}),
        ]],
    );
    fills.insert(
        "0xBBB".into(),
        vec![vec![
            json!({"closedPnl": "60.0", "fee": "2.0"}),
            json!({"closedPnl": "40.0", "fee": "1.0"}),
        ]],
    );
    let mut funding: HashMap<String, Vec<Vec<Value>>> = HashMap::new();
    funding.insert("0xAAA".into(), vec![vec![json!({"funding": "3.0"})]]);
    funding.insert("0xBBB".into(), vec![vec![json!({"amount": "10.0"})]]);
    let client = Arc::new(StubInfoClient::new(leaderboard, fills, funding));
    let mut params = LeaderboardParams::default();
    params.limit_addresses = 2;
    params.concurrency = 1;
    params.end_ms_override = Some(1_000);
    let results = fetch_top_wallets(client, params).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].address, "0xBBB");
    assert_eq!(results[0].rank, 1);
    assert!((results[0].net_pnl - 107.0).abs() < 1e-9);
    assert_eq!(results[1].rank, 2);
    assert_eq!(results[1].address, "0xAAA");
    assert!((results[1].net_pnl - 96.5).abs() < 1e-9);
    let breakdown = &results[1].breakdown;
    assert_eq!(breakdown.fills_count, 2);
    assert_eq!(breakdown.funding_events, 1);
    assert!((breakdown.fees - 7.0).abs() < 1e-9);
}

#[tokio::test]
async fn pagination_stops_on_empty_pages() {
    let leaderboard = vec![json!({"address": "0x111"})];
    let mut fills: HashMap<String, Vec<Vec<Value>>> = HashMap::new();
    fills.insert(
        "0x111".into(),
        vec![
            vec![json!({"closedPnl": 10.0, "fee": 1.0, "time": 5})],
            vec![json!({"closedPnl": 5.0, "fee": 0.5, "time": 6})],
        ],
    );
    let funding: HashMap<String, Vec<Vec<Value>>> = HashMap::new();
    let client = Arc::new(StubInfoClient::new(leaderboard, fills, funding));
    let mut params = LeaderboardParams::default();
    params.limit_addresses = 1;
    params.concurrency = 1;
    params.end_ms_override = Some(1_000);
    params.max_fill_pages = 5;
    let results = fetch_top_wallets(client, params).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].breakdown.fills_count, 2);
}
