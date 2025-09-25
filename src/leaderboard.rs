use std::cmp::Ordering;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;
use futures::{StreamExt, stream};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::time::sleep;

#[async_trait]
pub trait InfoApiClient: Send + Sync {
    async fn post(&self, body: Value) -> Result<Value>;
}

pub struct ReqwestInfoClient {
    client: Client,
    api_url: String,
}

impl ReqwestInfoClient {
    pub fn new(api_url: impl Into<String>) -> Result<Self> {
        let client = Client::builder().user_agent("hl-top-profit/0.1").build()?;
        Ok(Self {
            client,
            api_url: api_url.into(),
        })
    }
}

#[async_trait]
impl InfoApiClient for ReqwestInfoClient {
    async fn post(&self, body: Value) -> Result<Value> {
        let mut last_err = None;
        for attempt in 0..5u32 {
            match self
                .client
                .post(&self.api_url)
                .json(&body)
                .header("content-type", "application/json")
                .send()
                .await
            {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        return Ok(response.json().await?);
                    }
                    last_err = Some(anyhow!("http {}", status));
                }
                Err(err) => {
                    last_err = Some(anyhow!(err));
                }
            }
            sleep(std::time::Duration::from_millis(200 + 100 * attempt as u64)).await;
        }
        Err(anyhow!("request failed: {last_err:?}"))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Breakdown {
    pub fills_count: usize,
    pub funding_events: usize,
    pub fees: f64,
    pub funding: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WalletResult {
    pub rank: usize,
    pub address: String,
    pub realized_pnl: f64,
    pub net_pnl: f64,
    pub breakdown: Breakdown,
    pub leaderboard_stat: Value,
}

#[derive(Clone, Debug)]
pub struct LeaderboardParams {
    pub api_url: String,
    pub limit_addresses: usize,
    pub concurrency: usize,
    pub start_ms: u64,
    pub end_ms_override: Option<u64>,
    pub is_testnet: bool,
    pub max_fill_pages: usize,
    pub max_funding_pages: usize,
    pub max_items_soft_cap: usize,
    pub page_delay_ms: u64,
}

impl Default for LeaderboardParams {
    fn default() -> Self {
        Self {
            api_url: "https://api.hyperliquid.xyz/info".into(),
            limit_addresses: 100,
            concurrency: 8,
            start_ms: 0,
            end_ms_override: None,
            is_testnet: false,
            max_fill_pages: 10_000,
            max_funding_pages: 5_000,
            max_items_soft_cap: 100_000,
            page_delay_ms: 0,
        }
    }
}

pub async fn fetch_top_wallets(
    client: Arc<dyn InfoApiClient>,
    mut params: LeaderboardParams,
) -> Result<Vec<WalletResult>> {
    ensure!(!params.is_testnet, "leaderboard analytics are mainnet only");
    let end_ms = params.end_ms_override.take().unwrap_or_else(current_millis);
    let leaderboard = fetch_leaderboard(client.clone(), &params).await?;
    let results = stream::iter(leaderboard.into_iter().enumerate())
        .map(|(index, entry)| {
            let client = client.clone();
            let params = params.clone();
            async move {
                let rank = index + 1;
                let address = extract_address(&entry)?;
                let (realized, net, breakdown) =
                    compute_realized_and_net(client, &params, &address, params.start_ms, end_ms)
                        .await?;
                Ok::<_, anyhow::Error>(WalletResult {
                    rank,
                    address,
                    realized_pnl: realized,
                    net_pnl: net,
                    breakdown,
                    leaderboard_stat: entry,
                })
            }
        })
        .buffer_unordered(params.concurrency)
        .filter_map(|result| async move { result.ok() })
        .collect::<Vec<WalletResult>>()
        .await;
    Ok(rank_by_net(results))
}

fn rank_by_net(mut results: Vec<WalletResult>) -> Vec<WalletResult> {
    results.sort_by(|a, b| match b.net_pnl.partial_cmp(&a.net_pnl) {
        Some(ordering) => ordering,
        None => Ordering::Equal,
    });
    for (index, item) in results.iter_mut().enumerate() {
        item.rank = index + 1;
    }
    results
}

async fn fetch_leaderboard(
    client: Arc<dyn InfoApiClient>,
    params: &LeaderboardParams,
) -> Result<Vec<Value>> {
    let response = client.post(json!({"type": "leaderboard"})).await?;
    let array = response
        .as_array()
        .ok_or_else(|| anyhow!("unexpected leaderboard response"))?;
    Ok(array.iter().take(params.limit_addresses).cloned().collect())
}

async fn compute_realized_and_net(
    client: Arc<dyn InfoApiClient>,
    params: &LeaderboardParams,
    address: &str,
    start_ms: u64,
    end_ms: u64,
) -> Result<(f64, f64, Breakdown)> {
    let fills = fetch_fills(client.clone(), params, address, start_ms, end_ms).await?;
    let mut realized = 0.0;
    let mut fees = 0.0;
    for fill in &fills {
        if let Some(value) = fill.get("closedPnl") {
            realized += as_f64(value);
        }
        if let Some(value) = fill.get("fee") {
            fees += as_f64(value);
        }
        if let Some(value) = fill.get("builderFee") {
            fees += as_f64(value);
        }
    }
    let funding_events = fetch_funding(client, params, address, start_ms, end_ms).await?;
    let mut funding = 0.0;
    for event in &funding_events {
        if let Some(value) = event
            .get("funding")
            .or_else(|| event.get("amount"))
            .or_else(|| event.get("value"))
        {
            funding += as_f64(value);
        }
    }
    let net = realized - fees + funding;
    Ok((
        realized,
        net,
        Breakdown {
            fills_count: fills.len(),
            funding_events: funding_events.len(),
            fees,
            funding,
        },
    ))
}

async fn fetch_fills(
    client: Arc<dyn InfoApiClient>,
    params: &LeaderboardParams,
    address: &str,
    start_ms: u64,
    end_ms: u64,
) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let mut cursor = start_ms;
    for page in 0..params.max_fill_pages {
        let body = json!({
            "type": "userFillsByTime",
            "user": address,
            "startTime": cursor,
            "endTime": end_ms
        });
        let chunk = client.post(body).await?;
        let array = chunk.as_array().cloned().unwrap_or_default();
        if array.is_empty() {
            break;
        }
        let mut max_ts = cursor;
        for item in &array {
            if let Some(ts) = item.get("time").and_then(|v| v.as_u64()) {
                if ts > max_ts {
                    max_ts = ts;
                }
            }
        }
        out.extend(array);
        if out.len() >= params.max_items_soft_cap {
            break;
        }
        if max_ts <= cursor {
            break;
        }
        cursor = max_ts + 1;
        if params.page_delay_ms > 0 {
            sleep(std::time::Duration::from_millis(
                params.page_delay_ms + (page as u64 % 10) * 5,
            ))
            .await;
        }
    }
    Ok(out)
}

async fn fetch_funding(
    client: Arc<dyn InfoApiClient>,
    params: &LeaderboardParams,
    address: &str,
    start_ms: u64,
    end_ms: u64,
) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let mut cursor = start_ms;
    for page in 0..params.max_funding_pages {
        let body = json!({
            "type": "userFunding",
            "user": address,
            "startTime": cursor,
            "endTime": end_ms
        });
        let chunk = client.post(body).await?;
        let array = chunk.as_array().cloned().unwrap_or_default();
        if array.is_empty() {
            break;
        }
        let mut max_ts = cursor;
        for item in &array {
            if let Some(ts) = item.get("time").and_then(|v| v.as_u64()) {
                if ts > max_ts {
                    max_ts = ts;
                }
            }
        }
        out.extend(array);
        if out.len() >= params.max_items_soft_cap {
            break;
        }
        if max_ts <= cursor {
            break;
        }
        cursor = max_ts + 1;
        if params.page_delay_ms > 0 {
            sleep(std::time::Duration::from_millis(
                params.page_delay_ms + (page as u64 % 10) * 5,
            ))
            .await;
        }
    }
    Ok(out)
}

fn extract_address(entry: &Value) -> Result<String> {
    for key in ["address", "user", "wallet"] {
        if let Some(value) = entry.get(key) {
            if let Some(addr) = value.as_str() {
                if !addr.is_empty() {
                    return Ok(addr.to_string());
                }
            }
        }
    }
    Err(anyhow!("missing address"))
}

fn as_f64(value: &Value) -> f64 {
    match value {
        Value::Number(number) => number.as_f64().unwrap_or(0.0),
        Value::String(text) => text.parse::<f64>().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn current_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_by_net_desc() {
        let mut results = vec![
            WalletResult {
                rank: 1,
                address: "a".into(),
                realized_pnl: 0.0,
                net_pnl: 1.0,
                breakdown: Breakdown {
                    fills_count: 0,
                    funding_events: 0,
                    fees: 0.0,
                    funding: 0.0,
                },
                leaderboard_stat: Value::Null,
            },
            WalletResult {
                rank: 2,
                address: "b".into(),
                realized_pnl: 0.0,
                net_pnl: 5.0,
                breakdown: Breakdown {
                    fills_count: 0,
                    funding_events: 0,
                    fees: 0.0,
                    funding: 0.0,
                },
                leaderboard_stat: Value::Null,
            },
        ];
        results = rank_by_net(results);
        assert_eq!(results[0].address, "b");
        assert_eq!(results[0].rank, 1);
        assert_eq!(results[1].rank, 2);
    }

    #[test]
    fn as_f64_handles_inputs() {
        assert!((as_f64(&json!("1.5")) - 1.5).abs() < 1e-9);
        assert!((as_f64(&json!(2)) - 2.0).abs() < 1e-9);
        assert_eq!(as_f64(&json!("bad")), 0.0);
    }
}
