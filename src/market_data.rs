use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use futures::future::BoxFuture;
use nautilus_hyperliquid::http::{client::HyperliquidHttpClient, query::InfoRequest};
use serde_json::json;
use tokio::{sync::Mutex, task::JoinHandle, time::interval};

use crate::{
    config::BotConfig,
    interfaces::{MarketData, MarketDataProvider},
};

type Handler = Arc<dyn Fn(MarketData) -> BoxFuture<'static, Result<()>> + Send + Sync>;

#[derive(Default)]
struct MarketDataState {
    subscriptions: HashMap<String, Vec<Handler>>,
    latest: HashMap<String, MarketData>,
    task: Option<JoinHandle<()>>,
    running: bool,
}

pub struct HyperliquidMarketData {
    http_client: Arc<HyperliquidHttpClient>,
    state: Arc<Mutex<MarketDataState>>,
    testnet: bool,
}

impl HyperliquidMarketData {
    pub fn new(config: &BotConfig) -> Self {
        let testnet = config.exchange.testnet;
        let client = Arc::new(HyperliquidHttpClient::new(testnet, None));
        Self {
            http_client: client,
            state: Arc::new(Mutex::new(MarketDataState::default())),
            testnet,
        }
    }

    async fn start_loop(&self) {
        let state_arc = self.state.clone();
        let client = self.http_client.clone();
        let mut guard = state_arc.lock().await;
        if guard.running {
            return;
        }
        guard.running = true;
        let state = state_arc.clone();
        let task = tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(2));
            loop {
                ticker.tick().await;
                let request = InfoRequest {
                    request_type: "allMids".to_string(),
                    params: json!({}),
                };
                let response = match client.send_info_request_raw(&request).await {
                    Ok(value) => value,
                    Err(_) => continue,
                };
                let mids = match response.get("mids").and_then(|v| v.as_object()) {
                    Some(map) => map.clone(),
                    None => continue,
                };
                let (subscriptions, updates) = {
                    let mut locked = state.lock().await;
                    let subs = locked.subscriptions.clone();
                    let mut updates = Vec::new();
                    for (asset, value) in mids.iter() {
                        if let Some(price) = value.as_str().and_then(|s| s.parse::<f64>().ok()) {
                            let market_data = MarketData {
                                asset: asset.clone(),
                                price,
                                volume_24h: 0.0,
                                timestamp: Utc::now(),
                                bid: None,
                                ask: None,
                                volatility: None,
                            };
                            locked.latest.insert(asset.clone(), market_data.clone());
                            updates.push((asset.clone(), market_data));
                        }
                    }
                    (subs, updates)
                };
                for (asset, data) in updates {
                    if let Some(handlers) = subscriptions.get(&asset) {
                        for handler in handlers {
                            let fut = handler(data.clone());
                            tokio::spawn(async move {
                                let _ = fut.await;
                            });
                        }
                    }
                }
            }
        });
        guard.task = Some(task);
    }
}

#[async_trait]
impl MarketDataProvider for HyperliquidMarketData {
    async fn connect(&mut self) -> Result<()> {
        self.start_loop().await;
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        let mut guard = self.state.lock().await;
        if let Some(task) = guard.task.take() {
            task.abort();
        }
        guard.running = false;
        Ok(())
    }

    async fn subscribe_price_updates<F, Fut>(&mut self, asset: &str, handler: F) -> Result<()>
    where
        F: Fn(MarketData) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        let mut guard = self.state.lock().await;
        let entry = guard.subscriptions.entry(asset.to_string()).or_default();
        let handler_arc: Handler = Arc::new(move |data| {
            let fut = handler(data);
            Box::pin(fut)
        });
        entry.push(handler_arc);
        Ok(())
    }

    async fn unsubscribe(&mut self, asset: &str) -> Result<()> {
        let mut guard = self.state.lock().await;
        guard.subscriptions.remove(asset);
        Ok(())
    }

    fn get_status(&self) -> serde_json::Value {
        serde_json::json!({
            "testnet": self.testnet,
        })
    }
}
