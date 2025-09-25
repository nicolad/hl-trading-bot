use std::sync::Arc;

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use hyperliquid_bot::leaderboard::{
    InfoApiClient, LeaderboardParams, ReqwestInfoClient, fetch_top_wallets,
};

const API_URL: &str = "https://api.hyperliquid.xyz/info";
const LIMIT_ADDRESSES: usize = 100;
const CONCURRENCY: usize = 8;
const START_MS: u64 = 0;
const END_MS_OVERRIDE: Option<u64> = None;
const MAX_FILL_PAGES: usize = 10_000;
const MAX_FUNDING_PAGES: usize = 5_000;
const MAX_ITEMS_SOFT_CAP: usize = 100_000;
const PAGE_DELAY_MS: u64 = 0;
const PRETTY_JSON: bool = true;
const OUTPUT_PATH: &str = "leaderboard.json";

#[tokio::main]
async fn main() -> Result<()> {
    let (params, pretty) = build_params();
    ensure!(
        !params.is_testnet,
        "leaderboard CLI operates on mainnet only"
    );
    let client: Arc<dyn InfoApiClient> = Arc::new(ReqwestInfoClient::new(params.api_url.clone())?);
    let results = fetch_top_wallets(client, params).await?;
    let rendered = if pretty {
        serde_json::to_string_pretty(&results)?
    } else {
        serde_json::to_string(&results)?
    };
    fs::write(output_path(), rendered.as_bytes())
        .with_context(|| format!("failed to write {}", OUTPUT_PATH))?;
    println!("wrote leaderboard report to {}", OUTPUT_PATH);
    Ok(())
}

fn build_params() -> (LeaderboardParams, bool) {
    let mut params = LeaderboardParams::default();
    params.api_url = API_URL.into();
    params.limit_addresses = LIMIT_ADDRESSES;
    params.concurrency = CONCURRENCY.max(1);
    params.start_ms = START_MS;
    params.end_ms_override = END_MS_OVERRIDE;
    params.max_fill_pages = MAX_FILL_PAGES;
    params.max_funding_pages = MAX_FUNDING_PAGES;
    params.max_items_soft_cap = MAX_ITEMS_SOFT_CAP;
    params.page_delay_ms = PAGE_DELAY_MS;
    (params, PRETTY_JSON)
}

fn output_path() -> PathBuf {
    PathBuf::from(OUTPUT_PATH)
}
