use anyhow::Result;
use hyperliquid_bot::patterns::{Config, display_report, run};

fn main() -> Result<()> {
    let cfg = Config::default();
    let output = run(&cfg)?;
    display_report(&output);
    Ok(())
}
