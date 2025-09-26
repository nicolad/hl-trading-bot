use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use polars::prelude::*;
use serde::Serialize;

const FEATURE_COLUMNS: [&str; 20] = [
    "fills_count",
    "open_orders_count",
    "frontend_orders_count",
    "historical_orders_count",
    "twap_fills_count",
    "fills_window_count",
    "positions_count",
    "total_volume",
    "total_fees",
    "unique_coins",
    "avg_fill_size",
    "largest_fill",
    "activity_span_hours",
    "fills_with_timestamp_count",
    "account_value",
    "realized_pnl",
    "unrealized_pnl",
    "leverage",
    "win_rate",
    "expected_return",
];

#[derive(Clone, Debug)]
pub struct Config {
    pub features_path: PathBuf,
    pub patterns_path: Option<PathBuf>,
    pub signals_path: Option<PathBuf>,
    pub k_clusters: usize,
    pub min_cluster_size: usize,
    pub min_edge: f64,
    pub min_win_rate: f64,
    pub min_t_stat: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            features_path: PathBuf::from("trading_features.csv"),
            patterns_path: Some(PathBuf::from("strategy_patterns.json")),
            signals_path: Some(PathBuf::from("strategy_signals.csv")),
            k_clusters: 6,
            min_cluster_size: 10,
            min_edge: 0.001,
            min_win_rate: 0.52,
            min_t_stat: 1.5,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PatternSummary {
    pub cluster_id: u32,
    pub sample_count: usize,
    pub mean_return: f64,
    pub median_return: f64,
    pub return_std: f64,
    pub win_rate: f64,
    pub avg_volume: f64,
    pub avg_leverage: f64,
    pub confidence: f64,
    pub t_stat: f64,
    pub action: Option<String>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct StrategySignal {
    pub address: String,
    pub cluster_id: u32,
    pub action: String,
    pub expected_return: f64,
    pub confidence: f64,
}

#[derive(Clone, Debug)]
pub struct StrategyOutput {
    pub patterns: Vec<PatternSummary>,
    pub signals: Vec<StrategySignal>,
}

struct ClusterDecision {
    action: String,
    confidence: f64,
}

pub fn run(cfg: &Config) -> Result<StrategyOutput> {
    let mut df = load_features(&cfg.features_path)?;
    if df.height() == 0 {
        bail!("feature file is empty");
    }

    enrich_features(&mut df)?;
    let matrix = build_feature_matrix(&df)?;
    let n_samples = matrix.nrows();
    if n_samples < 2 {
        bail!("not enough samples to perform clustering");
    }

    let k = cfg.k_clusters.clamp(2, n_samples.max(2));
    let dataset = linfa::DatasetBase::from(matrix.clone());
    let model = KMeans::params(k)
        .max_n_iterations(300)
        .tolerance(1e-4)
        .fit(&dataset)?;
    let assignments = model.predict(&matrix);

    let clusters: Vec<u32> = assignments.iter().map(|&c| c as u32).collect();
    let mut df = df;
    let cluster_series = Series::new("cluster".into(), clusters);
    df.with_column(cluster_series)?;

    let summary_df = compute_cluster_summary(&df)?;
    let (patterns, decisions) = build_patterns(&summary_df, cfg);
    let signals = build_signals(&df, &decisions);

    if let Some(path) = cfg.patterns_path.as_ref() {
        write_patterns(path, &patterns)?;
    }
    if let Some(path) = cfg.signals_path.as_ref() {
        write_signals(path, &signals)?;
    }

    Ok(StrategyOutput { patterns, signals })
}

pub fn display_report(output: &StrategyOutput) {
    println!(
        "Generated {} patterns and {} signals",
        output.patterns.len(),
        output.signals.len()
    );

    for pattern in &output.patterns {
        println!(
            "cluster {} ⇒ samples: {}, mean_ret: {:.5}, win_rate: {:.3}, t_stat: {:.2}, action: {}",
            pattern.cluster_id,
            pattern.sample_count,
            pattern.mean_return,
            pattern.win_rate,
            pattern.t_stat,
            pattern.action.as_deref().unwrap_or("no-trade")
        );
    }
}

fn load_features(path: &Path) -> Result<DataFrame> {
    LazyCsvReader::new(path)
        .with_has_header(true)
        .with_infer_schema_length(Some(100))
        .finish()
        .with_context(|| format!("reading features CSV from {}", path.display()))?
        .collect()
        .context("parsing features CSV")
}

fn enrich_features(df: &mut DataFrame) -> Result<()> {
    let height = df.height();
    for &name in FEATURE_COLUMNS.iter().take(FEATURE_COLUMNS.len() - 1) {
        if df.get_column_index(name).is_none() {
            let series = Series::new(name.into(), vec![0.0f64; height]);
            df.with_column(series)?;
            continue;
        }

        let series = df.column(name)?.cast(&DataType::Float64)?;
        let filled = series.fill_null(FillNullStrategy::Zero)?;
        df.with_column(filled)?;
    }

    let account_series = df.column("account_value")?.cast(&DataType::Float64)?;
    let realized_series = df.column("realized_pnl")?.cast(&DataType::Float64)?;
    let account = account_series.f64().with_context(|| "account_value not float")?;
    let realized = realized_series.f64().with_context(|| "realized_pnl not float")?;
    let mut expected = Vec::with_capacity(height);
    for i in 0..height {
        let acc = account.get(i).unwrap_or(0.0);
        let pnl = realized.get(i).unwrap_or(0.0);
        let value = if acc.abs() > 1e-9 { pnl / acc } else { pnl };
        expected.push(value);
    }
    df.with_column(Series::new("expected_return".into(), expected))?;

    for name in ["win_rate", "leverage"].iter() {
        let series = df.column(name)?.cast(&DataType::Float64)?;
        let filled = series.fill_null(FillNullStrategy::Zero)?;
        df.with_column(filled)?;
    }

    Ok(())
}

fn build_feature_matrix(df: &DataFrame) -> Result<Array2<f64>> {
    let n = df.height();
    let m = FEATURE_COLUMNS.len();
    let mut mat = Array2::<f64>::zeros((n, m));

    for (j, name) in FEATURE_COLUMNS.iter().enumerate() {
        let series = df
            .column(name)
            .with_context(|| format!("missing column {name}"))?
            .cast(&DataType::Float64)?;
        let col = series
            .f64()
            .with_context(|| format!("column {name} not f64"))?;
        for i in 0..n {
            let val = col.get(i).unwrap_or(0.0);
            mat[[i, j]] = val;
        }
    }

    // Standardize columns (zero mean, unit variance)
    let (rows, cols) = mat.dim();
    for j in 0..cols {
        let mut mean = 0.0;
        for i in 0..rows {
            mean += mat[[i, j]];
        }
        mean /= rows as f64;

        let mut var = 0.0;
        for i in 0..rows {
            let d = mat[[i, j]] - mean;
            var += d * d;
        }
        let denom = (rows as f64 - 1.0).max(1.0);
        let std = (var / denom).sqrt().max(1e-9);
        for i in 0..rows {
            mat[[i, j]] = (mat[[i, j]] - mean) / std;
        }
    }

    Ok(mat)
}

fn compute_cluster_summary(df: &DataFrame) -> Result<DataFrame> {
    df.clone()
        .lazy()
        .group_by([col("cluster")])
        .agg([
            len().alias("sample_count"),
            col("expected_return").mean().alias("mean_return"),
            col("expected_return").median().alias("median_return"),
            col("expected_return").std(1).alias("return_std"),
            col("expected_return")
                .quantile(lit(0.1), QuantileInterpolOptions::Nearest)
                .alias("p10"),
            col("expected_return")
                .quantile(lit(0.25), QuantileInterpolOptions::Nearest)
                .alias("p25"),
            col("expected_return")
                .quantile(lit(0.75), QuantileInterpolOptions::Nearest)
                .alias("p75"),
            col("expected_return")
                .quantile(lit(0.9), QuantileInterpolOptions::Nearest)
                .alias("p90"),
            col("win_rate").mean().alias("avg_win_rate"),
            col("total_volume").mean().alias("avg_volume"),
            col("leverage").mean().alias("avg_leverage"),
        ])
        .sort_by_exprs(
            vec![col("cluster")],
            SortMultipleOptions::new().with_order_descending(false),
        )
        .collect()
        .context("computing cluster summary")
}

fn build_patterns(
    summary: &DataFrame,
    cfg: &Config,
) -> (Vec<PatternSummary>, HashMap<u32, ClusterDecision>) {
    let clusters = summary.column("cluster").unwrap().u32().unwrap();
    let counts = summary.column("sample_count").unwrap().u32().unwrap();
    let means = summary.column("mean_return").unwrap().f64().unwrap();
    let medians = summary.column("median_return").unwrap().f64().unwrap();
    let stds = summary.column("return_std").unwrap().f64().unwrap();
    let p10 = summary.column("p10").unwrap().f64().unwrap();
    let p25 = summary.column("p25").unwrap().f64().unwrap();
    let p75 = summary.column("p75").unwrap().f64().unwrap();
    let p90 = summary.column("p90").unwrap().f64().unwrap();
    let win_rates = summary.column("avg_win_rate").unwrap().f64().unwrap();
    let volumes = summary.column("avg_volume").unwrap().f64().unwrap();
    let leverages = summary.column("avg_leverage").unwrap().f64().unwrap();

    let mut patterns = Vec::new();
    let mut decisions = HashMap::new();

    for i in 0..summary.height() {
        let cluster_id = clusters.get(i).unwrap_or(0);
        let sample_count = counts.get(i).unwrap_or(0) as usize;
        let mean_return = means.get(i).unwrap_or(0.0);
        let median_return = medians.get(i).unwrap_or(0.0);
        let return_std = stds.get(i).unwrap_or(0.0).abs();
        let q10 = p10.get(i).unwrap_or(0.0);
        let q25 = p25.get(i).unwrap_or(0.0);
        let q75 = p75.get(i).unwrap_or(0.0);
        let q90 = p90.get(i).unwrap_or(0.0);
        let win_rate = win_rates.get(i).unwrap_or(0.0).max(0.0);
        let avg_volume = volumes.get(i).unwrap_or(0.0);
        let avg_leverage = leverages.get(i).unwrap_or(0.0);

        let t_stat = if sample_count > 1 && return_std.is_finite() && return_std > 0.0 {
            let std_err = return_std / (sample_count as f64).sqrt();
            if std_err > 0.0 {
                mean_return / std_err
            } else {
                0.0
            }
        } else {
            0.0
        };

        let confidence = t_stat.abs();

        let mut action = None;
        let mut stop = None;
        let mut take = None;

        let adjusted_win = cfg.min_win_rate - (0.1 / (sample_count.max(1) as f64).sqrt());

        if sample_count >= cfg.min_cluster_size
            && win_rate >= adjusted_win
            && confidence >= cfg.min_t_stat
        {
            if mean_return >= cfg.min_edge {
                action = Some("LONG".to_string());
                stop = Some(q25.abs().max(cfg.min_edge));
                take = Some(q90.max(cfg.min_edge));
            } else if mean_return <= -cfg.min_edge {
                action = Some("SHORT".to_string());
                stop = Some(q75.abs().max(cfg.min_edge));
                take = Some((-q10).max(cfg.min_edge));
            }

            if let Some(ref chosen) = action {
                decisions.insert(
                    cluster_id,
                    ClusterDecision {
                        action: chosen.clone(),
                        confidence,
                    },
                );
            }
        }

        patterns.push(PatternSummary {
            cluster_id,
            sample_count,
            mean_return,
            median_return,
            return_std,
            win_rate,
            avg_volume,
            avg_leverage,
            confidence,
            t_stat,
            action,
            stop_loss: stop,
            take_profit: take,
        });
    }

    if decisions.is_empty() {
        if let Some((idx, best)) = patterns
            .iter()
            .enumerate()
            .filter(|(_, p)| p.mean_return > cfg.min_edge)
            .max_by(|(_, a), (_, b)| a
                .mean_return
                .partial_cmp(&b.mean_return)
                .unwrap_or(std::cmp::Ordering::Equal))
        {
            let new_conf = best.mean_return.abs() / (best.return_std + cfg.min_edge);
            decisions.insert(
                best.cluster_id,
                ClusterDecision {
                    action: "LONG".to_string(),
                    confidence: new_conf,
                },
            );
            if let Some(entry) = patterns.get_mut(idx) {
                entry.action = Some("LONG".to_string());
                entry.confidence = new_conf;
                entry.stop_loss = Some((entry.return_std + cfg.min_edge).max(cfg.min_edge));
                entry.take_profit = Some((entry.mean_return.abs() + entry.return_std).max(cfg.min_edge));
            }
        } else if let Some((idx, best)) = patterns
            .iter()
            .enumerate()
            .filter(|(_, p)| p.mean_return < -cfg.min_edge)
            .min_by(|(_, a), (_, b)| a
                .mean_return
                .partial_cmp(&b.mean_return)
                .unwrap_or(std::cmp::Ordering::Equal))
        {
            let new_conf = best.mean_return.abs() / (best.return_std + cfg.min_edge);
            decisions.insert(
                best.cluster_id,
                ClusterDecision {
                    action: "SHORT".to_string(),
                    confidence: new_conf,
                },
            );
            if let Some(entry) = patterns.get_mut(idx) {
                entry.action = Some("SHORT".to_string());
                entry.confidence = new_conf;
                entry.stop_loss = Some((entry.return_std + cfg.min_edge).max(cfg.min_edge));
                entry.take_profit = Some((entry.mean_return.abs() + entry.return_std).max(cfg.min_edge));
            }
        }
    }

    (patterns, decisions)
}

fn build_signals(df: &DataFrame, decisions: &HashMap<u32, ClusterDecision>) -> Vec<StrategySignal> {
    let mut signals = Vec::new();
    if decisions.is_empty() {
        return signals;
    }

    let addresses = df.column("address").unwrap().str().unwrap();
    let expected = df.column("expected_return").unwrap().f64().unwrap();
    let win_rates = df.column("win_rate").unwrap().f64().unwrap();
    let clusters = df.column("cluster").unwrap().u32().unwrap();

    for i in 0..df.height() {
        let cluster = clusters.get(i).unwrap_or(0);
        if let Some(decision) = decisions.get(&cluster) {
            let expected_return = expected.get(i).unwrap_or(0.0);
            match decision.action.as_str() {
                "LONG" if expected_return <= 0.0 => continue,
                "SHORT" if expected_return >= 0.0 => continue,
                _ => {}
            }

            let addr = addresses.get(i).unwrap_or("");
            let row_conf = win_rates.get(i).unwrap_or(0.0).max(0.0);
            let confidence = (decision.confidence * row_conf).max(0.0);

            signals.push(StrategySignal {
                address: addr.to_string(),
                cluster_id: cluster,
                action: decision.action.clone(),
                expected_return,
                confidence,
            });
        }
    }

    signals
}

fn write_patterns(path: &Path, patterns: &[PatternSummary]) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    serde_json::to_writer_pretty(file, patterns).context("writing patterns JSON")
}

fn write_signals(path: &Path, signals: &[StrategySignal]) -> Result<()> {
    let mut writer =
        csv::Writer::from_path(path).with_context(|| format!("creating {}", path.display()))?;
    writer.write_record([
        "address",
        "cluster_id",
        "action",
        "expected_return",
        "confidence",
    ])?;
    for sig in signals {
        writer.write_record([
            sig.address.as_str(),
            &sig.cluster_id.to_string(),
            sig.action.as_str(),
            &format!("{:.6}", sig.expected_return),
            &format!("{:.6}", sig.confidence),
        ])?;
    }
    writer.flush()?;
    Ok(())
}
