use anyhow::{Result, bail};
use nautilus_indicators::average::ema::ExponentialMovingAverage;
use nautilus_indicators::indicator::{Indicator, MovingAverage};
use serde_json::json;
use serde_yaml::{Value, to_value};

use crate::config::BotConfig;
use crate::interfaces::{MarketData, Position, SignalType, TradingSignal, TradingStrategy};

#[derive(Clone, Debug)]
pub struct HyperliquidConfig {
    pub app_symbol: String,
    pub hl_symbol: String,
    pub tick_size: f64,
    pub step_size: f64,
    pub min_size: f64,
    pub leverage: Option<f64>,
    pub tif: OrderTimeInForce,
    pub use_reduce_only: bool,
    pub post_only: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum OrderTimeInForce {
    Gtc,
    Ioc,
    Fok,
}

impl ToString for OrderTimeInForce {
    fn to_string(&self) -> String {
        match self {
            OrderTimeInForce::Gtc => "GTC".to_string(),
            OrderTimeInForce::Ioc => "IOC".to_string(),
            OrderTimeInForce::Fok => "FOK".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HyperliquidAdapter {
    cfg: HyperliquidConfig,
}

impl HyperliquidAdapter {
    pub fn new(cfg: HyperliquidConfig) -> Self {
        Self { cfg }
    }

    pub fn hl_symbol(&self) -> &str {
        &self.cfg.hl_symbol
    }

    pub fn round_size(&self, raw: f64) -> f64 {
        let step = self.cfg.step_size.max(1e-12);
        (raw / step).round() * step
    }

    pub fn clamp_min_size(&self, sz: f64) -> f64 {
        if sz.abs() < self.cfg.min_size {
            if sz.is_sign_negative() {
                -self.cfg.min_size
            } else {
                self.cfg.min_size
            }
        } else {
            sz
        }
    }

    pub fn round_price(&self, raw: f64) -> f64 {
        let tick = self.cfg.tick_size.max(1e-12);
        (raw / tick).round() * tick
    }

    pub fn default_order_metadata(&self, reduce_only: bool, maybe_price: Option<f64>) -> Value {
        to_value(&json!({
            "venue": "hyperliquid",
            "symbol": self.hl_symbol(),
            "tif": self.cfg.tif.to_string(),
            "post_only": self.cfg.post_only,
            "reduce_only": reduce_only && self.cfg.use_reduce_only,
            "leverage": self.cfg.leverage,
            "price": maybe_price,
        }))
        .unwrap_or(Value::Null)
    }
}

impl Default for HyperliquidConfig {
    fn default() -> Self {
        Self {
            app_symbol: "BTC-PERP".to_string(),
            hl_symbol: "BTC".to_string(),
            tick_size: 0.1,
            step_size: 0.001,
            min_size: 0.001,
            leverage: Some(1.0),
            tif: OrderTimeInForce::Gtc,
            use_reduce_only: true,
            post_only: false,
        }
    }
}

pub struct EmaStrategy {
    symbol: String,
    order_size: f64,
    fast: ExponentialMovingAverage,
    slow: ExponentialMovingAverage,
    last_fast: Option<f64>,
    last_slow: Option<f64>,
    net_position: f64,
    hl: HyperliquidAdapter,
}

impl EmaStrategy {
    pub fn new(config: &BotConfig) -> Result<Self> {
        let ema_cfg = &config.strategy.ema;

        if ema_cfg.order_size <= 0.0 {
            bail!("order_size must be > 0");
        }

        let hl_cfg = Self::create_hyperliquid_config(&config.grid.symbol);

        Ok(Self {
            symbol: config.grid.symbol.clone(),
            order_size: ema_cfg.order_size,
            fast: ExponentialMovingAverage::new(ema_cfg.fast_period, None),
            slow: ExponentialMovingAverage::new(ema_cfg.slow_period, None),
            last_fast: None,
            last_slow: None,
            net_position: 0.0,
            hl: HyperliquidAdapter::new(hl_cfg),
        })
    }

    fn create_hyperliquid_config(symbol: &str) -> HyperliquidConfig {
        let (hl_symbol, tick_size, step_size, min_size) = match symbol {
            "BTC" | "BTC-PERP" => ("BTC", 0.1, 0.001, 0.001),
            "ETH" | "ETH-PERP" => ("ETH", 0.01, 0.0001, 0.0001),
            "SOL" | "SOL-PERP" => ("SOL", 0.001, 0.001, 0.001),
            _ => {
                let base_symbol = symbol.split('-').next().unwrap_or(symbol);
                (base_symbol, 0.001, 0.001, 0.001)
            }
        };

        HyperliquidConfig {
            app_symbol: symbol.to_string(),
            hl_symbol: hl_symbol.to_string(),
            tick_size,
            step_size,
            min_size,
            leverage: Some(1.0),
            tif: OrderTimeInForce::Gtc,
            use_reduce_only: true,
            post_only: false,
        }
    }

    fn position_from(&self, positions: &[Position]) -> f64 {
        positions
            .iter()
            .find(|p| p.asset == self.symbol)
            .map(|p| p.size)
            .unwrap_or(self.net_position)
    }

    fn build_signal(
        &self,
        signal_type: SignalType,
        reduce_only: bool,
        maybe_price: Option<f64>,
    ) -> TradingSignal {
        let sz = self.hl.clamp_min_size(self.hl.round_size(self.order_size));
        let px = maybe_price.map(|p| self.hl.round_price(p));

        TradingSignal {
            signal_type,
            asset: self.hl.hl_symbol().to_string(),
            size: sz,
            price: px,
            reason: Some("ema_crossover".to_string()),
            metadata: self.hl.default_order_metadata(reduce_only, px),
        }
    }
}

impl TradingStrategy for EmaStrategy {
    fn generate_signals(
        &mut self,
        market_data: &MarketData,
        positions: &[Position],
        _balance: f64,
    ) -> Result<Vec<TradingSignal>> {
        self.fast.update_raw(market_data.price);
        self.slow.update_raw(market_data.price);

        let fast_value = self.fast.value();
        let slow_value = self.slow.value();
        let mut signals = Vec::new();

        if !self.fast.initialized() || !self.slow.initialized() {
            self.last_fast = Some(fast_value);
            self.last_slow = Some(slow_value);
            return Ok(signals);
        }

        let prev_fast = self.last_fast.unwrap_or(fast_value);
        let prev_slow = self.last_slow.unwrap_or(slow_value);
        let current_pos = self.position_from(positions);

        if fast_value > slow_value && prev_fast <= prev_slow && current_pos <= 0.0 {
            signals.push(self.build_signal(SignalType::Buy, false, None));
        } else if fast_value < slow_value && prev_fast >= prev_slow && current_pos >= 0.0 {
            signals.push(self.build_signal(SignalType::Sell, true, None));
        }

        self.last_fast = Some(fast_value);
        self.last_slow = Some(slow_value);
        Ok(signals)
    }

    fn on_trade_executed(
        &mut self,
        signal: &TradingSignal,
        _executed_price: f64,
        executed_size: f64,
    ) -> Result<()> {
        let signed = match signal.signal_type {
            SignalType::Buy => executed_size,
            SignalType::Sell => -executed_size,
            _ => 0.0,
        };
        self.net_position += signed;
        Ok(())
    }

    fn name(&self) -> &str {
        "ema_hyperliquid"
    }

    fn start(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.last_fast = None;
        self.last_slow = None;
        self.net_position = 0.0;
    }

    fn get_status(&self) -> serde_yaml::Value {
        to_value(&json!({
            "app_symbol": self.symbol,
            "hl_symbol": self.hl.hl_symbol(),
            "fast": {
                "period": self.fast.period,
                "value": self.fast.value(),
                "initialized": self.fast.initialized(),
            },
            "slow": {
                "period": self.slow.period,
                "value": self.slow.value(),
                "initialized": self.slow.initialized(),
            },
            "order": {
                "step_size": self.hl.cfg.step_size,
                "tick_size": self.hl.cfg.tick_size,
                "min_size": self.hl.cfg.min_size,
                "tif": self.hl.cfg.tif.to_string(),
                "post_only": self.hl.cfg.post_only,
                "reduce_only_default": self.hl.cfg.use_reduce_only,
                "leverage": self.hl.cfg.leverage,
            },
            "net_position": self.net_position,
        }))
        .unwrap_or(Value::Null)
    }
}
