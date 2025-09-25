#[derive(Clone, Debug)]
pub struct HyperliquidEndpoints {
    pub info: String,
    pub exchange: String,
    pub websocket: String,
    pub evm: String,
}

impl HyperliquidEndpoints {
    pub fn new(testnet: bool) -> Self {
        if testnet {
            Self {
                info: "https://api.hyperliquid-testnet.xyz/info".into(),
                exchange: "https://api.hyperliquid-testnet.xyz/exchange".into(),
                websocket: "wss://api.hyperliquid-testnet.xyz/ws".into(),
                evm: "https://api.hyperliquid-testnet.xyz".into(),
            }
        } else {
            Self {
                info: "https://api.hyperliquid.xyz/info".into(),
                exchange: "https://api.hyperliquid.xyz/exchange".into(),
                websocket: "wss://api.hyperliquid.xyz/ws".into(),
                evm: "https://api.hyperliquid.xyz".into(),
            }
        }
    }
}
