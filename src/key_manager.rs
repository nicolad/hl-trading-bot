use std::{fs, path::Path};

use chrono::Utc;
use serde::Serialize;

use crate::errors::BotError;

#[derive(Clone, Debug, Serialize)]
pub struct KeyInfo {
    pub network: String,
    pub key_source: Option<String>,
    pub key_found: bool,
    pub warnings: Vec<String>,
    pub error: Option<String>,
    pub checked_at: chrono::DateTime<Utc>,
}

#[derive(Clone, Default)]
pub struct KeyManager;

impl KeyManager {
    pub fn get_private_key(
        &self,
        testnet: bool,
        bot_config: Option<&serde_json::Value>,
    ) -> Result<String, BotError> {
        let network = if testnet { "testnet" } else { "mainnet" };
        if let Some(value) = bot_config.and_then(|cfg| self.bot_key(cfg, testnet)) {
            return Ok(value);
        }
        if let Some(value) = self.env_key(testnet) {
            return Ok(value);
        }
        if let Some(value) = self.env_key_legacy() {
            return Ok(value);
        }
        if let Some(value) = self.file_key(testnet) {
            return Ok(value);
        }
        if let Some(value) = self.file_key_legacy() {
            return Ok(value);
        }
        Err(BotError::Configuration(format!(
            "no private key found for {}",
            network
        )))
    }

    fn bot_key(&self, config: &serde_json::Value, testnet: bool) -> Option<String> {
        let select = if testnet {
            ["testnet_private_key", "private_key"]
        } else {
            ["mainnet_private_key", "private_key"]
        };
        for key in select {
            if let Some(value) = config.get(key).and_then(|v| v.as_str()) {
                return Some(self.normalize_key(value));
            }
        }
        let file_key = if testnet {
            ["testnet_key_file", "private_key_file"]
        } else {
            ["mainnet_key_file", "private_key_file"]
        };
        for key in file_key {
            if let Some(path) = config.get(key).and_then(|v| v.as_str())
                && let Some(value) = self.read_key_file(path)
            {
                return Some(value);
            }
        }
        None
    }

    fn env_key(&self, testnet: bool) -> Option<String> {
        let env_var = if testnet {
            "HYPERLIQUID_TESTNET_PRIVATE_KEY"
        } else {
            "HYPERLIQUID_MAINNET_PRIVATE_KEY"
        };
        std::env::var(env_var)
            .ok()
            .map(|value| self.normalize_key(&value))
    }

    fn env_key_legacy(&self) -> Option<String> {
        std::env::var("HYPERLIQUID_PRIVATE_KEY")
            .ok()
            .map(|value| self.normalize_key(&value))
    }

    fn file_key(&self, testnet: bool) -> Option<String> {
        let env_var = if testnet {
            "HYPERLIQUID_TESTNET_KEY_FILE"
        } else {
            "HYPERLIQUID_MAINNET_KEY_FILE"
        };
        std::env::var(env_var)
            .ok()
            .and_then(|file| self.read_key_file(&file))
    }

    fn file_key_legacy(&self) -> Option<String> {
        std::env::var("HYPERLIQUID_PRIVATE_KEY_FILE")
            .ok()
            .and_then(|file| self.read_key_file(&file))
    }

    fn read_key_file(&self, path: &str) -> Option<String> {
        let file = Path::new(path);
        let contents = fs::read_to_string(file).ok()?;
        let trimmed = contents.trim();
        if trimmed.is_empty() {
            return None;
        }
        let normalized = self.normalize_key(trimmed);
        if normalized.len() == 66 {
            Some(normalized)
        } else {
            None
        }
    }

    fn normalize_key(&self, key: &str) -> String {
        if key.starts_with("0x") {
            key.to_lowercase()
        } else {
            format!("0x{}", key.to_lowercase())
        }
    }

    pub fn key_info(&self, testnet: bool, bot_config: Option<&serde_json::Value>) -> KeyInfo {
        let network = if testnet { "testnet" } else { "mainnet" }.to_string();
        let mut info = KeyInfo {
            network,
            key_source: None,
            key_found: false,
            warnings: Vec::new(),
            error: None,
            checked_at: Utc::now(),
        };
        let result = self
            .get_private_key(testnet, bot_config)
            .map(|_| ())
            .map_err(|err| err.to_string());
        match result {
            Ok(_) => {
                info.key_found = true;
                info.key_source = Some("resolved".to_string());
            }
            Err(err) => {
                info.error = Some(err);
            }
        }
        info
    }
}

pub static KEY_MANAGER: KeyManager = KeyManager;
