use thiserror::Error;

#[derive(Debug, Error)]
pub enum BotError {
    #[error("configuration error: {0}")]
    Configuration(String),
    #[error("strategy error: {0}")]
    Strategy(String),
    #[error("exchange error: {0}")]
    Exchange(String),
    #[error("order error: {0}")]
    Order(String),
    #[error("position error: {0}")]
    Position(String),
    #[error("grid error: {0}")]
    Grid(String),
    #[error("trading error: {0}")]
    Trading(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
