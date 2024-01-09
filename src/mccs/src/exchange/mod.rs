pub mod command;
pub mod engine;
pub mod message;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExchangeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),
}
