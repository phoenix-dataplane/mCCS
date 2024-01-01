pub mod agent;
pub mod buffer;
pub mod provider;
pub mod resources;
pub mod transporter;

use provider::NetProviderError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Net provider error: {0}")]
    NetProvider(#[from] NetProviderError),
    #[error("Ring buffer registration error: {0}")]
    BufferRegistration(String),
}
