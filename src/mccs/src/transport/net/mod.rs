pub mod resources;
pub mod buffer;
pub mod agent;
pub mod transporter;
pub mod provider;

use thiserror::Error;
use provider::NetProviderError;


#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Net provider error: {0}")]
    NetProvider(#[from] NetProviderError),
    #[error("Ring buffer registration error: {0}")]
    BufferRegistration(String),
}

