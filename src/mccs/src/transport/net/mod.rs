pub mod agent;
pub mod buffer;
pub mod config;
pub mod provider;
pub mod resources;
pub mod transporter;

use thiserror::Error;

use crate::transport::transporter::ConnectHandleError;
use provider::NetProviderError;
pub use transporter::NET_TRANSPORT;

#[derive(Debug, Error)]
pub enum NetTransportError {
    #[error("Failed to downcast setup resources")]
    DowncastSetupResources,
    #[error("Failed to downcast agent reply")]
    DowncastAgentReply,
    #[error("Invalid agent reply")]
    InvalidAgentReply,
    #[error("Connection handle: {0}")]
    ConnectionHandle(#[from] ConnectHandleError),
    #[error("Net provider error: {0}")]
    NetProvider(#[from] NetProviderError),
}

#[derive(Debug, Error)]
pub enum NetAgentError {
    #[error("Net provider error: {0}")]
    NetProvider(#[from] NetProviderError),
    #[error("Ring buffer registration error: {0}")]
    BufferRegistration(String),
    #[error("Failed to downcast agent request")]
    DowncastAgentRequest,
    #[error("Failed to downcast agent resources")]
    DowncastAgentResources,
    #[error("Transport catalog error: {0}")]
    TransportCatalog(#[from] crate::transport::catalog::Error),
}
