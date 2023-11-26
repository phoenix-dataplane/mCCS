use thiserror::Error;

use super::buffer::BufferMap;
use super::resources::{AgentRecvConnectRequest, AgentRecvSetup, AgentRecvResources};
use super::provider::NetProviderError;


#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Net provider error: {0}")]
    NetProvider(#[from] NetProviderError),
}

type Result<T> = std::result::Result<T, AgentError>;

pub async fn net_agent_send_connect(
    message: AgentRecvConnectRequest, 
    setup_resources: AgentRecvSetup
) -> Result<(BufferMap, AgentRecvResources)> {
    let provider = setup_resources.provider;
    let send_comm = provider.accept(setup_resources.listen_comm).await?;
    let map = BufferMap::new();
    
    todo!()
}