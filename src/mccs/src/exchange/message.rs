use std::net::SocketAddr;

use serde::{Deserialize, Serialize};

use crate::bootstrap::BootstrapHandle;
use crate::comm::CommunicatorId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExchangeMessage {
    BootstrapHandle(CommunicatorId, BootstrapHandle),
    BootstrapHandleRequest(CommunicatorId, SocketAddr),
}
