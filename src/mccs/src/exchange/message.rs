use std::net::SocketAddr;

use serde::{Serialize, Deserialize};

use crate::bootstrap::BootstrapHandle;
use crate::comm::CommunicatorId;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExchangeMessage {
    BootstrapHandle(CommunicatorId, BootstrapHandle),
    BootstrapHandleRequest(CommunicatorId, SocketAddr),
}