use std::net::SocketAddr;

use serde::{Deserialize, Serialize};

use ipc::mccs::reconfig::ExchangeReconfigCommand;

use crate::bootstrap::BootstrapHandle;
use crate::comm::CommunicatorId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExchangeProxyMessage {
    BootstrapHandle(CommunicatorId, BootstrapHandle),
    BootstrapHandleRequest(CommunicatorId, SocketAddr),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExchangeMessage {
    ProxyMessage(ExchangeProxyMessage),
    ReconfigCommand(ExchangeReconfigCommand),
}
