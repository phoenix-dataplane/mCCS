use std::net::SocketAddr;

use crate::bootstrap::BootstrapHandle;
use crate::comm::CommunicatorId;

pub enum ExchangeCommand {
    RegisterBootstrapHandle(CommunicatorId, BootstrapHandle),
    // communicator id, root mccs exchange engine listen addr
    RecvBootstrapHandle(CommunicatorId, SocketAddr),
}

pub enum ExchangeCompletion {
    RegisterBootstrapHandle,
    RecvBootstrapHandle(CommunicatorId, BootstrapHandle),
}
