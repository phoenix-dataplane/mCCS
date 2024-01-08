use std::net::SocketAddr;

use crate::comm::CommunicatorId;
use crate::bootstrap::BootstrapHandle;


pub enum ExchangeCommand {
    RegisterBootstrapHandle(CommunicatorId, BootstrapHandle),
    // communicator id, root mccs exchange engine listen addr
    RecvBootstrapHandle(CommunicatorId, SocketAddr),
}

pub enum ExchangeCompletion {
    RegisterBootstrapHandle,
    RecvBootstrapHandle(CommunicatorId, BootstrapHandle),
}
