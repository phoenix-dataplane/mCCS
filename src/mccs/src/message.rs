use crate::daemon::DaemonId;
use crate::proxy::command::{ProxyCommand, ProxyCompletion};

use crate::transport::engine::TransportEngineId;
use crate::transport::message::{TransportEngineReply, TransportEngineRequest};
use crate::utils::duplex_chan::DuplexChannel;

pub enum ControlRequest {
    NewDaemon(DaemonId),
    NewTransportEngine(TransportEngineId),
}

pub enum ControlCommand {
    NewDaemon {
        id: DaemonId,
        chan: DuplexChannel<ProxyCompletion, ProxyCommand>,
    },
    NewTransportEngine {
        id: TransportEngineId,
        chan: DuplexChannel<TransportEngineRequest, TransportEngineReply>,
    },
}
