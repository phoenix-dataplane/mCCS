use crate::daemon::DaemonId;

use crate::transport::engine::TransportEngineId;
use crate::transport::message::{TransportEngineReply, TransportEngineRequest};
use crate::utils::duplex_chan::DuplexChannel;

pub enum ControlRequest {
    NewDaemon(DaemonId),
    NewTransportEngine(TransportEngineId),
}

pub enum ControlNotification {
    NewDaemon {
        id: TransportEngineId,
    },
    NewTransportEngine {
        id: TransportEngineId,
        chan: DuplexChannel<TransportEngineRequest, TransportEngineReply>,
    },
}
