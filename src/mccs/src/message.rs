use crate::daemon::DaemonId;
use crossbeam::channel::{Receiver, Sender};

use crate::transport::engine::TransportEngineId;
use crate::transport::message::{TransportEngineReply, TransportEngineRequest};

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
        request_tx: Sender<TransportEngineRequest>,
        reply_rx: Receiver<TransportEngineReply>,
    },
}
