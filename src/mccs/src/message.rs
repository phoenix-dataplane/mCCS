use crossbeam::channel::{Receiver, Sender};

use crate::transport::engine::TransportEngineId;
use crate::transport::message::{TransportEngineReply, TransportEngineRequest};

pub enum ControlRequest {
    NewTransportEngine(TransportEngineId),
}

pub enum ControlNotification {
    NewTransportEngine {
        id: TransportEngineId,
        request_tx: Sender<TransportEngineRequest>,
        reply_rx: Receiver<TransportEngineReply>,
    },
}
