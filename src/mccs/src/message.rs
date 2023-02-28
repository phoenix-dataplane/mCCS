use crossbeam::channel::{Sender, Receiver};

use crate::transport::engine::TransportEngineId;
use crate::transport::message::{TransportEngineRequest, TransportEngineReply};

pub enum ControlRequest {
    NewTransportEngine(TransportEngineId)
}

pub enum ControlNotification {
    NewTransportEngine {
        id: TransportEngineId,
        request_tx: Sender<TransportEngineRequest>,
        reply_rx: Receiver<TransportEngineReply>,
    },
}