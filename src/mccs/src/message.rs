use crate::daemon::DaemonId;
use crate::proxy::command::{ProxyCommand, ProxyCompletion};
use cuda_runtime_sys::cudaStream_t;

use crate::transport::engine::TransportEngineId;
use crate::transport::message::{TransportEngineReply, TransportEngineRequest};
use crate::utils::duplex_chan::DuplexChannel;

pub enum ControlRequest {
    NewTransportEngine(TransportEngineId),
}

pub enum ControlNotification {
    NewDaemon {
        id: DaemonId,
        chan: DuplexChannel<ProxyCompletion, ProxyCommand>,
    },
    NewTransportEngine {
        id: TransportEngineId,
        chan: DuplexChannel<TransportEngineRequest, TransportEngineReply>,
    },
}

#[derive(Debug, Clone)]
pub struct CudaStream(usize);

impl Into<cudaStream_t> for CudaStream {
    fn into(self) -> cudaStream_t {
        self.0 as cudaStream_t
    }
}

impl From<cudaStream_t> for CudaStream {
    fn from(value: cudaStream_t) -> Self {
        Self(value as usize)
    }
}
