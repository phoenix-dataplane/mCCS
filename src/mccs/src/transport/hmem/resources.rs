use std::sync::Arc;

use cuda_runtime_sys::{cudaStream_t, cudaEvent_t};

use crate::transport::buffer::{RecvBufMeta, SendBufMeta, TransportBuffer};
use super::config::HostMemTptConfig;

#[allow(unused)]
pub struct HostMemTptAgentResource {
    progress_exchange: RecvBufMeta,
    buf_device: *mut u8,
    buf_host: *mut u8,

    step: u64,
    stream: cudaStream_t,
    events: [cudaEvent_t; 3],
}

pub struct HostMemTptSetupSender {
    pub host_mem: Arc<TransportBuffer<SendBufMeta>>,
    pub config: HostMemTptConfig,
}

pub struct HostMemTptSetupReceiver {
    pub host_mem: Arc<TransportBuffer<RecvBufMeta>>,
    pub config: HostMemTptConfig,
}

pub struct HostMemTptResource { 
    pub sender_host_mem: Arc<TransportBuffer<SendBufMeta>>,
    pub sender_device_mem: *mut SendBufMeta,
    pub receiver_host_mem: Arc<TransportBuffer<RecvBufMeta>>,
    pub receiver_device_mem: *mut RecvBufMeta,
}

// TBD
unsafe impl Send for HostMemTptResource {} 
unsafe impl Sync for HostMemTptResource {}