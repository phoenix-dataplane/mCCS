use std::sync::Arc;

use serde::{Deserialize, Serialize};

use cuda_runtime_sys::{cudaEvent_t, cudaStream_t};

use super::buffer::TransportBuffer;
use super::config::ShmLocality;
use crate::cuda::alloc::{DeviceAlloc, DeviceHostMapped};
use crate::cuda::ptr::DeviceNonNull;
use crate::transport::meta::{RecvBufMeta, SendBufMeta};
use crate::transport::{NUM_BUFFER_SLOTS, NUM_PROTOCOLS};

#[derive(Clone)]
pub struct ShmSendSetupResources {
    pub buf: Arc<TransportBuffer<SendBufMeta>>,
    pub buf_sizes: [usize; NUM_PROTOCOLS],
    pub locality: ShmLocality,
    // use_memcpy_send
    pub use_memcpy: bool,
    pub recv_use_memcpy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShmConnectHandle {
    pub buf_arc_ptr: usize,
}

#[derive(Clone)]
pub struct ShmRecvSetupResources {
    pub buf: Arc<TransportBuffer<RecvBufMeta>>,
    pub buf_sizes: [usize; NUM_PROTOCOLS],
    pub locality: ShmLocality,
    // use_memcpy_recv
    pub use_memcpy: bool,
}

pub struct ShmConnectedResources {
    pub sender_buf: Arc<TransportBuffer<SendBufMeta>>,
    pub sender_buf_dev: DeviceHostMapped<SendBufMeta>,
    pub receiver_buf: Arc<TransportBuffer<RecvBufMeta>>,
    pub receiver_buf_dev: DeviceHostMapped<RecvBufMeta>,
    pub buf_sizes: [usize; NUM_PROTOCOLS],
    pub locality: ShmLocality,
}

unsafe impl Send for ShmConnectedResources {}

pub struct ShmAgentRequest {
    pub locality: ShmLocality,
    pub buf_sizes: [usize; NUM_PROTOCOLS],
    pub sender_meta: Arc<TransportBuffer<SendBufMeta>>,
    pub receiver_meta: Arc<TransportBuffer<RecvBufMeta>>,
}

pub struct ShmAgentReply {
    pub meta_sync: DeviceNonNull<RecvBufMeta>,
    pub device_buf: DeviceNonNull<u8>,
}

unsafe impl Send for ShmAgentRequest {}
unsafe impl Send for ShmAgentReply {}

pub struct ShmAgentResources {
    // shared between the agent and GPU kernels
    // allocated via cudaHostAlloc, dropped via cudaFreeHost
    // by the agent
    pub meta_sync: DeviceHostMapped<RecvBufMeta>,
    // pointer to the buffer of sender buf or receiver buf
    // in sender_meta or receiver_meta
    // agent only handles SIMPLE protocol
    pub host_buf: *mut u8,
    // device private buffer shared between the agent and GPU kernels
    // allocated and dropped by the agent
    pub device_buf: DeviceAlloc<u8>,
    // buffer size for SIMPLE protocol
    pub buf_size: usize,
    pub sender_meta: Arc<TransportBuffer<SendBufMeta>>,
    pub receiver_meta: Arc<TransportBuffer<RecvBufMeta>>,
    // resources used only by the agent
    // step for progress mark
    // stream to launch async memcpy
    // events for checking completions
    pub step: u64,
    pub stream: cudaStream_t,
    pub events: [cudaEvent_t; NUM_BUFFER_SLOTS],
}

unsafe impl Send for ShmAgentResources {}
