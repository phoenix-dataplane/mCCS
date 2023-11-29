use std::ffi::c_void;

use crate::cuda::mapped_ptr::DeviceHostPtr;
use crate::transport::meta::{RecvBufMeta, SendBufMeta};

use super::buffer::BufferMap;
use super::provider::{AnyMrHandle, AnyNetComm, NetProvierWrap};
use crate::transport::transporter::ConnectHandle;
use crate::transport::NUM_PROTOCOLS;

pub struct AgentSetupRequest {
    pub(crate) rank: usize,
    pub(crate) local_rank: usize,
    pub(crate) remote_rank: usize,
    pub(crate) net_device: usize,
    pub(crate) use_gdr: bool,
    pub(crate) need_flush: bool,
    pub(crate) provider: &'static dyn NetProvierWrap,
}

pub struct AgentSendConnectRequest {
    handle: ConnectHandle,
}

pub struct AgentSendSetup {
    pub(crate) rank: usize,
    pub(crate) local_rank: usize,
    pub(crate) remote_rank: usize,
    pub(crate) net_device: usize,
    pub(crate) use_gdr: bool,
    pub(crate) use_dma_buf: bool,
    pub(crate) max_recvs: usize,
    pub(crate) provider: &'static dyn NetProvierWrap,
}

// https://github.com/NVIDIA/nccl/blob/v2.17.1-1/src/transport/net.cc#L84
pub struct AgentSendResources {
    pub(crate) map: BufferMap,
    pub(crate) send_comm: AnyNetComm,
    pub(crate) send_mem: DeviceHostPtr<SendBufMeta>,
    pub(crate) recv_mem: DeviceHostPtr<RecvBufMeta>,
    pub(crate) rank: usize,
    pub(crate) local_rank: usize,
    pub(crate) remote_rank: usize,
    pub(crate) net_device: usize,
    pub(crate) use_gdr: bool,
    pub(crate) use_dma_buf: bool,
    pub(crate) need_flush: bool,
    pub(crate) max_recvs: usize,
    pub(crate) gdc_sync: *mut u64,
    // gdr_desc
    pub(crate) buffers: [*mut c_void; NUM_PROTOCOLS],
    pub(crate) buffer_sizes: [usize; NUM_PROTOCOLS],
    pub(crate) mr_handles: [AnyMrHandle; NUM_PROTOCOLS],
    pub(crate) step: u64,
    pub(crate) provider: &'static dyn NetProvierWrap,
}

pub struct AgentRecvConnectRequest {
    agent_rank: usize,
}

pub struct AgentRecvSetup {
    pub(crate) listen_comm: AnyNetComm,
    pub(crate) rank: usize,
    pub(crate) local_rank: usize,
    pub(crate) remote_rank: usize,
    pub(crate) net_device: usize,
    pub(crate) use_gdr: bool,
    pub(crate) use_dma_buf: bool,
    pub(crate) need_flush: bool,
    pub(crate) max_recvs: usize,
    pub(crate) provider: &'static dyn NetProvierWrap,
}

// https://github.com/NVIDIA/nccl/blob/v2.17.1-1/src/transport/net.cc#L84
pub struct AgentRecvResources {
    pub(crate) map: BufferMap,
    pub(crate) recv_comm: AnyNetComm,
    pub(crate) send_mem: DeviceHostPtr<SendBufMeta>,
    pub(crate) recv_mem: DeviceHostPtr<RecvBufMeta>,
    pub(crate) rank: usize,
    pub(crate) local_rank: usize,
    pub(crate) remote_rank: usize,
    pub(crate) agent_rank: usize,
    pub(crate) net_device: usize,
    pub(crate) use_gdr: bool,
    pub(crate) use_dma_buf: bool,
    pub(crate) need_flush: bool,
    pub(crate) max_recvs: usize,
    pub(crate) gdc_sync: *mut u64,
    pub(crate) gdc_flush: *mut u64,
    // gdr_desc
    pub(crate) buffers: [*mut c_void; NUM_PROTOCOLS],
    pub(crate) buffer_sizes: [usize; NUM_PROTOCOLS],
    pub(crate) mr_handles: [AnyMrHandle; NUM_PROTOCOLS],
    pub(crate) step: u64,
    pub(crate) provider: &'static dyn NetProvierWrap,
}
