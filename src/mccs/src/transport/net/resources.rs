use std::ffi::c_void;
use std::pin::Pin;

use crate::cuda::mapped_ptr::DeviceHostPtr;
use crate::transport::meta::{RecvBufMeta, SendBufMeta};

use super::buffer::BufferMap;
use super::provider::{AnyMrHandle, AnyNetComm, NetProvierWrap};
use crate::transport::transporter::ConnectHandle;
use crate::transport::NUM_PROTOCOLS;

pub struct NetSendSetup {
    pub(crate) agent_rank: usize,
}

pub struct NetSendResources {
    pub(crate) map: BufferMap,
}

pub struct NetRecvResources {
    pub(crate) map: BufferMap,
}

pub struct AgentSetupRequest {
    pub(crate) rank: usize,
    pub(crate) local_rank: usize,
    pub(crate) remote_rank: usize,
    pub(crate) net_device: usize,
    pub(crate) use_gdr: bool,
    pub(crate) need_flush: bool,
    pub(crate) buffer_sizes: [usize; NUM_PROTOCOLS],
    pub(crate) provider: &'static dyn NetProvierWrap,
    pub(crate) udp_sport: Option<u16>,
    pub(crate) tc: Option<u8>,
}

pub struct AgentSendSetup {
    pub(crate) rank: usize,
    pub(crate) local_rank: usize,
    pub(crate) remote_rank: usize,
    pub(crate) net_device: usize,
    pub(crate) use_gdr: bool,
    pub(crate) use_dma_buf: bool,
    pub(crate) max_recvs: usize,
    pub(crate) buffer_sizes: [usize; NUM_PROTOCOLS],
    pub(crate) provider: &'static dyn NetProvierWrap,
    pub(crate) gdr_copy_sync_enable: bool,
    pub(crate) udp_sport: Option<u16>,
    pub(crate) tc: Option<u8>,
}

pub struct AgentSendConnectRequest {
    pub(crate) handle: ConnectHandle,
}

pub struct AgentSendConnectReply {
    pub(crate) map: BufferMap,
    pub(crate) agent_cuda_dev: i32,
}

// https://github.com/NVIDIA/nccl/blob/v2.17.1-1/src/transport/net.cc#L84
pub struct AgentSendResources {
    pub(crate) map: BufferMap,
    pub(crate) mr_handles: [Option<Box<AnyMrHandle>>; NUM_PROTOCOLS],
    pub(crate) send_comm: Pin<Box<AnyNetComm>>,
    pub(crate) send_mem: DeviceHostPtr<SendBufMeta>,
    pub(crate) recv_mem: DeviceHostPtr<RecvBufMeta>,
    pub(crate) rank: usize,
    pub(crate) local_rank: usize,
    pub(crate) remote_rank: usize,
    pub(crate) net_device: usize,
    pub(crate) use_gdr: bool,
    pub(crate) use_dma_buf: bool,
    pub(crate) max_recvs: usize,
    pub(crate) gdc_sync: *mut u64,
    // gdr_desc
    pub(crate) buffers: [*mut c_void; NUM_PROTOCOLS],
    pub(crate) buffer_sizes: [usize; NUM_PROTOCOLS],
    pub(crate) step: u64,
    pub(crate) provider: &'static dyn NetProvierWrap,
    pub(crate) qos_round: u64,
}

unsafe impl Send for AgentSendResources {}

pub struct AgentRecvSetup {
    pub(crate) listen_comm: Box<AnyNetComm>,
    pub(crate) rank: usize,
    pub(crate) local_rank: usize,
    pub(crate) remote_rank: usize,
    pub(crate) net_device: usize,
    pub(crate) use_gdr: bool,
    pub(crate) use_dma_buf: bool,
    pub(crate) need_flush: bool,
    pub(crate) max_recvs: usize,
    pub(crate) buffer_sizes: [usize; NUM_PROTOCOLS],
    pub(crate) provider: &'static dyn NetProvierWrap,
    pub(crate) gdr_copy_sync_enable: bool,
    pub(crate) gdr_copy_flush_enable: bool,
    pub(crate) tc: Option<u8>,
}

pub struct AgentRecvSetupReply {
    pub(crate) handle: ConnectHandle,
}

pub struct AgentRecvConnectRequest {
    pub(crate) send_agent_rank: usize,
}

pub struct AgentRecvConnectReply {
    pub(crate) map: BufferMap,
}

// https://github.com/NVIDIA/nccl/blob/v2.17.1-1/src/transport/net.cc#L84
pub struct AgentRecvResources {
    pub(crate) map: BufferMap,
    pub(crate) mr_handles: [Option<Box<AnyMrHandle>>; NUM_PROTOCOLS],
    pub(crate) recv_comm: Pin<Box<AnyNetComm>>,
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
    pub(crate) step: u64,
    pub(crate) provider: &'static dyn NetProvierWrap,
}

unsafe impl Send for AgentRecvResources {}
