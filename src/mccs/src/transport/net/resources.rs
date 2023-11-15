use crate::cuda::alloc::DeviceHostMapped;
use crate::transport::meta::{SendBufMeta, RecvBufMeta};

use super::buffer::BufferMap;
use super::provider::AnyNetComm;

pub struct AgentSetupRequest {
    rank: usize,
    local_rank: usize,
    remote_rank: usize,
    net_device: usize,
    use_gdr: bool,
    need_flush: bool,
}

pub struct AgentRecvSetup {
    listen_comm: AnyNetComm,
    rank: usize,
    local_rank: usize,
    remote_rank: usize,
    net_device: usize,
    use_gdr: bool,
    use_dma_buf: bool,
    need_flush: bool,
    max_recvs: usize,
}

pub struct AgentRecvResources {
    map: BufferMap,
    recv_comm: AnyNetComm,
    send_mem: DeviceHostMapped<SendBufMeta>,
    recv_mem: DeviceHostMapped<RecvBufMeta>,
    rank: usize,

}