use std::net::IpAddr;

use crate::comm::CommunicatorId;
use ipc::mccs::handle::CudaEventHandle;

pub struct InitCommunicator {
    pub communicator_id: CommunicatorId,
    pub root_mccs_addr: IpAddr,
    pub rank: usize,
    pub num_ranks: usize,
}

pub struct AllGatherRequest {
    pub communicator_id: CommunicatorId,
    pub send_buf_addr: usize,
    pub recv_buf_addr: usize,
    pub size: usize,
    pub app_ipc_event_handle: CudaEventHandle,
}

pub enum ProxyCommand {
    InitCommunicator(InitCommunicator),
    AllGather(AllGatherRequest),
    DestroyCommunicator(CommunicatorId),
}

pub enum ProxyCompletion {
    InitCommunicator,
    AllGather(CudaEventHandle),
}
