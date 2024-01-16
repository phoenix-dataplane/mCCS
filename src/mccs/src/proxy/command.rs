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
    // user stream handle
    pub user_stream: usize,
}

pub enum CollRequest {
    AllGather(AllGatherRequest),
}

pub enum ProxyCommand {
    InitCommunicator(InitCommunicator),
    // user stream and user event IPC handle
    RegisterStream(usize, CudaEventHandle),
    AllGather(AllGatherRequest),
    GroupCall(Vec<CollRequest>),
    DestroyCommunicator(CommunicatorId),
}

pub enum ProxyCompletion {
    InitCommunicator(CudaEventHandle),
    RegisterStream,
    AllGather,
    GroupCall,
}
