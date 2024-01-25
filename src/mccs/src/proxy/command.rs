use std::net::IpAddr;

use crate::comm::CommunicatorId;
use ipc::mccs::handle::CudaEventHandle;

use super::task::{TaskDataType, TaskReduceOpType};

pub struct InitCommunicator {
    pub communicator_id: CommunicatorId,
    pub root_mccs_addr: IpAddr,
    pub rank: usize,
    pub num_ranks: usize,
}

#[derive(Clone, Debug)]
pub struct AllGatherRequest {
    pub communicator_id: CommunicatorId,
    pub send_buf_addr: usize,
    pub recv_buf_addr: usize,
    pub size: usize,
    // user stream handle
    pub user_stream: usize,
}

#[derive(Clone, Debug)]
pub struct AllReduceRequest {
    pub communicator_id: CommunicatorId,
    pub send_buf_addr: usize,
    pub recv_buf_addr: usize,
    pub size: usize,
    pub data_type: TaskDataType,
    pub op_type: TaskReduceOpType,
    // user stream handle
    pub user_stream: usize,
}

pub enum CollRequest {
    AllGather(AllGatherRequest),
    AllReduce(AllReduceRequest),
}

pub enum ProxyCommand {
    InitCommunicator(InitCommunicator),
    // user stream and user event IPC handle
    RegisterStream(usize, CudaEventHandle),
    AllGather(AllGatherRequest),
    AllReduce(AllReduceRequest),
    GroupCall(Vec<CollRequest>),
    DestroyCommunicator(CommunicatorId),
}

pub enum ProxyCompletion {
    InitCommunicator(CudaEventHandle),
    RegisterStream,
    AllGather,
    AllReduce,
    GroupCall,
}
