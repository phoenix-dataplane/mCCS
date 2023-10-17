use crate::comm::CommunicatorId;
use ipc::mccs::handle::CudaEventHandle;

pub struct InitCommunicator {
    pub communicator_id: CommunicatorId,
    pub rank: usize,
    pub num_ranks: usize,
}

pub struct AllGather {
    pub communicator_id: CommunicatorId,
    pub send_buf_addr: usize,
    pub recv_buf_addr: usize,
    pub size: usize,
}

pub enum ProxyCommand {
    InitCommunicator(InitCommunicator),
    AllGather(AllGather),
}

pub enum ProxyCompletion {
    InitCommunicator,
    AllGather(CudaEventHandle),
}
