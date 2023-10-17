use crate::comm::CommunicatorId;
use crate::message::CudaStream;
use ipc::mccs::handle::CudaEventHandle;

pub struct InitCommunicator {
    pub communicator_id: CommunicatorId,
    pub rank: usize,
    pub num_ranks: usize,
}

pub struct AllGatherRequest {
    pub communicator_id: CommunicatorId,
    pub send_buf_addr: usize,
    pub recv_buf_addr: usize,
    pub size: usize,
    pub app_ipc_event_handle: CudaEventHandle,
    pub daemon_stream: CudaStream,
}

pub enum ProxyCommand {
    InitCommunicator(InitCommunicator),
    AllGather(AllGatherRequest),
}

pub enum ProxyCompletion {
    InitCommunicator,
    AllGather(CudaEventHandle),
}
