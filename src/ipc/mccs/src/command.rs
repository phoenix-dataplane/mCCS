use serde::{Deserialize, Serialize};

use crate::handle::{CudaMemHandle, CommunicatorHandle};

type IResult<T> = Result<T, ipc_core::control::Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicatorInit {
    pub id: u32,
    pub rank: usize,
    pub num_ranks: usize,
    pub cuda_device_idx: usize,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllGather {
    pub comm: CommunicatorHandle,
    // TBD, use some handles 
    pub send_buf: (),
    pub recv_buf: (),
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    // device, size
    CudaMalloc(usize, usize),
    InitCommunicator(CommunicatorInit),
    AllGather(AllGather)
}

#[derive(Debug, Serialize, Deserialize)]
pub enum CompletionKind {
    CudaMalloc(CudaMemHandle),
    InitCommunicator(CommunicatorHandle),
    AllGather,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Completion(pub IResult<CompletionKind>);
