use serde::{Deserialize, Serialize};

use crate::handle::{CommunicatorHandle, CudaMemHandle};

type IResult<T> = Result<T, ipc_core::control::Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicatorInit {
    pub id: u32,
    pub rank: usize,
    pub num_ranks: usize,
    pub cuda_device_idx: usize,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct MccsDeviceMemoryHandle {
    pub id: u64,
    pub offset: usize,
    pub len: usize,
}

impl MccsDeviceMemoryHandle {
    pub fn add(&self, len: usize) -> Result<Self, ()> {
        if self.offset + len > self.len {
            Err(())
        } else {
            Ok(Self {
                id: self.id,
                offset: self.offset + len,
                len: self.len,
            })
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllGather {
    pub comm: CommunicatorHandle,
    pub send_buf: MccsDeviceMemoryHandle,
    pub recv_buf: MccsDeviceMemoryHandle,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    // device, size
    CudaMalloc(usize, usize),
    InitCommunicator(CommunicatorInit),
    AllGather(AllGather),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum CompletionKind {
    CudaMalloc((CudaMemHandle, MccsDeviceMemoryHandle)),
    InitCommunicator(CommunicatorHandle),
    AllGather,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Completion(pub IResult<CompletionKind>);
