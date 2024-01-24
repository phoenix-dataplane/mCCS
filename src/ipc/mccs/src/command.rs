use std::net::IpAddr;

use serde::{Deserialize, Serialize};

use crate::handle::{CommunicatorHandle, CudaEventHandle, CudaMemHandle};

type IResult<T> = Result<T, ipc_core::control::Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicatorInit {
    pub id: u32,
    pub rank: usize,
    pub num_ranks: usize,
    pub root_addr: IpAddr,
    pub cuda_device_idx: i32,
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
    // user stream handle
    pub user_stream: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollOperation {
    AllGather(AllGather),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    // device, size
    CudaMalloc(i32, usize),
    InitCommunicator(CommunicatorInit),
    AllGather(AllGather),
    // different requests can be scheduled on different user stream
    // however, currently, they must belong to the same communicator
    GroupCall(Vec<CollOperation>),
    // cuda device, user stream handle, IPC event handle for user event
    // currently, a user CUDA stream (and a communicator)
    // could only be used by a single thread
    RegisterStream(i32, usize, CudaEventHandle), // TODO: deregister stream, destroy communicator
}

#[derive(Debug, Serialize, Deserialize)]
pub enum CompletionKind {
    CudaMalloc((CudaMemHandle, MccsDeviceMemoryHandle)),
    // communicator handle and IPC event handle for the backend stream
    InitCommunicator((CommunicatorHandle, CudaEventHandle)),
    AllGather,
    GroupCall,
    RegisterStream,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Completion(pub IResult<CompletionKind>);
