use serde::{Deserialize, Serialize};

use crate::handle::CudaMemHandle;

type IResult<T> = Result<T, ipc_core::control::Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    CudaMalloc(usize),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum CompletionKind {
    CudaMalloc(CudaMemHandle),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Completion(pub IResult<CompletionKind>);
