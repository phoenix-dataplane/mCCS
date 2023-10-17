use std::os::raw::c_char;

use cuda_runtime_sys::cudaIpcEventHandle_t;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaMemHandle(#[serde(with = "BigArray")] pub [c_char; 64usize]);

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CommunicatorHandle(pub u64);

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaEventHandle(#[serde(with = "BigArray")] pub [c_char; 64usize]);

impl From<cudaIpcEventHandle_t> for CudaEventHandle {
    fn from(value: cudaIpcEventHandle_t) -> Self {
        Self(value.reserved)
    }
}

impl Into<cudaIpcEventHandle_t> for CudaEventHandle {
    fn into(self) -> cudaIpcEventHandle_t {
        cudaIpcEventHandle_t { reserved: self.0 }
    }
}
