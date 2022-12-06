use std::os::raw::c_void;

use ipc::mccs::command::{Command, CompletionKind};
use cuda_runtime_sys::{cudaIpcMemHandle_t, cudaError};
use cuda_runtime_sys::cudaIpcOpenMemHandle;
use cuda_runtime_sys::cudaIpcMemLazyEnablePeerAccess;

use crate::MCCS_CTX;
use crate::rx_recv_impl;
use crate::Error;

pub fn cuda_malloc(size: usize) -> Result<*mut c_void, Error> {
    MCCS_CTX.with(|ctx| {
        let req = Command::CudaMalloc(size);
        ctx.service.send_cmd(req)?;
        rx_recv_impl!(ctx.service, CompletionKind::CudaMalloc, handle, {
            let mut dev_ptr: *mut c_void = std::ptr::null_mut();
            let handle = cudaIpcMemHandle_t { reserved: handle.0 };
            let err = unsafe { cudaIpcOpenMemHandle(
                &mut dev_ptr as *mut _, 
                handle, 
                cudaIpcMemLazyEnablePeerAccess
            )};
            if err != cudaError::cudaSuccess {
                return Err(Error::Cuda(err))
            }
            Ok(dev_ptr)
        })
    })
}