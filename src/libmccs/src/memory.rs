use std::os::raw::c_void;

use cuda_runtime_sys::cudaIpcMemLazyEnablePeerAccess;
use cuda_runtime_sys::cudaIpcOpenMemHandle;
use cuda_runtime_sys::{cudaError, cudaIpcMemHandle_t};
use ipc::mccs::command::{Command, CompletionKind};

use crate::rx_recv_impl;
use crate::Error;
use crate::MCCS_CTX;

pub fn cuda_malloc(device_idx: usize, size: usize) -> Result<*mut c_void, Error> {
    MCCS_CTX.with(|ctx| {
        let req = Command::CudaMalloc(device_idx, size);
        ctx.service.send_cmd(req)?;
        rx_recv_impl!(ctx.service, CompletionKind::CudaMalloc, handle, {
            let mut dev_ptr: *mut c_void = std::ptr::null_mut();
            let handle = cudaIpcMemHandle_t { reserved: handle.0 };
            let err = unsafe {
                cudaIpcOpenMemHandle(
                    &mut dev_ptr as *mut _,
                    handle,
                    cudaIpcMemLazyEnablePeerAccess,
                )
            };
            if err != cudaError::cudaSuccess {
                return Err(Error::Cuda(err));
            }
            Ok(dev_ptr)
        })
    })
}
