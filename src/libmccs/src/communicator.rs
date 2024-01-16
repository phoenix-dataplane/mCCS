use std::net::IpAddr;

use cuda_runtime_sys::cudaIpcEventHandle_t;
use cuda_runtime_sys::cudaStream_t;
use cuda_runtime_sys::{cudaEventCreateWithFlags, cudaIpcGetEventHandle, cudaIpcOpenEventHandle};
use cuda_runtime_sys::{cudaEventDisableTiming, cudaEventInterprocess};
use ipc::mccs::command::{Command, CommunicatorInit, CompletionKind};

use crate::checked_cuda;
use crate::rx_recv_impl;
use crate::Error;
use crate::MccsCommunicatorHandle;
use crate::{MCCS_CTX, MCCS_STREAM_SYNC};

pub fn init_communicator_rank(
    unique_id: u32,
    rank: usize,
    num_ranks: usize,
    cuda_device_idx: i32,
    root_addr: IpAddr,
) -> Result<MccsCommunicatorHandle, Error> {
    let init = CommunicatorInit {
        id: unique_id,
        rank,
        num_ranks,
        root_addr,
        cuda_device_idx,
    };
    let (comm_handle, event_handle) = MCCS_CTX.with(move |ctx| {
        let req = Command::InitCommunicator(init);
        ctx.service.send_cmd(req)?;
        rx_recv_impl!(ctx.service, CompletionKind::InitCommunicator, handles, {
            Ok(handles)
        })
    })?;
    let mut event = std::ptr::null_mut();
    unsafe {
        checked_cuda!(cudaIpcOpenEventHandle(&mut event, event_handle.into()));
    };
    let handle = MccsCommunicatorHandle {
        comm_handle,
        backend_event: event,
    };
    Ok(handle)
}

pub fn register_stream(cuda_dev: i32, stream: cudaStream_t) -> Result<(), Error> {
    let mut event = std::ptr::null_mut();
    let mut event_handle = cudaIpcEventHandle_t::default();
    unsafe {
        checked_cuda!(cudaEventCreateWithFlags(
            &mut event,
            cudaEventInterprocess | cudaEventDisableTiming
        ));
        checked_cuda!(cudaIpcGetEventHandle(&mut event_handle, event));
    }
    MCCS_STREAM_SYNC.with_borrow_mut(|sync| {
        sync.insert(stream, event);
    });

    MCCS_CTX.with(move |ctx| {
        let req = Command::RegisterStream(cuda_dev, stream.addr(), event_handle.into());
        ctx.service.send_cmd(req)?;
        rx_recv_impl!(ctx.service, CompletionKind::RegisterStream)
    })?;
    Ok(())
}
