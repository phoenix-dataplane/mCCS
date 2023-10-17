use crate::checked_cuda;
use cuda_runtime_sys::{
    cudaEventCreateWithFlags, cudaEventDisableTiming, cudaEventInterprocess, cudaEventRecord,
    cudaEvent_t, cudaIpcEventHandle_t, cudaIpcGetEventHandle, cudaIpcOpenEventHandle, cudaStream_t,
};
use ipc::mccs::command::{AllGather, Command, CompletionKind};
use ipc::mccs::handle::CommunicatorHandle;

use crate::Error;
use crate::MCCS_CTX;
use crate::{rx_recv_impl, DevicePtr};

pub fn all_gather(
    comm: CommunicatorHandle,
    send_buf: DevicePtr,
    recv_buf: DevicePtr,
    size: usize,
    stream: cudaStream_t,
) -> Result<cudaEvent_t, Error> {
    let handle = unsafe {
        let mut event = std::ptr::null_mut();
        checked_cuda!(cudaEventCreateWithFlags(
            &mut event,
            cudaEventInterprocess | cudaEventDisableTiming
        ));
        checked_cuda!(cudaEventRecord(event, stream));
        let mut handle = cudaIpcEventHandle_t::default();
        checked_cuda!(cudaIpcGetEventHandle(&mut handle, event));
        handle
    };
    let op = AllGather {
        comm,
        send_buf: send_buf.backup_mem,
        recv_buf: recv_buf.backup_mem,
        size,
        ipc_event_handle: handle.into(),
    };
    MCCS_CTX.with(move |ctx| {
        let req = Command::AllGather(op);
        ctx.service.send_cmd(req)?;
        rx_recv_impl!(ctx.service, CompletionKind::AllGather, handle, {
            let mut event = std::ptr::null_mut();
            checked_cuda!(unsafe { cudaIpcOpenEventHandle(&mut event, handle.into()) });
            Ok(event)
        })
    })
}
