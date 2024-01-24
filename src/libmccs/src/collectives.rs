use crate::checked_cuda;
use crate::Error;
use crate::MccsCommunicatorHandle;
use crate::MCCS_CTX;
use crate::MCCS_STREAM_SYNC;
use crate::{rx_recv_impl, DevicePtr};
use cuda_runtime_sys::cudaStreamWaitEvent;
use cuda_runtime_sys::{cudaEventRecord, cudaStream_t};
use ipc::mccs::command::{AllGather, Command, CompletionKind};

pub fn all_gather(
    comm: MccsCommunicatorHandle,
    send_buf: DevicePtr,
    recv_buf: DevicePtr,
    size: usize,
    stream: cudaStream_t,
) -> Result<(), Error> {
    unsafe {
        let user_event = MCCS_STREAM_SYNC.with_borrow(|sync| *sync.get(&stream).unwrap());
        checked_cuda!(cudaEventRecord(user_event, stream));
    };
    let op = AllGather {
        comm: comm.comm_handle,
        send_buf: send_buf.backup_mem,
        recv_buf: recv_buf.backup_mem,
        size,
        user_stream: stream as usize,
    };
    MCCS_CTX.with(move |ctx| {
        let req = Command::AllGather(op);
        ctx.service.send_cmd(req)?;
        rx_recv_impl!(ctx.service, CompletionKind::AllGather)
    })?;

    unsafe {
        checked_cuda!(cudaStreamWaitEvent(stream, comm.backend_event, 0));
    }

    Ok(())
}
