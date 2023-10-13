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
) -> Result<(), Error> {
    let op = AllGather {
        comm,
        send_buf: send_buf.backup_mem,
        recv_buf: recv_buf.backup_mem,
        size,
    };
    MCCS_CTX.with(move |ctx| {
        let req = Command::AllGather(op);
        ctx.service.send_cmd(req)?;
        rx_recv_impl!(ctx.service, CompletionKind::AllGather)?;
        Ok(())
    })
}
