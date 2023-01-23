use ipc::mccs::command::{Command, CompletionKind, AllGather};
use ipc::mccs::handle::CommunicatorHandle;

use crate::MCCS_CTX;
use crate::rx_recv_impl;
use crate::Error;


pub fn all_gather(
    comm: CommunicatorHandle,
    // TBD
    send_buf: (),
    recv_buf: (),
    size: usize,
) -> Result<(), Error> {
    let op = AllGather {
        comm,
        send_buf,
        recv_buf,
        size,
    };
    MCCS_CTX.with(move |ctx| {
        let req = Command::AllGather(op);
        ctx.service.send_cmd(req)?;
        rx_recv_impl!(ctx.service, CompletionKind::AllGather)?;
        Ok(())
    })
}