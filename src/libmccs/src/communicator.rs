use ipc::mccs::command::{Command, CompletionKind, CommunicatorInit};
use ipc::mccs::handle::CommunicatorHandle;

use crate::MCCS_CTX;
use crate::rx_recv_impl;
use crate::Error;


pub fn init_communicator_rank(
    unique_id: u32,
    rank: usize,
    num_ranks: usize,
    cuda_device_idx: usize,
) -> Result<CommunicatorHandle, Error> {
    let init = CommunicatorInit {
        id: unique_id,
        rank,
        num_ranks,
        cuda_device_idx,
    };
    MCCS_CTX.with(move |ctx| {
        let req = Command::InitCommunicator(init);
        ctx.service.send_cmd(req)?;
        rx_recv_impl!(ctx.service, CompletionKind::InitCommunicator, handle, {
            Ok(handle)
        })
    })
}