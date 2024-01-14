use std::net::IpAddr;

use ipc::mccs::command::{Command, CommunicatorInit, CompletionKind};
use ipc::mccs::handle::CommunicatorHandle;

use crate::rx_recv_impl;
use crate::Error;
use crate::MCCS_CTX;

pub fn init_communicator_rank(
    unique_id: u32,
    rank: usize,
    num_ranks: usize,
    cuda_device_idx: i32,
    root_addr: IpAddr,
) -> Result<CommunicatorHandle, Error> {
    // TODO
    let init = CommunicatorInit {
        id: unique_id,
        rank,
        num_ranks,
        cuda_device_idx,
        root_addr: todo!(),
    };
    MCCS_CTX.with(move |ctx| {
        let req = Command::InitCommunicator(init);
        ctx.service.send_cmd(req)?;
        rx_recv_impl!(ctx.service, CompletionKind::InitCommunicator, handle, {
            Ok(handle)
        })
    })
}
