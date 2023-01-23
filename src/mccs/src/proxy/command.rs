use crossbeam::channel::{Sender, Receiver};

use crate::daemon::DaemonId;

pub struct InitCommunicator {
    pub daemon_id: DaemonId,
    pub communicator_id: u32,
    pub rank: usize,
    pub num_ranks: usize,
}

pub struct AllGather {
    pub daemon_id: DaemonId,
    pub communicator_id: u32,
    pub send_buf_addr: usize,
    pub recv_buf_addr: usize,
    pub size: usize,
}

pub enum ProxyCommand {
    InitCommunicator(InitCommunicator),
    AllGather(AllGather),
}

pub enum ProxyCompletion {
    InitCommunicator,
    AllGather,
}

#[allow(unused)]
pub struct CommandEndpointProxy {
    pub daemon_id: DaemonId,
    pub command_rx: Receiver<ProxyCommand>,
    pub completion_tx: Sender<ProxyCompletion>,
}