use std::collections::HashMap;

use crate::communicator::PeerInfo;
use crate::daemon::DaemonId;

pub struct LocalCommunicatorBootstrap {
    pub daemon_id: DaemonId,
    pub communicator_id: u32,
    pub rank: usize,
    pub peers_info: HashMap<usize, PeerInfo>,
    pub num_ranks: usize,
}

pub struct SetupTransport {
    pub daemon_id: DaemonId,
    pub communicator_id: u32,
}

pub struct AllGather {
    pub daemon_id: DaemonId,
    pub communicator_id: u32,
    pub send_buf: *mut u8,
    pub recv_buf: *mut u8,
    pub size: usize,
    pub step: u32,
    pub recv_completed: bool,
    pub send_completed: bool,
}

unsafe impl Send for AllGather {} 
unsafe impl Sync for AllGather {}

pub enum ProxyOp {
    InitCommunicator(LocalCommunicatorBootstrap),
    InitCommSetupTransport(SetupTransport),
    AllGather(AllGather),
}
