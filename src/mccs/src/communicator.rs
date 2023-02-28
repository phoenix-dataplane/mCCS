use std::net::SocketAddr;

use crate::transport::channel::CommChannel;
use crate::pattern::RingPattern;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CommunicatorId(u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostIdent(pub SocketAddr);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerType {
    Local,
    IntraNode,
    InterNode,
}

#[derive(Clone)]
pub struct PeerInfo {
    pub peer_type: PeerType,
    pub host: HostIdent,
    pub cuda_device_idx: usize,
}

pub struct Communicator {
    pub id: CommunicatorId,
    pub rank: usize,
    pub num_ranks: usize,
    pub peers_info: Vec<PeerInfo>,
    pub channels: Vec<CommChannel>,
}

pub struct ChannelCommPattern {
    // channel id
    pub channel: u32,
    pub ring: RingPattern,
}