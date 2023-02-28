use std::ptr::NonNull;
use std::any::Any;

use super::engine::TransportEngineId;
use super::transporter::Transporter;
use crate::pattern::RingPattern;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConnType {
    Send,
    Recv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PeerConnId {
    pub(crate) peer_rank: usize,
    pub(crate) channel: u32,
    pub(crate) conn_index: u32,
    pub(crate) conn_type: ConnType,
}

pub struct PeerConnInfo {
    bufs: [NonNull<u8>; 3],
    head: NonNull<u64>,
    tail: NonNull<u64>,
    slots_sizes: NonNull<[i32]>,
}

pub struct PeerConnector {
    pub conn_index: u32,
    pub conn_info: PeerConnInfo,
    pub transport_agent_engine: Option<TransportEngineId>,
    pub transporter: &'static dyn Transporter,
    pub transport_resources: Box<dyn Any>,
}

pub struct ChannelPeerConn {
    pub send: Vec<PeerConnector>,
    pub recv: Vec<PeerConnector>,
    pub peer_rank: usize,
}

pub struct CommChannel {
    pub id: u32,
    pub peers: Vec<ChannelPeerConn>,
    pub ring: RingPattern,
}