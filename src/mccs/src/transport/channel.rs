use std::any::Any;
use std::collections::HashMap;

use super::engine::TransportEngineId;
use super::transporter::Transporter;
use super::NUM_PROTOCOLS;
use crate::cuda::ptr::DeviceNonNull;
use crate::pattern::RingPattern;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConnType {
    Send,
    Recv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PeerConnId {
    pub(crate) peer_rank: usize,
    pub(crate) channel: ChannelId,
    pub(crate) conn_index: u32,
    pub(crate) conn_type: ConnType,
}

pub struct PeerConnInfo {
    pub bufs: [DeviceNonNull<u8>; NUM_PROTOCOLS],
    pub head: DeviceNonNull<u64>,
    pub tail: DeviceNonNull<u64>,
    pub slots_size: Option<DeviceNonNull<u32>>,
}

pub struct PeerConnector {
    pub conn_info: PeerConnInfo,
    pub transport_agent_engine: Option<TransportEngineId>,
    pub transporter: &'static dyn Transporter,
    pub transport_resources: Box<dyn Any>,
}

pub struct ChannelPeerConn {
    // conn_index -> PeerConnector
    pub send: HashMap<u32, PeerConnector>,
    // conn_index -> PeerConnector
    pub recv: HashMap<u32, PeerConnector>,
}

pub struct CommChannel {
    // peer -> ChannelPeerConn
    pub peers: HashMap<usize, ChannelPeerConn>,
    pub ring: RingPattern,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChannelId(pub u32);
