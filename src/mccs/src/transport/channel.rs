use std::any::Any;
use std::collections::HashMap;

use super::engine::TransportEngineId;
use super::transporter::Transporter;
use super::NUM_PROTOCOLS;
use crate::cuda::ptr::DeviceNonNull;
use crate::pattern::RingPattern;
use std::fmt::Display;

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

pub const CHANNEL_MAX_CONNS: usize = 2;

pub struct ChannelPeerConn {
    // conn_index -> PeerConnector
    pub send: [Option<PeerConnector>; CHANNEL_MAX_CONNS],
    // conn_index -> PeerConnector
    pub recv: [Option<PeerConnector>; CHANNEL_MAX_CONNS],
}

pub struct CommChannel {
    // peer -> ChannelPeerConn
    pub peers: HashMap<usize, ChannelPeerConn>,
    pub ring: RingPattern,
    pub work_queue_next_available: u32,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChannelId(pub u32);

impl Display for ChannelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.to_string().as_str())
    }
}
