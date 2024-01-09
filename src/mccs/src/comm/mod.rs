pub mod device;
pub mod profile;

pub use profile::CommProfile;

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;

use bytes::{Buf, BufMut};
use serde::{Deserialize, Serialize};

use crate::pattern::RingPattern;
use crate::proxy::plan::{ChanWorkSchedule, KernelPlan};
use crate::proxy::task::TaskQueue;
use crate::transport::channel::CommChannel;
use crate::utils::tcp;

use cuda_runtime_sys::{cudaEvent_t, cudaStream_t};
use device::CommDevResources;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CommunicatorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerType {
    Local,
    IntraNode,
    InterNode,
}

#[derive(Clone)]
pub struct PeerInfo {
    pub rank: usize,
    pub local_rank: usize,
    pub peer_type: PeerType,
    pub host: SocketAddr,
    pub cuda_device_idx: i32,
}

pub const PEER_INFO_EXCHANGE_SEND_SIZE: usize = 40;

pub struct PeerInfoExchange {
    pub rank: usize,
    pub host: SocketAddr,
    pub cuda_device_idx: i32,
}

impl PeerInfoExchange {
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u64(self.rank as u64);
        buf.put_i32(self.cuda_device_idx);
        tcp::encode_socket_addr(&self.host, buf);
    }

    pub fn decode<B: Buf>(buf: &mut B) -> Self {
        let rank = buf.get_u64() as usize;
        let cuda_idx = buf.get_i32();
        let host = tcp::decode_socket_addr(buf);
        Self {
            rank,
            host,
            cuda_device_idx: cuda_idx,
        }
    }
}

pub struct Communicator {
    pub id: CommunicatorId,
    pub rank: usize,
    pub num_ranks: usize,
    pub peers_info: Vec<PeerInfo>,
    // channel_id -> CommChannel
    pub channels: HashMap<u32, CommChannel>,
    pub profile: CommProfile,
    pub dev_resources: CommDevResources,

    pub task_queue: TaskQueue,

    pub plan_schedule: HashMap<u32, ChanWorkSchedule>,
    pub unlaunched_plans: VecDeque<KernelPlan>,

    pub stream: cudaStream_t,
    pub event: cudaEvent_t,
}

// TBD
unsafe impl Send for Communicator {}

pub struct ChannelCommPattern {
    // channel id
    pub channel: u32,
    pub ring: RingPattern,
}
