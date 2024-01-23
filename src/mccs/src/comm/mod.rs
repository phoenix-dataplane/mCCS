pub mod device;
pub mod profile;

pub use profile::CommProfile;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use bytes::{Buf, BufMut};
use serde::{Deserialize, Serialize};

use collectives_sys::mccsDevWork;
use cuda_runtime_sys::{cudaEvent_t, cudaStream_t};

use crate::bootstrap::BootstrapState;
use crate::cuda::alloc::DeviceHostMapped;
use crate::cuda_warning;
use crate::daemon::DaemonId;
use crate::pattern::RingPattern;
use crate::proxy::plan::{ChanWorkSchedule, KernelPlan};
use crate::proxy::task::TaskQueue;
use crate::transport::channel::{ChannelId, CommChannel};
use crate::transport::NUM_PROTOCOLS;
use crate::utils::tcp;
use device::CommDevResources;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
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
    pub host: IpAddr,
    pub cuda_device_idx: i32,
}

pub const MCCS_WORK_FIFO_DEPTH: usize = 64 << 10;
pub const MCCS_MAX_CHANNELS: usize = 32;

pub const PEER_INFO_EXCHANGE_SEND_SIZE: usize = 40;

pub struct PeerInfoExchange {
    pub rank: usize,
    pub host: IpAddr,
    pub cuda_device_idx: i32,
}

impl PeerInfoExchange {
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u64(self.rank as u64);
        buf.put_i32(self.cuda_device_idx);
        let sock_addr = SocketAddr::new(self.host.clone(), 0);
        tcp::encode_socket_addr(&sock_addr, buf);
    }

    pub fn decode<B: Buf>(buf: &mut B) -> Self {
        let rank = buf.get_u64() as usize;
        let cuda_idx = buf.get_i32();
        let sock_addr = tcp::decode_socket_addr(buf);
        let host = sock_addr.ip();
        Self {
            rank,
            host,
            cuda_device_idx: cuda_idx,
        }
    }
}

pub struct Communicator {
    pub id: CommunicatorId,
    pub cuda_dev: i32,
    pub daemon: DaemonId,

    pub rank: usize,
    pub num_ranks: usize,

    pub peers_info: Vec<PeerInfo>,

    // channel_id -> CommChannel
    pub channels: BTreeMap<ChannelId, CommChannel>,

    pub profile: CommProfile,
    pub dev_resources: CommDevResources,

    pub bootstrap_state: Arc<BootstrapState>,

    pub work_queue_acked_min: u32,
    pub work_queue_next_available: u32,

    // enqueue system intermediate objects
    pub task_queue: TaskQueue,
    pub plan_schedule: BTreeMap<ChannelId, ChanWorkSchedule>,
    pub unlaunched_plans: VecDeque<KernelPlan>,

    // backend stream and backend event
    pub stream: cudaStream_t,
    pub event: cudaEvent_t,
}

pub(crate) struct EnqueueState {}

// Communicator will only be held by a single proxy engine
// on a single thread (runtime)
unsafe impl Send for Communicator {}

#[derive(Debug, Clone)]
pub struct ChannelCommPattern {
    // channel id
    pub channel: ChannelId,
    pub ring: RingPattern,
}
