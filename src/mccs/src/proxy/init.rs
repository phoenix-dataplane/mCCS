use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::Hash;
use std::mem::MaybeUninit;

use cuda_runtime_sys::{cudaEventCreate, cudaStreamCreate};

use crate::comm::device::CommDevResources;
use crate::comm::{
    ChannelCommPattern, CommProfile, Communicator, CommunicatorId, PeerInfo, MCCS_MAX_CHANNELS,
    MCCS_WORK_FIFO_DEPTH,
};
use crate::cuda::alloc::DeviceHostMapped;
use crate::cuda_warning;
use crate::transport::channel::{
    ChannelId, ChannelPeerConn, CommChannel, ConnType, PeerConnId, PeerConnector,
};
use crate::transport::engine::TransportEngineId;
use crate::transport::transporter::{AgentMessage, AnyResources, ConnectHandle, Transporter};

use super::plan::ChanWorkSchedule;
use super::task::TaskQueue;

pub struct PeerConnConstruct {
    pub transporter: &'static dyn Transporter,
    pub resources: Option<AnyResources>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommInitStage {
    RegisterRank,
    PeerInfoExchange,
    ConnectChannel,
}

pub struct CommInitState {
    pub id: CommunicatorId,
    pub stage: CommInitStage,

    pub rank: usize,
    pub num_ranks: usize,
    pub profile: CommProfile,
    pub peers_info: HashMap<usize, PeerInfo>,
    pub peers_await_exchange: Vec<usize>,

    pub comm_patterns: BTreeMap<ChannelId, ChannelCommPattern>,

    pub to_setup: VecDeque<PeerConnId>,
    pub to_setup_agent_cb: VecDeque<(PeerConnId, AgentMessage)>,
    pub to_connect: VecDeque<(PeerConnId, ConnectHandle)>,
    pub to_connect_agent_cb: VecDeque<(PeerConnId, AgentMessage)>,
    pub peer_transport_assigned: HashMap<PeerConnId, TransportEngineId>,

    pub peer_setup_pre_agent: HashMap<PeerConnId, PeerConnConstruct>,
    pub peer_setup: HashMap<PeerConnId, PeerConnConstruct>,
    pub peer_connect_pre_agent: HashMap<PeerConnId, PeerConnConstruct>,

    pub peer_connected: HashMap<PeerConnId, PeerConnector>,
    pub await_connections: usize,
}

// TBD
unsafe impl Send for CommInitState {}

impl CommInitState {
    pub fn new(
        id: CommunicatorId,
        rank: usize,
        num_ranks: usize,
        comm_profile: CommProfile,
    ) -> Self {
        CommInitState {
            id,
            stage: CommInitStage::RegisterRank,
            rank,
            num_ranks,
            profile: comm_profile,
            peers_info: HashMap::new(),
            peers_await_exchange: Vec::new(),
            comm_patterns: Default::default(),
            to_setup: VecDeque::new(),
            to_setup_agent_cb: VecDeque::new(),
            to_connect: VecDeque::new(),
            to_connect_agent_cb: VecDeque::new(),
            peer_transport_assigned: HashMap::new(),
            peer_setup_pre_agent: HashMap::new(),
            peer_setup: HashMap::new(),
            peer_connect_pre_agent: HashMap::new(),
            peer_connected: HashMap::new(),
            await_connections: 0,
        }
    }
}

fn new_chan_peer_conn() -> ChannelPeerConn {
    ChannelPeerConn {
        send: HashMap::new(),
        recv: HashMap::new(),
    }
}

impl CommInitState {
    pub fn enqueue_channels_setup(&mut self) {
        for pattern in self.comm_patterns.values() {
            let ring_send = PeerConnId {
                peer_rank: pattern.ring.next,
                channel: pattern.channel,
                conn_index: 0,
                conn_type: ConnType::Send,
            };
            let ring_recv = PeerConnId {
                peer_rank: pattern.ring.prev,
                channel: pattern.channel,
                conn_index: 0,
                conn_type: ConnType::Recv,
            };
            self.to_setup.push_back(ring_send);
            self.to_setup.push_back(ring_recv);
        }
    }

    pub fn finalize_communicator(self) -> Communicator {
        let mut channels = BTreeMap::new();
        for chan_pattern in self.comm_patterns.into_values() {
            let channel = CommChannel {
                peers: HashMap::new(),
                ring: chan_pattern.ring,
                work_queue_next_available: 0,
            };
            channels.insert(chan_pattern.channel, channel);
        }
        for (peer_conn, peer_connector) in self.peer_connected {
            let channel = channels.get_mut(&peer_conn.channel).unwrap();
            let peer = channel
                .peers
                .entry(peer_conn.peer_rank)
                .or_insert_with(new_chan_peer_conn);
            let peer_conns = match peer_conn.conn_type {
                ConnType::Send => &mut peer.send,
                ConnType::Recv => &mut peer.recv,
            };
            peer_conns.insert(peer_conn.conn_index, peer_connector);
        }

        let dev_resources = CommDevResources::new(
            self.rank,
            self.num_ranks,
            MCCS_WORK_FIFO_DEPTH,
            &self.profile,
            &channels,
        );

        let mut plan_schedule = BTreeMap::new();
        for chan in channels.keys() {
            let schedule = ChanWorkSchedule {
                coll_bytes: 0,
                work_queue: Vec::new(),
                agent_task_queue: Vec::new(),
            };
            plan_schedule.insert(*chan, schedule);
        }
        let task_queue = TaskQueue {
            coll_queue: VecDeque::new(),
        };
        let event = unsafe {
            let mut event = std::ptr::null_mut();
            cuda_warning!(cudaEventCreate(&mut event));
            event
        };
        let stream = unsafe {
            let mut stream = std::ptr::null_mut();
            cuda_warning!(cudaStreamCreate(&mut stream));
            stream
        };

        let mut peers_info = Vec::new();
        for _ in 0..self.num_ranks {
            peers_info.push(MaybeUninit::uninit());
        }
        for (peer_rank, peer_info) in self.peers_info {
            peers_info[peer_rank].write(peer_info);
        }

        let peers_info = peers_info
            .into_iter()
            .map(|x| unsafe { x.assume_init() })
            .collect();
        Communicator {
            id: self.id,
            rank: self.rank,
            num_ranks: self.num_ranks,
            peers_info,
            channels,
            profile: self.profile,
            dev_resources,
            work_queue_acked_min: 0,
            work_queue_next_available: 0,
            task_queue,
            plan_schedule,
            unlaunched_plans: VecDeque::new(),
            stream,
            event,
        }
    }
}
