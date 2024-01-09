use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::Hash;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cuda_runtime_sys::{cudaEventCreate, cudaStreamCreate};

use crate::bootstrap::{BootstrapHandle, BootstrapState};
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
use crate::transport::setup::TransportConnectState;

use super::plan::ChanWorkSchedule;
use super::task::TaskQueue;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommInitStage {
    BootstrapInit,
    AllGatherPeerInfo,
    ConnectRing,
    Finished,
}

pub struct CommInitState {
    pub id: CommunicatorId,
    pub stage: CommInitStage,

    pub rank: usize,
    pub num_ranks: usize,
    pub profile: CommProfile,

    pub peers_info: Option<Vec<PeerInfo>>,

    pub comm_patterns: Option<BTreeMap<ChannelId, ChannelCommPattern>>,

    pub bootstrap_handle: Option<BootstrapHandle>,
    pub bootstrap_state: Option<Arc<BootstrapState>>,

    pub transport_connect: Option<TransportConnectState>,
    pub transport_engine_assignment: HashMap<PeerConnId, TransportEngineId>,
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
            stage: CommInitStage::BootstrapInit,
            rank,
            num_ranks,
            profile: comm_profile,
            peers_info: None,
            comm_patterns: None,
            bootstrap_handle: None,
            bootstrap_state: None,
            transport_connect: None,
            transport_engine_assignment: HashMap::new(),
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
    pub fn finalize_communicator(mut self) -> Communicator {
        let mut channels = HashMap::new();
        for chan_pattern in self.comm_patterns.unwrap() {
            let channel = CommChannel {
                peers: HashMap::new(),
                ring: chan_pattern.ring,
                work_queue_next_available: 0,
            };
            channels.insert(chan_pattern.channel, channel);
        }
        for (peer_conn, peer_connected) in self.transport_connect.unwrap().peer_connected {
            let channel = channels.get_mut(&peer_conn.channel).unwrap();
            let peer = channel
                .peers
                .entry(peer_conn.peer_rank)
                .or_insert_with(new_chan_peer_conn);
            let peer_conns = match peer_conn.conn_type {
                ConnType::Send => &mut peer.send,
                ConnType::Recv => &mut peer.recv,
            };
            let transport_engine_idx = self.transport_engine_assignment.remove(&peer_conn);
            let peer_connector = PeerConnector {
                conn_info: peer_connected.conn_info,
                transport_agent_engine: transport_engine_idx,
                transporter: peer_connected.transporter,
                transport_resources: peer_connected.resources,
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
        for (peer_rank, peer_info) in self.peers_info.unwrap().into_iter().enumerate() {
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
