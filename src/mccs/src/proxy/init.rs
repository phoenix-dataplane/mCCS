use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::Write;
use std::hash::Hash;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cuda_runtime_sys::{cudaError, cudaEventCreateWithFlags, cudaStreamCreate};
use cuda_runtime_sys::{cudaEventDisableTiming, cudaEventInterprocess};
use cuda_runtime_sys::{cudaEvent_t, cudaStream_t};
use ipc::mccs::reconfig::CommPatternReconfig;

use crate::bootstrap::{BootstrapHandle, BootstrapState};
use crate::comm::device::CommDevResources;
use crate::comm::{
    ChannelCommPattern, CommProfile, Communicator, CommunicatorId, PeerInfo, MCCS_MAX_CHANNELS,
    MCCS_WORK_FIFO_DEPTH,
};
use crate::cuda::alloc::DeviceHostMapped;
use crate::cuda_warning;
use crate::daemon::DaemonId;
use crate::transport::catalog::TransportCatalog;
use crate::transport::channel::{
    ChannelId, ChannelPeerConn, CommChannel, ConnType, PeerConnId, PeerConnector,
};
use crate::transport::engine::TransportEngineId;
use crate::transport::setup::TransportConnectState;
use crate::transport::transporter::TransportAgentId;

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
    pub cuda_dev: i32,
    pub daemon: DaemonId,

    pub stage: CommInitStage,

    pub rank: usize,
    pub num_ranks: usize,
    pub profile: CommProfile,

    pub peers_info: Option<Vec<PeerInfo>>,

    pub comm_patterns: Option<Vec<ChannelCommPattern>>,

    pub bootstrap_handle: Option<BootstrapHandle>,
    pub bootstrap_state: Option<Arc<BootstrapState>>,

    pub transport_connect: Option<TransportConnectState>,
    pub transport_engine_assignment: HashMap<PeerConnId, TransportEngineId>,

    pub stream: Option<cudaStream_t>,
    pub event: Option<cudaEvent_t>,
}

// TBD
unsafe impl Send for CommInitState {}

impl CommInitState {
    pub fn new(
        id: CommunicatorId,
        cuda_dev: i32,
        daemon_id: DaemonId,
        rank: usize,
        num_ranks: usize,
        comm_profile: CommProfile,
    ) -> Self {
        CommInitState {
            id,
            cuda_dev,
            daemon: daemon_id,
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
            stream: None,
            event: None,
        }
    }
}

fn new_chan_peer_conn() -> ChannelPeerConn {
    ChannelPeerConn {
        send: std::array::from_fn(|_| None),
        recv: std::array::from_fn(|_| None),
    }
}

impl CommInitState {
    pub fn finalize_communicator(mut self) -> Communicator {
        let mut channels = BTreeMap::new();
        for chan_pattern in self.comm_patterns.unwrap().iter() {
            let channel = CommChannel {
                peers: HashMap::new(),
                ring: chan_pattern.ring.clone(),
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
            peer_conns[peer_conn.conn_index as usize] = Some(peer_connector);
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
        let event = if let Some(event) = self.event {
            event
        } else {
            unsafe {
                let mut event = std::ptr::null_mut();
                cuda_warning!(cudaEventCreateWithFlags(
                    &mut event,
                    cudaEventInterprocess | cudaEventDisableTiming
                ));
                event
            }
        };
        let stream = if let Some(stream) = self.stream {
            stream
        } else {
            unsafe {
                let mut stream = std::ptr::null_mut();
                cuda_warning!(cudaStreamCreate(&mut stream));
                stream
            }
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

        let bootstrap_state = self.bootstrap_state.unwrap();
        Communicator {
            id: self.id,
            cuda_dev: self.cuda_dev,
            daemon: self.daemon,
            rank: self.rank,
            num_ranks: self.num_ranks,
            peers_info,
            channels,
            profile: self.profile,
            dev_resources,
            bootstrap_state,
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

pub enum CommReconfigTask {
    CommPattern(Vec<ChannelCommPattern>, CommPatternReconfig),
}

pub enum CommReconfigOutout {
    CommPattern(CommInitState),
}

pub struct CommSuspendState {
    pub stream_completed: bool,
    pub agents_pending_shutdown: HashSet<TransportAgentId>,
    pub reconfig_task: CommReconfigTask,
}

impl CommSuspendState {
    pub fn init(comm: &Communicator, task: CommReconfigTask) -> Self {
        let stream_completed = unsafe {
            match cuda_runtime_sys::cudaStreamQuery(comm.stream) {
                cudaError::cudaSuccess => true,
                cudaError::cudaErrorNotReady => false,
                e => {
                    log::error!(
                        "CUDA runtime failed with {:?} at {}:{}.",
                        e,
                        file!(),
                        line!()
                    );
                    panic!("CUDA runtime error");
                }
            }
        };
        let mut agents_pending_shutdown = HashSet::new();
        for (channel_id, chan) in comm.channels.iter() {
            for (peer_rank, peers) in chan.peers.iter() {
                for (conn_index, peer) in peers.send.iter().enumerate() {
                    if let Some(peer) = peer {
                        if let Some(transport_engine) = peer.transport_agent_engine {
                            let peer_conn = PeerConnId {
                                peer_rank: *peer_rank,
                                channel: *channel_id,
                                conn_index: conn_index as u32,
                                conn_type: ConnType::Send,
                            };
                            let agent_id = TransportAgentId {
                                communicator_id: comm.id,
                                client_rank: comm.rank,
                                client_cuda_dev: comm.cuda_dev,
                                peer_conn,
                            };
                            agents_pending_shutdown.insert(agent_id);
                        }
                    }
                }
                for (conn_index, peer) in peers.recv.iter().enumerate() {
                    if let Some(peer) = peer {
                        if let Some(transport_engine) = peer.transport_agent_engine {
                            let peer_conn = PeerConnId {
                                peer_rank: *peer_rank,
                                channel: *channel_id,
                                conn_index: conn_index as u32,
                                conn_type: ConnType::Recv,
                            };
                            let agent_id = TransportAgentId {
                                communicator_id: comm.id,
                                client_rank: comm.rank,
                                client_cuda_dev: comm.cuda_dev,
                                peer_conn,
                            };
                            agents_pending_shutdown.insert(agent_id);
                        }
                    }
                }
            }
        }
        let state = CommSuspendState {
            stream_completed,
            agents_pending_shutdown,
            reconfig_task: task,
        };
        state
    }
}

impl CommSuspendState {
    #[inline]
    pub fn check_suspended(&self) -> bool {
        self.stream_completed && self.agents_pending_shutdown.is_empty()
    }

    pub fn emit(
        self,
        mut comm: Communicator,
        transport_catalog: &TransportCatalog,
    ) -> CommReconfigOutout {
        match self.reconfig_task {
            CommReconfigTask::CommPattern(channels, comm_pattern) => {
                let mut udp_sport_map = HashMap::new();
                let mut channel_net_dev_map = HashMap::new();
                for channel in comm_pattern.channels.iter() {
                    if let Some(sport_map) = channel.udp_sport.as_ref() {
                        for spec in sport_map.iter() {
                            let send_rank = spec.0;
                            let recv_rank = spec.1;
                            if send_rank != comm.rank {
                                continue;
                            }
                            let conn_id = PeerConnId {
                                peer_rank: recv_rank,
                                channel: ChannelId(channel.channel_id),
                                conn_index: 0,
                                conn_type: ConnType::Send,
                            };
                            udp_sport_map.insert(conn_id, spec.2);
                        }
                    }
                    if let Some(net_dev) = channel.net_dev.as_ref() {
                        channel_net_dev_map.insert(ChannelId(channel.channel_id), net_dev.clone());
                    }
                }
                let tc = comm_pattern.ib_traffic_class;
                let profile = CommProfile {
                    buff_sizes: comm.profile.buff_sizes,
                    udp_sport_map,
                    channel_net_device_map: channel_net_dev_map,
                    tc,
                };

                let mut init = CommInitState::new(
                    comm.id,
                    comm.cuda_dev,
                    comm.daemon,
                    comm.rank,
                    comm.num_ranks,
                    profile,
                );
                let peers_info = comm.peers_info;

                let mut transport_connect =
                    TransportConnectState::new(comm.rank, comm.num_ranks, channels.len());
                for pattern in channels.iter() {
                    let ix_zero = pattern
                        .ring
                        .user_ranks
                        .iter()
                        .position(|x| *x == 0)
                        .unwrap();
                    let mut ring_log = String::new();
                    for i in 0..comm.num_ranks {
                        let ring_rank = pattern.ring.user_ranks[(i + ix_zero) % comm.num_ranks];
                        write!(ring_log, "{} -> ", ring_rank);
                    }
                    ring_log.push('0');
                    log::info!("Comm {:?}, ring: {}", comm.id, ring_log);

                    let ring_next = PeerConnId {
                        peer_rank: pattern.ring.next,
                        channel: pattern.channel,
                        conn_index: 0,
                        conn_type: ConnType::Send,
                    };
                    let ring_prev = PeerConnId {
                        peer_rank: pattern.ring.prev,
                        channel: pattern.channel,
                        conn_index: 0,
                        conn_type: ConnType::Recv,
                    };
                    transport_connect.register_connect(&ring_next).unwrap();
                    transport_connect.register_connect(&ring_prev).unwrap();
                }
                transport_connect
                    .post_setup_tasks(peers_info.as_slice(), &comm.profile, &transport_catalog)
                    .unwrap();
                init.comm_patterns = Some(channels);
                init.peers_info = Some(peers_info);
                init.transport_connect = Some(transport_connect);
                init.stage = CommInitStage::ConnectRing;
                init.bootstrap_state = Some(comm.bootstrap_state);
                init.stream = Some(comm.stream);
                init.event = Some(comm.event);

                comm.stream = std::ptr::null_mut();
                comm.event = std::ptr::null_mut();

                CommReconfigOutout::CommPattern(init)
            }
        }
    }
}
