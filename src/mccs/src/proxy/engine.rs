use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::Write;
use std::net::SocketAddr;
use std::sync::Arc;
use std::task::{Context, Poll};

use crossbeam::channel::{Receiver, Sender, TryRecvError};
use futures::future::BoxFuture;
use futures::FutureExt;

use cuda_runtime_sys::{
    cudaError, cudaEventCreateWithFlags, cudaEventDisableTiming, cudaEventInterprocess,
    cudaEventRecord, cudaEvent_t, cudaIpcEventHandle_t, cudaIpcGetEventHandle,
    cudaIpcOpenEventHandle, cudaStreamQuery, cudaStreamWaitEvent,
};

use super::command::{self, ProxyCommand, ProxyCompletion};
use super::init::{CommInitStage, CommInitState, CommReconfigTask, CommSuspendState};
use super::op::ProxyOp;
use super::task::{CollTask, TaskDataType, TaskFuncType, TaskReduceOp};
use super::DeviceInfo;
use crate::bootstrap::BootstrapState;
use crate::bootstrap::{bootstrap_create_root, bootstrap_root};
use crate::comm::PEER_INFO_EXCHANGE_SEND_SIZE;
use crate::comm::{
    CommProfile, Communicator, CommunicatorId, PeerInfo, PeerInfoExchange, PeerType,
};
use crate::cuda::ptr::DeviceNonNull;
use crate::cuda_warning;
use crate::daemon::DaemonId;
use crate::engine::{Engine, EngineStatus};
use crate::exchange::command::{ExchangeCommand, ExchangeNotification};
use crate::message::{ControlNotification, ControlRequest};
use crate::pattern::{
    ALLGATHER_CHUNK_STEPS, ALLGATHER_SLICE_STEPS, ALLREDUCE_CHUNK_STEPS, ALLREDUCE_SLICE_STEPS,
};
use crate::proxy::init::CommReconfigOutout;
use crate::registry::GlobalRegistry;
use crate::transport::channel::{ChannelId, ConnType, PeerConnId};
use crate::transport::engine::TransportEngineId;
use crate::transport::message::{TransportEngineReply, TransportEngineRequest};
use crate::transport::setup::exchange_connect_handle;
use crate::transport::setup::{TransportConnectState, TransportConnectTask};
use crate::transport::transporter::{
    ConnectHandle, TransportAgentId, TransportConnect, TransportSetup,
};
use crate::utils::duplex_chan::DuplexChannel;
use crate::utils::pool::WorkPool;

pub struct ProxyResources {
    pub device_info: DeviceInfo,
    // control engine
    pub control_chan: DuplexChannel<ControlRequest, ControlNotification>,
    // daemons
    pub daemon_tx: HashMap<DaemonId, Sender<ProxyCompletion>>,
    pub daemon_rx: Vec<(DaemonId, Receiver<ProxyCommand>)>,
    // exchange engine
    pub exchange_chan: DuplexChannel<ExchangeCommand, ExchangeNotification>,
    // communications and transport
    pub comms_init: HashMap<CommunicatorId, CommInitState>,
    pub comms_suspended: HashMap<CommunicatorId, (Communicator, CommSuspendState)>,
    // for each daemon, user stream handle -> user event
    pub user_events: HashMap<DaemonId, HashMap<usize, cudaEvent_t>>,
    // established communicators
    pub communicators: HashMap<CommunicatorId, Communicator>,
    pub global_registry: GlobalRegistry,
    pub transport_engines_tx: HashMap<TransportEngineId, Sender<TransportEngineRequest>>,
    pub transport_engines_rx: Vec<(TransportEngineId, Receiver<TransportEngineReply>)>,
    pub transport_submission_cache: HashMap<TransportEngineId, VecDeque<TransportEngineRequest>>,
    pub task_submit_pool: Vec<TaskSubmission>,
    pub daemon_shutdown: Vec<(DaemonId, usize)>,
    pub transport_shutdown: Vec<(TransportEngineId, usize)>,
    // TODO: FIXME
    pub queued_commands: Vec<(DaemonId, super::command::AllReduceRequest)>,
}

unsafe impl Send for ProxyResources {}

enum AsyncTaskOutput {
    BootstrapRoot,
    BootstrapInit(BootstrapState),
    HandleExchange(HashMap<PeerConnId, ConnectHandle>),
    AllGatherPeerInfo(Vec<PeerInfoExchange>),
}

pub struct AsyncTask {
    comm_id: CommunicatorId,
    fut: BoxFuture<'static, Result<AsyncTaskOutput, anyhow::Error>>,
}

pub enum TaskSubmission {
    AsyncTask(AsyncTask),
    ProxyOp(ProxyOp),
}

impl ProxyResources {
    fn progress_async_task(&mut self, task: &mut AsyncTask) -> bool {
        let waker = futures::task::noop_waker_ref();
        let mut cx = Context::from_waker(waker);
        let poll = task.fut.as_mut().poll(&mut cx);
        match poll {
            Poll::Ready(result) => {
                let output = result.unwrap();
                let comm = self.comms_init.get_mut(&task.comm_id).unwrap();
                match output {
                    AsyncTaskOutput::BootstrapInit(state) => {
                        comm.bootstrap_state = Some(Arc::new(state));
                    }
                    AsyncTaskOutput::HandleExchange(handles) => {
                        let transport_connect = comm.transport_connect.as_mut().unwrap();
                        transport_connect.put_peer_connect_handles(handles);
                    }
                    AsyncTaskOutput::AllGatherPeerInfo(info) => {
                        assert_eq!(info.len(), comm.num_ranks);
                        let mut peers_info = info
                            .into_iter()
                            .map(|x| {
                                let peer_type = if x.rank == comm.rank {
                                    PeerType::Local
                                } else if x.host == self.device_info.host {
                                    PeerType::IntraNode
                                } else {
                                    PeerType::InterNode
                                };
                                PeerInfo {
                                    rank: x.rank,
                                    local_rank: 0,
                                    peer_type,
                                    host: x.host,
                                    cuda_device_idx: x.cuda_device_idx,
                                }
                            })
                            .collect::<Vec<_>>();
                        let mut host_dev_count = HashMap::new();
                        for info in peers_info.iter_mut() {
                            let count = host_dev_count.entry(info.host.clone()).or_insert(0);
                            info.local_rank = *count;
                            *count += 1;
                        }
                        comm.peers_info = Some(peers_info);
                    }
                    AsyncTaskOutput::BootstrapRoot => {}
                }
                true
            }
            Poll::Pending => false,
        }
    }
}

impl ProxyResources {
    fn register_daemon_engine(
        &mut self,
        id: DaemonId,
        chan: DuplexChannel<ProxyCompletion, ProxyCommand>,
    ) {
        self.user_events.insert(id, HashMap::new());
        self.daemon_tx.insert(id, chan.tx);
        self.daemon_rx.push((id, chan.rx));
    }

    fn register_transport_engine(
        &mut self,
        id: TransportEngineId,
        chan: DuplexChannel<TransportEngineRequest, TransportEngineReply>,
    ) {
        let pool = self.transport_submission_cache.remove(&id);
        if let Some(requests) = pool {
            for req in requests {
                chan.tx.send(req).unwrap();
            }
        }
        self.transport_engines_tx.insert(id, chan.tx);
        let rx_idx = self
            .transport_engines_rx
            .iter()
            .position(|(rx_id, _)| id == *rx_id);
        if let Some(idx) = rx_idx {
            self.transport_engines_rx.swap_remove(idx);
        }
        self.transport_engines_rx.push((id, chan.rx));
    }
}

impl ProxyResources {
    fn send_transport_request(
        request: TransportEngineRequest,
        transport_engine: TransportEngineId,
        transport_tx: &mut HashMap<TransportEngineId, Sender<TransportEngineRequest>>,
        submission_pool: &mut HashMap<TransportEngineId, VecDeque<TransportEngineRequest>>,
    ) {
        use crossbeam::channel::SendError;

        let sender = transport_tx.get_mut(&transport_engine);
        submission_pool
            .entry(transport_engine)
            .or_insert_with(VecDeque::new);
        if let Some(sender) = sender {
            match sender.send(request) {
                Ok(()) => (),
                Err(SendError(request)) => {
                    // disconnected
                    Self::enqueue_submission_cache(submission_pool, transport_engine, request);
                }
            }
        } else {
            Self::enqueue_submission_cache(submission_pool, transport_engine, request);
        }
    }

    fn enqueue_submission_cache(
        pool: &mut HashMap<TransportEngineId, VecDeque<TransportEngineRequest>>,
        transport_engine: TransportEngineId,
        request: TransportEngineRequest,
    ) {
        let queue = pool.entry(transport_engine).or_insert_with(VecDeque::new);
        queue.push_back(request);
    }

    fn init_communicator(&mut self, comm_id: CommunicatorId) -> bool {
        let comm = self.comms_init.get_mut(&comm_id).unwrap();
        if comm.stage == CommInitStage::BootstrapInit {
            if let Some(handle) = comm.bootstrap_handle.take() {
                let mut listen_addr = SocketAddr::new(self.device_info.host.clone(), 0);
                let fut = BootstrapState::init(handle, listen_addr, comm.rank, comm.num_ranks).map(
                    |state| {
                        state
                            .map(|s| AsyncTaskOutput::BootstrapInit(s))
                            .map_err(|e| anyhow::Error::new(e))
                    },
                );
                let task = AsyncTask {
                    comm_id,
                    fut: Box::pin(fut),
                };
                self.task_submit_pool.push(TaskSubmission::AsyncTask(task));
            } else if let Some(state) = comm.bootstrap_state.as_ref() {
                let peer_info_exchange = PeerInfoExchange {
                    rank: comm.rank,
                    host: self.device_info.host.clone(),
                    cuda_device_idx: self.device_info.cuda_device_idx,
                };
                let mut slice = vec![0u8; PEER_INFO_EXCHANGE_SEND_SIZE];
                let mut buf = slice.as_mut_slice();
                peer_info_exchange.encode(&mut buf);
                let num_ranks = comm.num_ranks;
                let fut = Arc::clone(state).bootstrap_all_gather(slice).map(move |x| {
                    x.map(|all_data| {
                        let mut peers_info = Vec::with_capacity(num_ranks);
                        for r in 0..num_ranks {
                            let mut buf = &all_data[r * PEER_INFO_EXCHANGE_SEND_SIZE
                                ..(r + 1) * PEER_INFO_EXCHANGE_SEND_SIZE];
                            let peer_info = PeerInfoExchange::decode(&mut buf);
                            peers_info.push(peer_info);
                        }
                        AsyncTaskOutput::AllGatherPeerInfo(peers_info)
                    })
                    .map_err(|e| anyhow::Error::new(e))
                });
                let task = AsyncTask {
                    comm_id,
                    fut: Box::pin(fut),
                };
                self.task_submit_pool.push(TaskSubmission::AsyncTask(task));
                comm.stage = CommInitStage::AllGatherPeerInfo;
            }
        } else if comm.stage == CommInitStage::AllGatherPeerInfo {
            if comm.peers_info.is_some() {
                let pattern_override = self.global_registry.comm_pattern_override.get(&comm_id);
                let channels = if let Some(pattern) = pattern_override {
                    let mut channels = Vec::with_capacity(pattern.channels.len());
                    for chan in pattern.channels.iter() {
                        assert_eq!(chan.ring.len(), comm.num_ranks);
                        let ix_rank = chan.ring.iter().position(|x| *x == comm.rank).unwrap();
                        let ix_zero = chan.ring.iter().position(|x| *x == 0).unwrap();
                        let mut user_ranks = Vec::with_capacity(comm.num_ranks);
                        for i in 0..comm.num_ranks {
                            let ring_rank = chan.ring[(i + ix_rank) % comm.num_ranks];
                            assert!(ring_rank < comm.num_ranks);
                            user_ranks.push(ring_rank);
                        }
                        let ring = crate::pattern::RingPattern {
                            prev: user_ranks[comm.num_ranks - 1],
                            next: user_ranks[1],
                            user_ranks,
                            index: (ix_rank + comm.num_ranks - ix_zero) % comm.num_ranks,
                        };
                        let chan_pattern = crate::comm::ChannelCommPattern {
                            channel: ChannelId(chan.channel_id),
                            ring,
                        };
                        channels.push(chan_pattern);
                    }
                    channels
                } else {
                    // 0 -> 1 -> ... -> numRanks-1 -> 0
                    let ring_next = (comm.rank + 1) % comm.num_ranks;
                    let ring_prev = (comm.rank + comm.num_ranks - 1) % comm.num_ranks;
                    // in current implentation, ring follows the same ordering of ranks
                    let ring_index = comm.rank;

                    let mut user_ranks = Vec::with_capacity(comm.num_ranks);
                    for idx in 0..comm.num_ranks {
                        let ring_rank = (comm.rank + idx) % comm.num_ranks;
                        user_ranks.push(ring_rank);
                    }
                    let ring_pattern = crate::pattern::RingPattern {
                        prev: ring_prev,
                        next: ring_next,
                        user_ranks,
                        index: ring_index,
                    };

                    let channels = (0..self.global_registry.default_comm_config.channel_count)
                        .map(|idx| crate::comm::ChannelCommPattern {
                            channel: ChannelId(idx),
                            ring: ring_pattern.clone(),
                        })
                        .collect::<Vec<_>>();
                    channels
                };

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
                comm.comm_patterns = Some(channels);
                let peers_info = comm.peers_info.as_ref().unwrap();
                transport_connect
                    .post_setup_tasks(
                        peers_info.as_slice(),
                        &comm.profile,
                        &self.global_registry.transport_catalog,
                    )
                    .unwrap();
                comm.transport_connect = Some(transport_connect);
                comm.stage = CommInitStage::ConnectRing;
            }
        } else if comm.stage == CommInitStage::ConnectRing {
            let transport_connect = comm.transport_connect.as_mut().unwrap();
            let task = transport_connect.get_task().unwrap();
            match task {
                TransportConnectTask::Idle => {
                    comm.stage = CommInitStage::Finished;
                }
                TransportConnectTask::WaitingOutstandingTask => {}
                TransportConnectTask::PeerTransportSetup(setup) => {
                    let peer_conn = &setup.id;
                    let transporter = setup.transporter;
                    let peers_info = comm.peers_info.as_ref().unwrap();
                    let setup_result = match peer_conn.conn_type {
                        ConnType::Send => transporter
                            .send_setup(
                                peer_conn,
                                &peers_info[comm.rank],
                                &peers_info[peer_conn.peer_rank],
                                &comm.profile,
                                &self.global_registry.transport_catalog,
                            )
                            .unwrap(),
                        ConnType::Recv => transporter
                            .recv_setup(
                                peer_conn,
                                &peers_info[comm.rank],
                                &peers_info[peer_conn.peer_rank],
                                &comm.profile,
                                &self.global_registry.transport_catalog,
                            )
                            .unwrap(),
                    };
                    match setup_result {
                        TransportSetup::PreAgentCb {
                            agent_cuda_dev,
                            agent_request,
                            setup_resources,
                        } => {
                            let agent = TransportAgentId {
                                communicator_id: comm_id,
                                client_rank: comm.rank,
                                client_cuda_dev: self.device_info.cuda_device_idx,
                                peer_conn: *peer_conn,
                            };
                            let transport_engine_idx = self
                                .global_registry
                                .transport_delegator
                                .assign_transport_engine(
                                    agent_cuda_dev,
                                    agent,
                                    &mut self.control_chan.tx,
                                );
                            comm.transport_engine_assignment
                                .insert(*peer_conn, transport_engine_idx);

                            let request = match peer_conn.conn_type {
                                ConnType::Send => TransportEngineRequest::AgentSetup(
                                    transporter,
                                    agent,
                                    agent_request,
                                ),
                                ConnType::Recv => TransportEngineRequest::AgentSetup(
                                    transporter,
                                    agent,
                                    agent_request,
                                ),
                            };
                            Self::send_transport_request(
                                request,
                                transport_engine_idx,
                                &mut self.transport_engines_tx,
                                &mut self.transport_submission_cache,
                            );
                            transport_connect
                                .put_peer_setup_pre_agent(peer_conn, setup_resources)
                                .unwrap();
                        }
                        TransportSetup::Setup {
                            peer_connect_handle,
                            setup_resources,
                        } => {
                            transport_connect
                                .put_peer_setup(peer_conn, peer_connect_handle, setup_resources)
                                .unwrap();
                        }
                    }
                }
                TransportConnectTask::PeerTransportSetupAgentCb(setup_cb) => {
                    let peer_conn = &setup_cb.id;
                    let transporter = setup_cb.transporter;
                    let agent_reply = setup_cb.agent_message;
                    let setup_resources = setup_cb.setup_resources;
                    let setup_result = match peer_conn.conn_type {
                        ConnType::Send => transporter
                            .send_setup_agent_callback(
                                comm.rank,
                                &peer_conn,
                                agent_reply,
                                setup_resources,
                            )
                            .unwrap(),
                        ConnType::Recv => transporter
                            .recv_setup_agent_callback(
                                comm.rank,
                                &peer_conn,
                                agent_reply,
                                setup_resources,
                            )
                            .unwrap(),
                    };
                    match setup_result {
                        TransportSetup::PreAgentCb { .. } => {
                            panic!("PreAgentCb variant is not expected")
                        }
                        TransportSetup::Setup {
                            peer_connect_handle,
                            setup_resources,
                        } => {
                            transport_connect
                                .put_peer_setup(peer_conn, peer_connect_handle, setup_resources)
                                .unwrap();
                        }
                    }
                }
                TransportConnectTask::PeerTransportConnect(connect) => {
                    let peer_conn = &connect.id;
                    let transporter = connect.transporter;
                    let handle = connect.peer_connect_handle;
                    let setup_resources = connect.setup_resources;
                    let connect_result = match peer_conn.conn_type {
                        ConnType::Send => transporter
                            .send_connect(&peer_conn, handle, setup_resources)
                            .unwrap(),
                        ConnType::Recv => transporter
                            .recv_connect(&peer_conn, handle, setup_resources)
                            .unwrap(),
                    };
                    match connect_result {
                        TransportConnect::PreAgentCb {
                            agent_request,
                            transport_resources,
                        } => {
                            let agent = TransportAgentId {
                                communicator_id: comm_id,
                                client_rank: comm.rank,
                                client_cuda_dev: self.device_info.cuda_device_idx,
                                peer_conn: *peer_conn,
                            };
                            let transport_engine = *comm
                                .transport_engine_assignment
                                .entry(*peer_conn)
                                .or_insert_with(|| {
                                    self.global_registry
                                        .transport_delegator
                                        .assign_transport_engine(
                                            self.device_info.cuda_device_idx,
                                            agent,
                                            &mut self.control_chan.tx,
                                        )
                                });
                            let request = match peer_conn.conn_type {
                                ConnType::Send => TransportEngineRequest::AgentConnect(
                                    transporter,
                                    agent,
                                    agent_request,
                                ),
                                ConnType::Recv => TransportEngineRequest::AgentConnect(
                                    transporter,
                                    agent,
                                    agent_request,
                                ),
                            };
                            Self::send_transport_request(
                                request,
                                transport_engine,
                                &mut self.transport_engines_tx,
                                &mut self.transport_submission_cache,
                            );
                            transport_connect
                                .put_peer_connect_pre_agent(peer_conn, transport_resources)
                                .unwrap();
                        }
                        TransportConnect::Connect {
                            conn_info,
                            transport_resources,
                        } => {
                            transport_connect
                                .put_peer_connect(peer_conn, conn_info, transport_resources)
                                .unwrap();
                        }
                    }
                }
                TransportConnectTask::PeerTransportConnectAgentCb(connect_cb) => {
                    let peer_conn = &connect_cb.id;
                    let transporter = connect_cb.transporter;
                    let transport_resources = connect_cb.transport_resources;
                    let agent_reply = connect_cb.agent_message;
                    let connect_result = match peer_conn.conn_type {
                        ConnType::Send => transporter
                            .send_connect_agent_callback(
                                &peer_conn,
                                agent_reply,
                                transport_resources,
                            )
                            .unwrap(),
                        ConnType::Recv => transporter
                            .recv_connect_agent_callback(
                                &peer_conn,
                                agent_reply,
                                transport_resources,
                            )
                            .unwrap(),
                    };
                    match connect_result {
                        TransportConnect::PreAgentCb { .. } => {
                            panic!("PreAgentCb variant is not expected")
                        }
                        TransportConnect::Connect {
                            conn_info,
                            transport_resources,
                        } => {
                            transport_connect
                                .put_peer_connect(peer_conn, conn_info, transport_resources)
                                .unwrap();
                        }
                    }
                }
                TransportConnectTask::PeerTransportConnectHandleExchange(exchange_task) => {
                    let fut = exchange_connect_handle(
                        Arc::clone(comm.bootstrap_state.as_ref().unwrap()),
                        0,
                        exchange_task,
                    )
                    .map(|o| {
                        o.map(|handles| AsyncTaskOutput::HandleExchange(handles))
                            .map_err(|e| anyhow::Error::new(e))
                    });
                    let task = AsyncTask {
                        comm_id: comm.id,
                        fut: Box::pin(fut),
                    };
                    self.task_submit_pool.push(TaskSubmission::AsyncTask(task));
                }
            }
        }

        if comm.stage == CommInitStage::Finished {
            let comm_init = self.comms_init.remove(&comm_id).unwrap();
            let communicator = comm_init.finalize_communicator();
            self.communicators.insert(comm_id, communicator);
            true
        } else {
            false
        }
    }
}

impl ProxyResources {
    #[allow(unreachable_code, unused_variables)]
    fn process_op(&mut self, op: &mut ProxyOp) -> bool {
        match op {
            ProxyOp::InitCommunicator(daemon_id, comm_id) => {
                let res = self.init_communicator(*comm_id);
                if res {
                    let comm = self.communicators.get(comm_id).unwrap();
                    let mut handle = cudaIpcEventHandle_t::default();
                    unsafe {
                        cuda_warning!(cudaIpcGetEventHandle(&mut handle, comm.event));
                    }
                    self.daemon_tx
                        .get_mut(daemon_id)
                        .unwrap()
                        .send(ProxyCompletion::InitCommunicator(handle.into()))
                        .unwrap();
                }
                res
            }
            ProxyOp::RebootCommunicator(comm_id) => {
                let res = self.init_communicator(*comm_id);
                if res {
                    let comm = self.communicators.get_mut(comm_id).unwrap();
                    for (daemon_id, coll) in self.queued_commands.drain(..) {
                        let user_events = self.user_events.get(&daemon_id).unwrap();
                        comm.schedule_all_reduce(coll, user_events);
                        comm.launch_scheduled_and_record(&mut self.transport_engines_tx);
                        let sender = self.daemon_tx.get(&daemon_id).unwrap();
                        sender.send(ProxyCompletion::AllReduce).unwrap();
                    }
                }
                res
            }
            ProxyOp::PollCommunicatorComplete(comm_id) => {
                let (comm, state) = self.comms_suspended.get_mut(comm_id).unwrap();
                let stream_completed = unsafe {
                    match cudaStreamQuery(comm.stream) {
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
                if stream_completed {
                    state.stream_completed = true;
                    if state.check_suspended() {
                        let (comm, state) = self.comms_suspended.remove(&comm_id).unwrap();
                        let output = state.emit(comm, &self.global_registry.transport_catalog);
                        match output {
                            CommReconfigOutout::CommPattern(init) => {
                                let op = ProxyOp::RebootCommunicator(init.id);
                                self.comms_init.insert(init.id, init);
                                self.task_submit_pool.push(TaskSubmission::ProxyOp(op));
                            }
                        }
                    }
                }
                stream_completed
            }
        }
    }
}

pub struct ProxyEngine {
    pub resources: ProxyResources,
    pub ops: WorkPool<ProxyOp>,
    pub async_tasks: WorkPool<AsyncTask>,
}

impl ProxyEngine {
    pub fn new(
        device_info: DeviceInfo,
        global_registry: GlobalRegistry,
        control_chan: DuplexChannel<ControlRequest, ControlNotification>,
        exchange_chan: DuplexChannel<ExchangeCommand, ExchangeNotification>,
    ) -> Self {
        let resources = ProxyResources {
            device_info,
            control_chan,
            daemon_tx: HashMap::new(),
            daemon_rx: Vec::new(),
            exchange_chan,
            comms_init: HashMap::new(),
            comms_suspended: HashMap::new(),
            user_events: HashMap::new(),
            communicators: HashMap::new(),
            global_registry,
            transport_engines_tx: HashMap::new(),
            transport_engines_rx: Vec::new(),
            transport_submission_cache: HashMap::new(),
            task_submit_pool: Vec::new(),
            daemon_shutdown: Vec::new(),
            transport_shutdown: Vec::new(),
            queued_commands: Vec::new(),
        };
        let engine = ProxyEngine {
            resources,
            ops: WorkPool::new(),
            async_tasks: WorkPool::new(),
        };
        engine
    }
}

impl Engine for ProxyEngine {
    fn progress(&mut self) -> EngineStatus {
        self.check_daemon_command();

        if fastrand::usize(..10) < 1 {
            self.progress_ops();
            self.check_transport_reply();
            self.check_control_notify();
            self.check_exchange_reply();
            self.progress_async_tasks();
            self.enqueue_async_task();
        }

        EngineStatus::Progressed
    }
}

impl ProxyEngine {
    #[inline]
    fn enqueue_async_task(&mut self) {
        for task in self.resources.task_submit_pool.drain(..) {
            match task {
                TaskSubmission::AsyncTask(task) => {
                    self.async_tasks.enqueue(task);
                }
                TaskSubmission::ProxyOp(op) => {
                    self.ops.enqueue(op);
                }
            }
        }
    }

    #[inline]
    fn progress_ops(&mut self) {
        self.ops.progress(|op| self.resources.process_op(op))
    }

    #[inline]
    fn progress_async_tasks(&mut self) {
        self.async_tasks
            .progress(|x| self.resources.progress_async_task(x));
    }

    fn check_exchange_reply(&mut self) {
        match self.resources.exchange_chan.rx.try_recv() {
            Ok(msg) => match msg {
                ExchangeNotification::RegisterBootstrapHandle => {}
                ExchangeNotification::RecvBootstrapHandle(comm_id, handle) => {
                    let comm = self.resources.comms_init.get_mut(&comm_id).unwrap();
                    comm.bootstrap_handle = Some(handle);
                }
                ExchangeNotification::CommPatternReconfig(comm_pattern) => {
                    let comm_id = CommunicatorId(comm_pattern.communicator_id.0);
                    let comm = self.resources.communicators.remove(&comm_id);
                    if let Some(comm) = comm {
                        let mut channels = Vec::with_capacity(comm_pattern.channels.len());
                        for chan in comm_pattern.channels.iter() {
                            assert_eq!(chan.ring.len(), comm.num_ranks);
                            let ix_rank = chan.ring.iter().position(|x| *x == comm.rank).unwrap();
                            let ix_zero = chan.ring.iter().position(|x| *x == 0).unwrap();
                            let mut user_ranks = Vec::with_capacity(comm.num_ranks);
                            for i in 0..comm.num_ranks {
                                let ring_rank = chan.ring[(i + ix_rank) % comm.num_ranks];
                                assert!(ring_rank < comm.num_ranks);
                                user_ranks.push(ring_rank);
                            }
                            let ring = crate::pattern::RingPattern {
                                prev: user_ranks[comm.num_ranks - 1],
                                next: user_ranks[1],
                                user_ranks,
                                index: (ix_rank + comm.num_ranks - ix_zero) % comm.num_ranks,
                            };
                            let chan_pattern = crate::comm::ChannelCommPattern {
                                channel: ChannelId(chan.channel_id),
                                ring,
                            };
                            channels.push(chan_pattern);
                        }
                        let task = CommReconfigTask::CommPattern(channels, comm_pattern);
                        let state = CommSuspendState::init(&comm, task);
                        if state.check_suspended() {
                            let output =
                                state.emit(comm, &self.resources.global_registry.transport_catalog);
                            match output {
                                CommReconfigOutout::CommPattern(init) => {
                                    let op = ProxyOp::RebootCommunicator(init.id);
                                    self.resources.comms_init.insert(init.id, init);
                                    self.ops.enqueue(op);
                                }
                            }
                        } else {
                            if !state.stream_completed {
                                let op = ProxyOp::PollCommunicatorComplete(comm_id);
                                self.ops.enqueue(op);
                            }
                            Self::shutdown_transport_agents(
                                &mut self.resources.transport_engines_tx,
                                &comm,
                            );
                            self.resources
                                .comms_suspended
                                .insert(comm_id, (comm, state));
                        }
                    }
                }
            },
            Err(TryRecvError::Empty) => (),
            Err(TryRecvError::Disconnected) => {
                panic!("Exchange engine shall never shutdown")
            }
        }
    }

    fn check_transport_reply(&mut self) {
        for (_, transport_rx) in self.resources.transport_engines_rx.iter_mut() {
            if let Ok(msg) = transport_rx.try_recv() {
                match msg {
                    TransportEngineReply::AgentSetup(agent_id, reply) => {
                        let peer_conn = agent_id.peer_conn;
                        let comm = self
                            .resources
                            .comms_init
                            .get_mut(&agent_id.communicator_id)
                            .unwrap();
                        comm.transport_connect
                            .as_mut()
                            .unwrap()
                            .put_peer_agent_setup_message(&peer_conn, reply)
                            .unwrap();
                    }
                    TransportEngineReply::AgentConnect(agent_id, reply) => {
                        let peer_conn = agent_id.peer_conn;
                        let comm = self
                            .resources
                            .comms_init
                            .get_mut(&agent_id.communicator_id)
                            .unwrap();
                        comm.transport_connect
                            .as_mut()
                            .unwrap()
                            .put_peer_agent_connect_message(&peer_conn, reply)
                            .unwrap();
                    }
                    TransportEngineReply::AgentShutdown(agent_id) => {
                        let comm_id = agent_id.communicator_id;
                        let removed = if let Some((communicator, suspend_state)) =
                            self.resources.comms_suspended.get_mut(&comm_id)
                        {
                            suspend_state.agents_pending_shutdown.remove(&agent_id);
                            suspend_state.check_suspended()
                        } else {
                            false
                        };
                        if removed {
                            let (comm, state) =
                                self.resources.comms_suspended.remove(&comm_id).unwrap();
                            let output =
                                state.emit(comm, &self.resources.global_registry.transport_catalog);
                            match output {
                                CommReconfigOutout::CommPattern(init) => {
                                    let op = ProxyOp::RebootCommunicator(init.id);
                                    self.resources.comms_init.insert(init.id, init);
                                    self.ops.enqueue(op);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn check_control_notify(&mut self) {
        if let Ok(msg) = self.resources.control_chan.rx.try_recv() {
            match msg {
                ControlNotification::NewTransportEngine { id, chan } => {
                    self.resources.register_transport_engine(id, chan);
                }
                ControlNotification::NewDaemon { id, chan } => {
                    self.resources.register_daemon_engine(id, chan);
                }
            }
        }
    }

    fn check_daemon_command(&mut self) {
        for (idx, (daemon_id, daemon_rx)) in self.resources.daemon_rx.iter_mut().enumerate() {
            match daemon_rx.try_recv() {
                Ok(msg) => {
                    match msg {
                        ProxyCommand::InitCommunicator(init) => {
                            let mut udp_sport_map = HashMap::new();
                            let mut channel_net_dev_map = HashMap::new();
                            let comm_pattern = self
                                .resources
                                .global_registry
                                .comm_pattern_override
                                .get(&init.communicator_id);
                            if let Some(comm_pattern) = comm_pattern {
                                for channel in comm_pattern.channels.iter() {
                                    if let Some(sport_map) = channel.udp_sport.as_ref() {
                                        for spec in sport_map.iter() {
                                            let send_rank = spec.0;
                                            let recv_rank = spec.1;
                                            if send_rank != init.rank {
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
                                        channel_net_dev_map
                                            .insert(ChannelId(channel.channel_id), net_dev.clone());
                                    }
                                }
                            }
                            let tc = match comm_pattern {
                                Some(pattern) => pattern.ib_traffic_class,
                                None => None,
                            };
                            let profile = CommProfile {
                                buff_sizes: self
                                    .resources
                                    .global_registry
                                    .default_comm_config
                                    .buf_sizes,
                                udp_sport_map,
                                channel_net_device_map: channel_net_dev_map,
                                tc,
                            };
                            let mut comm_init = CommInitState::new(
                                init.communicator_id,
                                self.resources.device_info.cuda_device_idx,
                                *daemon_id,
                                init.rank,
                                init.num_ranks,
                                profile,
                            );
                            if init.rank == 0 {
                                let mut listen_addr = SocketAddr::new(init.root_mccs_addr, 0);
                                let (root_socket, bootstrap_handle) =
                                    bootstrap_create_root(&listen_addr).unwrap();
                                let cmd = ExchangeCommand::RegisterBootstrapHandle(
                                    init.communicator_id,
                                    bootstrap_handle.clone(),
                                );
                                self.resources.exchange_chan.tx.send(cmd).unwrap();
                                // bootstrap root task
                                let fut = bootstrap_root(root_socket, bootstrap_handle.magic).map(
                                    |state| {
                                        state
                                            .map(|_| AsyncTaskOutput::BootstrapRoot)
                                            .map_err(|e| anyhow::Error::new(e))
                                    },
                                );
                                let task = AsyncTask {
                                    comm_id: init.communicator_id,
                                    fut: Box::pin(fut),
                                };
                                self.async_tasks.enqueue(task);
                                comm_init.bootstrap_handle = Some(bootstrap_handle);
                            } else {
                                let root_addr = SocketAddr::new(
                                    init.root_mccs_addr,
                                    self.resources.device_info.listen_port,
                                );
                                let cmd = ExchangeCommand::RecvBootstrapHandle(
                                    init.communicator_id,
                                    root_addr,
                                );
                                self.resources.exchange_chan.tx.send(cmd).unwrap();
                            };
                            self.resources
                                .comms_init
                                .insert(init.communicator_id, comm_init);
                            let op = ProxyOp::InitCommunicator(*daemon_id, init.communicator_id);
                            self.ops.enqueue(op);
                        }
                        ProxyCommand::AllGather(coll) => {
                            let comm = self
                                .resources
                                .communicators
                                .get_mut(&coll.communicator_id)
                                .unwrap();
                            let user_events = self.resources.user_events.get(daemon_id).unwrap();
                            comm.schedule_all_gather(coll, user_events);
                            comm.launch_scheduled_and_record(
                                &mut self.resources.transport_engines_tx,
                            );
                            let sender = self.resources.daemon_tx.get(daemon_id).unwrap();
                            sender.send(ProxyCompletion::AllGather).unwrap();
                        }
                        ProxyCommand::AllReduce(coll) => {
                            let comm = self.resources.communicators.get_mut(&coll.communicator_id);
                            if let Some(comm) = comm {
                                let user_events =
                                    self.resources.user_events.get(daemon_id).unwrap();
                                comm.schedule_all_reduce(coll, user_events);
                                comm.launch_scheduled_and_record(
                                    &mut self.resources.transport_engines_tx,
                                );
                                let sender = self.resources.daemon_tx.get(daemon_id).unwrap();
                                sender.send(ProxyCompletion::AllReduce).unwrap();
                            } else {
                                self.resources.queued_commands.push((*daemon_id, coll));
                            }
                        }
                        ProxyCommand::GroupCall(colls) => {
                            let comm_id = match &colls[0] {
                                command::CollRequest::AllGather(all_gather) => {
                                    all_gather.communicator_id
                                }
                                command::CollRequest::AllReduce(all_reduce) => {
                                    all_reduce.communicator_id
                                }
                            };
                            let comm = self.resources.communicators.get_mut(&comm_id).unwrap();
                            let user_events = self.resources.user_events.get(daemon_id).unwrap();
                            comm.schedule_group_call(colls, user_events);
                            comm.launch_scheduled_and_record(
                                &mut self.resources.transport_engines_tx,
                            );
                            let sender = self.resources.daemon_tx.get(daemon_id).unwrap();
                            sender.send(ProxyCompletion::GroupCall).unwrap();
                        }
                        ProxyCommand::DestroyCommunicator(comm_id) => {
                            let mut comm = self.resources.communicators.remove(&comm_id).unwrap();
                            let request = ExchangeCommand::RemoveCommunicator(comm_id);
                            self.resources.exchange_chan.tx.send(request).unwrap();
                            Self::shutdown_transport_agents(
                                &mut self.resources.transport_engines_tx,
                                &comm,
                            );
                            comm.destory_stream_and_event();
                        }
                        ProxyCommand::RegisterStream(user_stream, user_event_handle) => {
                            let user_events =
                                self.resources.user_events.get_mut(daemon_id).unwrap();
                            let mut event = std::ptr::null_mut();
                            unsafe {
                                let event_handle = user_event_handle.into();
                                cuda_warning!(cudaIpcOpenEventHandle(&mut event, event_handle));
                            }
                            user_events.insert(user_stream, event);
                            let sender = self.resources.daemon_tx.get(daemon_id).unwrap();
                            sender.send(ProxyCompletion::RegisterStream).unwrap();
                        }
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    self.resources.daemon_shutdown.push((*daemon_id, idx));
                }
                Err(TryRecvError::Empty) => (),
            }
        }
        for (daemon_id, idx) in self.resources.daemon_shutdown.drain(..).rev() {
            self.resources.daemon_tx.remove(&daemon_id);
            self.resources.daemon_rx.swap_remove(idx);
            self.resources.user_events.remove(&daemon_id);
            self.resources.communicators.retain(|_, comm| {
                // use crate::transport::net::agent::QOS_DISABLE;
                // if comm.id == CommunicatorId(201) {
                //     QOS_DISABLE.store(true, std::sync::atomic::Ordering::Relaxed);
                // }
                if comm.daemon == daemon_id {
                    Self::shutdown_transport_agents(&mut self.resources.transport_engines_tx, comm);
                    let request = ExchangeCommand::RemoveCommunicator(comm.id);
                    self.resources.exchange_chan.tx.send(request).unwrap();
                    comm.destory_stream_and_event();
                }
                comm.daemon != daemon_id
            });
        }
    }

    fn shutdown_transport_agents(
        transport_txs: &mut HashMap<TransportEngineId, Sender<TransportEngineRequest>>,
        comm: &Communicator,
    ) {
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
                            let sender = transport_txs.get_mut(&transport_engine).unwrap();
                            let request = TransportEngineRequest::AgentShutdown(agent_id);
                            sender.send(request).unwrap();
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
                            let sender = transport_txs.get_mut(&transport_engine).unwrap();
                            let request = TransportEngineRequest::AgentShutdown(agent_id);
                            sender.send(request).unwrap();
                        }
                    }
                }
            }
        }
    }
}

impl Communicator {
    fn destory_stream_and_event(&mut self) {
        unsafe {
            if !self.stream.is_null() {
                cuda_warning!(cuda_runtime_sys::cudaStreamDestroy(self.stream));
                self.stream = std::ptr::null_mut();
            }
            if !self.event.is_null() {
                cuda_warning!(cuda_runtime_sys::cudaEventDestroy(self.event));
                self.event = std::ptr::null_mut();
            }
        }
    }

    #[inline]
    fn wait_user_event(&self, user_event: cudaEvent_t) {
        unsafe {
            cuda_warning!(cudaStreamWaitEvent(self.stream, user_event, 0));
        }
    }

    fn schedule_all_gather(
        &mut self,
        coll: command::AllGatherRequest,
        user_events: &HashMap<usize, cudaEvent_t>,
    ) {
        let user_event = *user_events.get(&coll.user_stream).unwrap();
        self.wait_user_event(user_event);

        let send_buf = DeviceNonNull::new(coll.send_buf_addr as *mut u8).unwrap();
        let recv_buf = DeviceNonNull::new(coll.recv_buf_addr as *mut u8).unwrap();
        let task = CollTask {
            func: TaskFuncType::AllGather,
            send_buf,
            recv_buf,
            count: coll.size,
            root: 0,
            data_type: TaskDataType::Uint8,
            reduce_op: None,
            chunk_steps: ALLGATHER_CHUNK_STEPS,
            slice_steps: ALLGATHER_SLICE_STEPS,
        };
        self.task_queue.coll_queue.push_back(task);
    }

    fn schedule_all_reduce(
        &mut self,
        coll: command::AllReduceRequest,
        user_events: &HashMap<usize, cudaEvent_t>,
    ) {
        let user_event = *user_events.get(&coll.user_stream).unwrap();
        self.wait_user_event(user_event);

        let send_buf = DeviceNonNull::new(coll.send_buf_addr as *mut u8).unwrap();
        let recv_buf = DeviceNonNull::new(coll.recv_buf_addr as *mut u8).unwrap();
        let task = CollTask {
            func: TaskFuncType::AllReduce,
            send_buf,
            recv_buf,
            count: coll.size,
            root: 0,
            data_type: coll.data_type,
            reduce_op: Some(TaskReduceOp {
                op: coll.op_type,
                arg: 0,
            }),
            chunk_steps: ALLREDUCE_CHUNK_STEPS,
            slice_steps: ALLREDUCE_SLICE_STEPS,
        };
        self.task_queue.coll_queue.push_back(task);
    }

    fn schedule_group_call(
        &mut self,
        colls: Vec<command::CollRequest>,
        user_events: &HashMap<usize, cudaEvent_t>,
    ) {
        for coll in colls.into_iter() {
            match coll {
                command::CollRequest::AllGather(all_gather) => {
                    self.schedule_all_gather(all_gather, user_events);
                }
                command::CollRequest::AllReduce(all_reduce) => {
                    self.schedule_all_reduce(all_reduce, user_events);
                }
            }
        }
    }

    #[inline]
    fn record_backend_event(&self) {
        unsafe {
            cuda_warning!(cudaEventRecord(self.event, self.stream));
        };
    }

    #[inline]
    fn launch_scheduled_and_record(
        &mut self,
        transport_txs: &mut HashMap<TransportEngineId, Sender<TransportEngineRequest>>,
    ) {
        self.pre_launch_schedule(transport_txs, self.cuda_dev);
        self.launch_plan();
        self.record_backend_event();
    }
}
