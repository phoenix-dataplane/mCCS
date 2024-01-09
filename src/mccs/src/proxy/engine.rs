use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::task::{Context, Poll};

use crossbeam::channel::{Receiver, Sender, TryRecvError};
use futures::future::BoxFuture;
use futures::FutureExt;

use cuda_runtime_sys::{
    cudaError, cudaEventCreateWithFlags, cudaEventDisableTiming, cudaEventInterprocess,
    cudaEventQuery, cudaEventRecord, cudaIpcEventHandle_t, cudaIpcGetEventHandle,
    cudaIpcOpenEventHandle, cudaStreamWaitEvent,
};

use super::command::{ProxyCommand, ProxyCompletion};
use super::init::{CommInitStage, CommInitState};
use super::op::ProxyOp;
use super::task::{CollTask, TaskDataType, TaskFuncType};
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
use crate::exchange::command::{ExchangeCommand, ExchangeCompletion};
use crate::message::{ControlCommand, ControlRequest};
use crate::pattern::{ALLGATHER_CHUNK_STEPS, ALLGATHER_SLICE_STEPS};
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
    pub control_chan: DuplexChannel<ControlRequest, ControlCommand>,
    // daemons
    pub daemon_tx: HashMap<DaemonId, Sender<ProxyCompletion>>,
    pub daemon_rx: Vec<(DaemonId, Receiver<ProxyCommand>)>,
    // exchange engine
    pub exchange_tx: Sender<ExchangeCommand>,
    pub exchange_rx: Receiver<ExchangeCompletion>,
    // communications and transport
    pub comms_init: HashMap<CommunicatorId, CommInitState>,
    pub communicators: HashMap<CommunicatorId, Communicator>,
    pub global_registry: Arc<GlobalRegistry>,
    pub transport_engines_tx: HashMap<TransportEngineId, Sender<TransportEngineRequest>>,
    pub transport_engines_rx: Vec<(TransportEngineId, Receiver<TransportEngineReply>)>,
    pub transport_submission_pool: HashMap<TransportEngineId, Vec<TransportEngineRequest>>,
    pub task_submit_pool: Vec<AsyncTask>,
}

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
        self.daemon_tx.insert(id, chan.tx);
        self.daemon_rx.push((id, chan.rx));
    }

    fn register_transport_engine(
        &mut self,
        id: TransportEngineId,
        chan: DuplexChannel<TransportEngineRequest, TransportEngineReply>,
    ) {
        let pool = self.transport_submission_pool.remove(&id);
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
        submission_pool: &mut HashMap<TransportEngineId, Vec<TransportEngineRequest>>,
    ) {
        use crossbeam::channel::SendError;

        let sender = transport_tx.get_mut(&transport_engine);
        if let Some(sender) = sender {
            match sender.send(request) {
                Ok(()) => (),
                Err(SendError(request)) => {
                    // disconnected
                    Self::enqueue_submission_pool(submission_pool, transport_engine, request);
                }
            }
        } else {
            Self::enqueue_submission_pool(submission_pool, transport_engine, request);
        }
    }

    fn enqueue_submission_pool(
        pool: &mut HashMap<TransportEngineId, Vec<TransportEngineRequest>>,
        transport_engine: TransportEngineId,
        request: TransportEngineRequest,
    ) {
        let queue = pool.entry(transport_engine).or_insert_with(Vec::new);
        queue.push(request);
    }

    fn init_communicator(&mut self, comm_id: CommunicatorId) -> bool {
        let comm = self.comms_init.get_mut(&comm_id).unwrap();
        if comm.stage == CommInitStage::BootstrapInit {
            if let Some(handle) = comm.bootstrap_handle.take() {
                let mut listen_addr = self.device_info.host.clone();
                listen_addr.set_port(0);
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
                self.task_submit_pool.push(task);
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
                self.task_submit_pool.push(task);
                comm.stage = CommInitStage::AllGatherPeerInfo;
            }
        } else if comm.stage == CommInitStage::AllGatherPeerInfo {
            if comm.peers_info.is_some() {
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

                let channel = crate::comm::ChannelCommPattern {
                    channel: ChannelId(0),
                    ring: ring_pattern,
                };
                let mut channels = BTreeMap::new();
                channels.insert(ChannelId(0), channel);

                let mut transport_connect =
                    TransportConnectState::new(comm.rank, comm.num_ranks, channels.len());
                for pattern in channels.values() {
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
                                &mut self.transport_submission_pool,
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
                                &mut self.transport_submission_pool,
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
                    self.task_submit_pool.push(task);
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
    fn check_transport_reply(&mut self) {
        for (_, transport_rx) in self.transport_engines_rx.iter_mut() {
            if let Ok(msg) = transport_rx.try_recv() {
                match msg {
                    TransportEngineReply::AgentSetup(agent_id, reply) => {
                        let peer_conn = agent_id.peer_conn;
                        let comm = self.comms_init.get_mut(&agent_id.communicator_id).unwrap();
                        comm.transport_connect
                            .as_mut()
                            .unwrap()
                            .put_peer_agent_setup_message(&peer_conn, reply)
                            .unwrap();
                    }
                    TransportEngineReply::AgentConnect(agent_id, reply) => {
                        let peer_conn = agent_id.peer_conn;
                        let comm = self.comms_init.get_mut(&agent_id.communicator_id).unwrap();
                        comm.transport_connect
                            .as_mut()
                            .unwrap()
                            .put_peer_agent_connect_message(&peer_conn, reply)
                            .unwrap();
                    }
                }
            }
        }
    }

    fn check_control_notify(&mut self) {
        if let Ok(msg) = self.control_chan.rx.try_recv() {
            match msg {
                ControlCommand::NewTransportEngine { id, chan } => {
                    self.register_transport_engine(id, chan);
                }
                ControlCommand::NewDaemon { id, chan } => {
                    self.register_daemon_engine(id, chan);
                }
            }
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
                    let _ = self
                        .daemon_tx
                        .get_mut(daemon_id)
                        .unwrap()
                        .send(ProxyCompletion::InitCommunicator);
                }
                res
            }
            ProxyOp::PollCudaEvent(daemon_id, comm_id) => {
                todo!("Should not be invoked for now");
                let comm = self.communicators.get_mut(comm_id).unwrap();
                unsafe {
                    let state = cudaEventQuery(comm.event);
                    if state == cudaError::cudaSuccess {
                        let _ = self
                            .daemon_tx
                            .get_mut(daemon_id)
                            .unwrap()
                            .send(ProxyCompletion::AllGather(todo!()));
                        true
                    } else {
                        false
                    }
                }
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
    pub fn mainloop(&mut self) {
        loop {
            self.check_daemon_command();
            self.check_exchange_reply();
            self.progress_async_tasks();
            self.resources.check_transport_reply();
            self.resources.check_control_notify();

            self.ops.progress(|op| self.resources.process_op(op));

            for task in self.resources.task_submit_pool.drain(..) {
                self.async_tasks.enqueue(task);
            }
        }
    }
}

impl ProxyEngine {
    fn progress_async_tasks(&mut self) {
        self.async_tasks
            .progress(|x| self.resources.progress_async_task(x));
    }

    pub fn check_exchange_reply(&mut self) {
        match self.resources.exchange_rx.try_recv() {
            Ok(msg) => match msg {
                ExchangeCompletion::RegisterBootstrapHandle => {}
                ExchangeCompletion::RecvBootstrapHandle(comm_id, handle) => {
                    let comm = self.resources.comms_init.get_mut(&comm_id).unwrap();
                    comm.bootstrap_handle = Some(handle);
                }
            },
            Err(TryRecvError::Empty) => (),
            Err(TryRecvError::Disconnected) => {
                unreachable!("Exchange engine shall never shutdown")
            }
        }
    }

    pub fn check_daemon_command(&mut self) {
        for (daemon_id, daemon_rx) in self.resources.daemon_rx.iter_mut() {
            if let Ok(msg) = daemon_rx.try_recv() {
                match msg {
                    ProxyCommand::InitCommunicator(init) => {
                        // TODO: get default profile from central registry
                        let profile = CommProfile {
                            buff_sizes: [8 * 1024 * 1024],
                        };
                        let mut comm_init = CommInitState::new(
                            init.communicator_id,
                            init.rank,
                            init.num_ranks,
                            profile,
                        );
                        if init.rank == 0 {
                            let mut listen_addr = init.root_mccs_addr;
                            listen_addr.set_port(0);
                            let (root_socket, bootstrap_handle) =
                                bootstrap_create_root(&listen_addr).unwrap();
                            let cmd = ExchangeCommand::RegisterBootstrapHandle(
                                init.communicator_id,
                                bootstrap_handle.clone(),
                            );
                            self.resources.exchange_tx.send(cmd).unwrap();
                            // bootstrap root task
                            let fut =
                                bootstrap_root(root_socket, bootstrap_handle.magic).map(|state| {
                                    state
                                        .map(|_| AsyncTaskOutput::BootstrapRoot)
                                        .map_err(|e| anyhow::Error::new(e))
                                });
                            let task = AsyncTask {
                                comm_id: init.communicator_id,
                                fut: Box::pin(fut),
                            };
                            self.async_tasks.enqueue(task);
                            comm_init.bootstrap_handle = Some(bootstrap_handle);
                        } else {
                            let cmd = ExchangeCommand::RecvBootstrapHandle(
                                init.communicator_id,
                                init.root_mccs_addr,
                            );
                            self.resources.exchange_tx.send(cmd).unwrap();
                        };
                        self.resources
                            .comms_init
                            .insert(init.communicator_id, comm_init);
                        let op = ProxyOp::InitCommunicator(*daemon_id, init.communicator_id);
                        self.ops.enqueue(op);
                    }
                    ProxyCommand::AllGather(coll) => {
                        // recover event and register waiting order
                        let event = {
                            let event_handle = coll.app_ipc_event_handle.into();
                            let mut event = std::ptr::null_mut();
                            cuda_warning!(unsafe {
                                cudaIpcOpenEventHandle(&mut event, event_handle)
                            });
                            event
                        };

                        let comm = self
                            .resources
                            .communicators
                            .get_mut(&coll.communicator_id)
                            .unwrap();
                        cuda_warning!(unsafe { cudaStreamWaitEvent(comm.stream, event, 0) });
                        // prepare arguments
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
                        comm.task_queue.coll_queue.push_back(task);
                        comm.pre_launch_schedule(
                            &mut self.resources.transport_submission_pool,
                            &self
                                .resources
                                .global_registry
                                .transport_delegator
                                .agent_assignments,
                            self.resources.device_info.cuda_device_idx,
                        );
                        comm.launch_plan();

                        // record event for daemon_stream
                        let handle = unsafe {
                            let mut event = std::ptr::null_mut();
                            cuda_warning!(cudaEventCreateWithFlags(
                                &mut event,
                                cudaEventInterprocess | cudaEventDisableTiming
                            ));
                            cuda_warning!(cudaEventRecord(event, comm.stream));
                            let mut handle = cudaIpcEventHandle_t::default();
                            cuda_warning!(cudaIpcGetEventHandle(&mut handle, event));
                            handle
                        };
                        let _ = self
                            .resources
                            .daemon_tx
                            .get_mut(daemon_id)
                            .unwrap()
                            .send(ProxyCompletion::AllGather(handle.into()));
                        // let op = ProxyOp::PollCudaEvent(*daemon_id, coll.communicator_id);
                        // self.ops.enqueue(op);
                    }
                }
            }
        }
    }
}
