use std::collections::HashMap;
use std::sync::Arc;

use crossbeam::channel::{Receiver, Sender};
use cuda_runtime_sys::{cudaError, cudaEventQuery};

use super::command::{ProxyCommand, ProxyCompletion};
use super::init::{CommInitStage, CommInitState};
use super::message::ProxyPeerMessage;
use super::op::ProxyOp;
use super::task::{CollTask, TaskDataType, TaskFuncType};
use super::DeviceInfo;
use crate::comm::{CommProfile, Communicator, CommunicatorId, PeerInfo, PeerType};
use crate::cuda::ptr::DeviceNonNull;
use crate::daemon::DaemonId;
use crate::message::{ControlNotification, ControlRequest};
use crate::proxy::init::PeerConnConstruct;
use crate::registry::GlobalRegistry;
use crate::transport::channel::{ConnType, PeerConnId, PeerConnector};
use crate::transport::engine::TransportEngineId;
use crate::transport::message::{TransportEngineReply, TransportEngineRequest};
use crate::transport::transporter::{
    ConnectInfo, TransportAgentId, TransportConnect, TransportSetup,
};
use crate::utils::pool::WorkPool;

pub struct ProxyResources {
    pub device_info: DeviceInfo,
    pub control_tx: Sender<ControlRequest>,
    pub control_rx: Receiver<ControlNotification>,
    pub daemon_tx: HashMap<DaemonId, Sender<ProxyCompletion>>,
    pub daemon_rx: Vec<(DaemonId, Receiver<ProxyCommand>)>,
    pub proxy_peer_tx: Vec<Sender<ProxyPeerMessage>>,
    pub proxy_peer_rx: Receiver<ProxyPeerMessage>,
    pub comms_init: HashMap<CommunicatorId, CommInitState>,
    pub communicators: HashMap<CommunicatorId, Communicator>,
    pub global_registry: Arc<GlobalRegistry>,
    pub transport_engines_tx: HashMap<TransportEngineId, Sender<TransportEngineRequest>>,
    pub transport_engines_rx: Vec<(TransportEngineId, Receiver<TransportEngineReply>)>,
    pub transport_submission_pool: HashMap<TransportEngineId, Vec<TransportEngineRequest>>,
}

impl ProxyResources {
    fn register_transport_engine(
        &mut self,
        id: TransportEngineId,
        tx: Sender<TransportEngineRequest>,
        rx: Receiver<TransportEngineReply>,
    ) {
        let pool = self.transport_submission_pool.remove(&id);
        if let Some(requests) = pool {
            for req in requests {
                tx.send(req).unwrap();
            }
        }
        self.transport_engines_tx.insert(id, tx);
        let rx_idx = self
            .transport_engines_rx
            .iter()
            .position(|(rx_id, _)| id == *rx_id);
        if let Some(idx) = rx_idx {
            self.transport_engines_rx.swap_remove(idx);
        }
        self.transport_engines_rx.push((id, rx));
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

    fn exchange_connect_info(
        comm: &CommInitState,
        peer_conn: &PeerConnId,
        peer_connect_info: ConnectInfo,
        peer_proxy_tx: &mut [Sender<ProxyPeerMessage>],
    ) {
        let remote_conn_type = match peer_conn.conn_type {
            ConnType::Send => ConnType::Recv,
            ConnType::Recv => ConnType::Send,
        };
        let remote_peer_conn = PeerConnId {
            peer_rank: comm.rank,
            channel: peer_conn.channel,
            conn_index: peer_conn.conn_index,
            conn_type: remote_conn_type,
        };
        let peer_rank = peer_conn.peer_rank;
        let peer_info = comm.peers_info.get(&peer_rank).unwrap();
        match peer_info.peer_type {
            PeerType::Local => panic!("self-connection encountered"),
            PeerType::IntraNode => {
                let peer_message = ProxyPeerMessage::ConnectInfoExchange(
                    comm.id,
                    remote_peer_conn,
                    peer_connect_info,
                );
                peer_proxy_tx[peer_rank].send(peer_message).unwrap();
            }
            PeerType::InterNode => todo!("inter-node connect info exchange is not implemented"),
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
        if comm.stage == CommInitStage::RegisterRank {
            let rank_info = PeerInfo {
                peer_type: PeerType::Local,
                host: self.device_info.host,
                cuda_device_idx: self.device_info.cuda_device_idx,
            };
            self.global_registry.register_communicator_rank(
                comm_id,
                comm.rank,
                comm.num_ranks,
                &rank_info,
            );
            comm.peers_info.insert(comm.rank, rank_info);

            comm.peers_await_exchange.extend(0..comm.rank);
            comm.peers_await_exchange
                .extend((comm.rank + 1)..comm.num_ranks);
            comm.stage = CommInitStage::PeerInfoExchange;
        } else if comm.stage == CommInitStage::PeerInfoExchange {
            if comm.peers_info.len() == comm.num_ranks {
                let patterns = self
                    .global_registry
                    .arbitrate_comm_patterns(comm_id, comm.rank);
                if let Some(patterns) = patterns {
                    comm.comm_patterns = patterns;
                    for pattern in comm.comm_patterns.iter() {
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
                        comm.to_setup.push_back(ring_next);
                        comm.to_setup.push_back(ring_prev);
                        comm.await_connections += 2;
                    }
                    comm.stage = CommInitStage::ConnectChannel;
                }
            }
            self.global_registry.query_communicator_peers(
                comm_id,
                comm.rank,
                &mut comm.peers_await_exchange,
                &mut comm.peers_info,
            );
        } else if let Some(peer_conn) = comm.to_setup.pop_front() {
            let transporter = self
                .global_registry
                .arbitrate_conn_transporter(comm_id, comm.rank, &peer_conn);
            let setup_result = match peer_conn.conn_type {
                ConnType::Send => transporter.send_setup(
                    &comm.profile,
                    &peer_conn,
                    &self.global_registry.transport_catalog,
                ),
                ConnType::Recv => transporter.recv_setup(
                    &comm.profile,
                    &peer_conn,
                    &self.global_registry.transport_catalog,
                ),
            };
            match setup_result {
                TransportSetup::PreAgentCb {
                    agent_request,
                    setup_resources,
                } => {
                    let agent = TransportAgentId {
                        communicator_id: comm_id,
                        client_rank: comm.rank,
                        client_cuda_dev: self.device_info.cuda_device_idx,
                        peer_conn,
                    };
                    let transport_engine_idx = self
                        .global_registry
                        .transport_delegator
                        .assign_transport_engine(
                            self.device_info.cuda_device_idx,
                            agent,
                            &mut self.control_tx,
                        );
                    let request = match peer_conn.conn_type {
                        ConnType::Send => {
                            TransportEngineRequest::AgentSetup(transporter, agent, agent_request)
                        }
                        ConnType::Recv => {
                            TransportEngineRequest::AgentSetup(transporter, agent, agent_request)
                        }
                    };
                    Self::send_transport_request(
                        request,
                        transport_engine_idx,
                        &mut self.transport_engines_tx,
                        &mut self.transport_submission_pool,
                    );

                    let construct = PeerConnConstruct {
                        transporter,
                        resources: setup_resources,
                    };
                    comm.peer_setup_pre_agent.insert(peer_conn, construct);
                }
                TransportSetup::Setup {
                    peer_connect_info,
                    setup_resources,
                } => {
                    let construct = PeerConnConstruct {
                        transporter,
                        resources: setup_resources,
                    };
                    comm.peer_setup.insert(peer_conn, construct);
                    Self::exchange_connect_info(
                        comm,
                        &peer_conn,
                        peer_connect_info,
                        &mut self.proxy_peer_tx,
                    );
                }
            }
        } else if let Some((peer_conn, agent_reply)) = comm.to_setup_agent_cb.pop_front() {
            let construct = comm.peer_setup_pre_agent.remove(&peer_conn).unwrap();
            let setup_result = match peer_conn.conn_type {
                ConnType::Send => construct.transporter.send_setup_agent_callback(
                    &peer_conn,
                    agent_reply,
                    construct.resources,
                ),
                ConnType::Recv => construct.transporter.recv_setup_agent_callback(
                    &peer_conn,
                    agent_reply,
                    construct.resources,
                ),
            };
            match setup_result {
                TransportSetup::PreAgentCb { .. } => {
                    panic!("PreAgentCb variant is not expected")
                }
                TransportSetup::Setup {
                    peer_connect_info,
                    setup_resources,
                } => {
                    let construct = PeerConnConstruct {
                        transporter: construct.transporter,
                        resources: setup_resources,
                    };
                    comm.peer_setup.insert(peer_conn, construct);
                    Self::exchange_connect_info(
                        comm,
                        &peer_conn,
                        peer_connect_info,
                        &mut self.proxy_peer_tx,
                    );
                }
            }
        } else if let Some((peer_conn, conn_info)) = comm.to_connect.pop_front() {
            let setup = comm.peer_setup.remove(&peer_conn).unwrap();
            let connect_result = match peer_conn.conn_type {
                ConnType::Send => {
                    setup
                        .transporter
                        .send_connect(&peer_conn, conn_info, setup.resources)
                }
                ConnType::Recv => {
                    setup
                        .transporter
                        .recv_connect(&peer_conn, conn_info, setup.resources)
                }
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
                        peer_conn,
                    };
                    let transport_engine = *comm
                        .peer_transport_assigned
                        .entry(peer_conn)
                        .or_insert_with(|| {
                            self.global_registry
                                .transport_delegator
                                .assign_transport_engine(
                                    self.device_info.cuda_device_idx,
                                    agent,
                                    &mut self.control_tx,
                                )
                        });
                    let request = match peer_conn.conn_type {
                        ConnType::Send => TransportEngineRequest::AgentConnect(
                            setup.transporter,
                            agent,
                            agent_request,
                        ),
                        ConnType::Recv => TransportEngineRequest::AgentConnect(
                            setup.transporter,
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

                    let construct = PeerConnConstruct {
                        transporter: setup.transporter,
                        resources: transport_resources,
                    };
                    comm.peer_connect_pre_agent.insert(peer_conn, construct);
                }
                TransportConnect::Connect {
                    conn_info,
                    transport_resources,
                } => {
                    let transport_engine = comm.peer_transport_assigned.get(&peer_conn).copied();

                    let peer_connector = PeerConnector {
                        conn_info,
                        transport_agent_engine: transport_engine,
                        transporter: setup.transporter,
                        transport_resources,
                    };
                    comm.peer_connected.insert(peer_conn, peer_connector);
                    comm.await_connections -= 1;
                }
            }
        } else if let Some((peer_conn, agent_reply)) = comm.to_connect_agent_cb.pop_front() {
            let construct = comm.peer_connect_pre_agent.remove(&peer_conn).unwrap();
            let connect_result = match peer_conn.conn_type {
                ConnType::Send => construct.transporter.send_connect_agent_callback(
                    &peer_conn,
                    agent_reply,
                    construct.resources,
                ),
                ConnType::Recv => construct.transporter.recv_connect_agent_callback(
                    &peer_conn,
                    agent_reply,
                    construct.resources,
                ),
            };
            match connect_result {
                TransportConnect::PreAgentCb { .. } => {
                    panic!("PreAgentCb variant is not expected")
                }
                TransportConnect::Connect {
                    conn_info,
                    transport_resources,
                } => {
                    let transport_engine = comm.peer_transport_assigned.get(&peer_conn).copied();

                    let peer_connector = PeerConnector {
                        conn_info,
                        transport_agent_engine: transport_engine,
                        transporter: construct.transporter,
                        transport_resources,
                    };
                    comm.peer_connected.insert(peer_conn, peer_connector);
                    comm.await_connections -= 1;
                }
            }
        }

        if comm.stage == CommInitStage::ConnectChannel && comm.await_connections == 0 {
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
                        comm.to_setup_agent_cb.push_back((peer_conn, reply));
                    }
                    TransportEngineReply::AgentConnect(agent_id, reply) => {
                        let peer_conn = agent_id.peer_conn;
                        let comm = self.comms_init.get_mut(&agent_id.communicator_id).unwrap();
                        comm.to_connect_agent_cb.push_back((peer_conn, reply));
                    }
                }
            }
        }
    }

    fn check_proxy_peer_message(&mut self) {
        if let Ok(msg) = self.proxy_peer_rx.try_recv() {
            match msg {
                ProxyPeerMessage::ConnectInfoExchange(comm_id, peer_conn, conn_info) => {
                    let comm = self.comms_init.get_mut(&comm_id).unwrap();
                    comm.to_connect.push_back((peer_conn, conn_info));
                }
            }
        }
    }

    fn check_control_notify(&mut self) {
        if let Ok(msg) = self.control_rx.try_recv() {
            match msg {
                ControlNotification::NewTransportEngine {
                    id,
                    request_tx,
                    reply_rx,
                } => {
                    self.register_transport_engine(id, request_tx, reply_rx);
                }
            }
        }
    }
}

impl ProxyResources {
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
                let comm = self.communicators.get_mut(comm_id).unwrap();
                unsafe {
                    let state = cudaEventQuery(comm.event);
                    if state == cudaError::cudaSuccess {
                        let _ = self
                            .daemon_tx
                            .get_mut(daemon_id)
                            .unwrap()
                            .send(ProxyCompletion::AllGather);
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
}

impl ProxyEngine {
    pub fn mainloop(&mut self) {
        loop {
            self.check_daemon_command();
            self.resources.check_proxy_peer_message();
            self.resources.check_transport_reply();
            self.resources.check_control_notify();

            self.ops.progress(|op| self.resources.process_op(op));
        }
    }
}

impl ProxyEngine {
    pub fn check_daemon_command(&mut self) {
        for (daemon_id, daemon_rx) in self.resources.daemon_rx.iter_mut() {
            if let Ok(msg) = daemon_rx.try_recv() {
                match msg {
                    ProxyCommand::InitCommunicator(init) => {
                        let profile = CommProfile {
                            buff_sizes: [8 * 1024 * 1024],
                        };
                        let comm_init = CommInitState::new(
                            init.communicator_id,
                            init.rank,
                            init.num_ranks,
                            profile,
                        );
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
                            chunk_steps: 0,
                            slice_steps: 0,
                        };
                        comm.task_queue.coll_queue.push_back(task);
                        comm.pre_launch_schedule();
                        comm.launch_plan();
                        let op = ProxyOp::PollCudaEvent(*daemon_id, coll.communicator_id);
                        self.ops.enqueue(op);
                    }
                }
            }
        }
    }
}
