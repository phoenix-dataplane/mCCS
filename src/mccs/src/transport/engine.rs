use std::collections::HashMap;
use std::sync::Arc;
use std::task::{Context, Poll};

use crossbeam::channel::TryRecvError;
use futures::future::BoxFuture;
use futures::FutureExt;

use qos_service::QosSchedule;

use crate::engine::{Engine, EngineStatus};
use crate::registry::GlobalRegistry;
use crate::utils::duplex_chan::DuplexChannel;
use crate::utils::pool::WorkPool;

use super::channel::ConnType;
use super::message::{TransportEngineReply, TransportEngineRequest};
use super::op::{TransportOp, TransportOpState};
use super::queue::TransrportOpQueue;
use super::transporter::{AgentMessage, AnyResources, TransportAgentId, Transporter};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransportEngineId {
    pub cuda_device_idx: i32,
    pub index: u32,
}

pub struct TransportAgent {
    transporter: &'static dyn Transporter,
    agent_resources: AnyResources,
}

enum AsyncTaskResult {
    Setup {
        setup_resources: AnyResources,
        reply: AgentMessage,
    },
    Connect {
        agent_resources: AnyResources,
        reply: AgentMessage,
    },
}

pub struct AsyncTask {
    agent_id: TransportAgentId,
    transporter: &'static dyn Transporter,
    task: BoxFuture<'static, AsyncTaskResult>,
}

fn new_setup_task(
    transporter: &'static dyn Transporter,
    id: TransportAgentId,
    request: AgentMessage,
    registry: &GlobalRegistry,
) -> AsyncTask {
    let setup = match id.peer_conn.conn_type {
        ConnType::Send => {
            transporter.agent_send_setup(id, request, Arc::clone(&registry.transport_catalog))
        }
        ConnType::Recv => {
            transporter.agent_recv_setup(id, request, Arc::clone(&registry.transport_catalog))
        }
    };
    let task = setup.map(|res| {
        let (resources, reply) = res.unwrap();
        AsyncTaskResult::Setup {
            setup_resources: resources,
            reply,
        }
    });
    let pinned = Box::pin(task);
    AsyncTask {
        agent_id: id,
        transporter,
        task: pinned,
    }
}

fn new_connect_task(
    transporter: &'static dyn Transporter,
    id: TransportAgentId,
    request: AgentMessage,
    setup_resources: Option<AnyResources>,
) -> AsyncTask {
    let connect = match id.peer_conn.conn_type {
        ConnType::Send => transporter.agent_send_connect(id, request, setup_resources),
        ConnType::Recv => transporter.agent_recv_connect(id, request, setup_resources),
    };
    let task = connect.map(|res| {
        let (resources, reply) = res.unwrap();
        AsyncTaskResult::Connect {
            agent_resources: resources,
            reply,
        }
    });
    let pinned = Box::pin(task);
    AsyncTask {
        agent_id: id,
        transporter,
        task: pinned,
    }
}

pub struct TransportEngineResources {
    pub agent_setup: HashMap<TransportAgentId, AnyResources>,
    pub agent_connected: HashMap<TransportAgentId, TransportAgent>,
    pub proxy_chan: Vec<DuplexChannel<TransportEngineReply, TransportEngineRequest>>,
    pub global_registry: GlobalRegistry,
    pub qos_schedule: QosSchedule,
}

impl TransportEngineResources {
    fn progress_op(&mut self, agent_id: &TransportAgentId, op: &mut TransportOp) -> bool {
        let agent = self.agent_connected.get_mut(agent_id).unwrap();
        match agent_id.peer_conn.conn_type {
            ConnType::Send => agent.transporter.agent_send_progress_op(
                op,
                &mut agent.agent_resources,
                &self.qos_schedule,
            ),
            ConnType::Recv => agent.transporter.agent_recv_progress_op(
                op,
                &mut agent.agent_resources,
                &self.qos_schedule,
            ),
        }
        .unwrap();
        op.state == TransportOpState::Completed
    }

    fn progress_async_task(&mut self, task: &mut AsyncTask) -> bool {
        let waker = futures::task::noop_waker_ref();
        let mut cx = Context::from_waker(waker);
        let poll = task.task.as_mut().poll(&mut cx);
        match poll {
            Poll::Ready(result) => {
                match result {
                    AsyncTaskResult::Setup {
                        setup_resources,
                        reply,
                    } => {
                        let reply = TransportEngineReply::AgentSetup(task.agent_id, reply);
                        self.proxy_chan[task.agent_id.client_cuda_dev as usize]
                            .tx
                            .send(reply)
                            .unwrap();
                        self.agent_setup.insert(task.agent_id, setup_resources);
                    }
                    AsyncTaskResult::Connect {
                        agent_resources,
                        reply,
                    } => {
                        let connected = TransportAgent {
                            transporter: task.transporter,
                            agent_resources,
                        };
                        let reply = TransportEngineReply::AgentConnect(task.agent_id, reply);
                        self.proxy_chan[task.agent_id.client_cuda_dev as usize]
                            .tx
                            .send(reply)
                            .unwrap();
                        self.agent_connected.insert(task.agent_id, connected);
                    }
                }
                true
            }
            Poll::Pending => false,
        }
    }
}

pub struct TransportEngine {
    pub id: TransportEngineId,
    pub resources: TransportEngineResources,
    pub async_tasks: WorkPool<AsyncTask>,
    pub op_queue: TransrportOpQueue,
}

impl TransportEngine {
    pub fn new(
        id: TransportEngineId,
        proxy_chan: Vec<DuplexChannel<TransportEngineReply, TransportEngineRequest>>,
        global_registry: GlobalRegistry,
        qos_schedule: QosSchedule,
    ) -> Self {
        let resources = TransportEngineResources {
            agent_setup: HashMap::new(),
            agent_connected: HashMap::new(),
            proxy_chan,
            global_registry,
            qos_schedule,
        };
        let engine = TransportEngine {
            id,
            resources,
            async_tasks: WorkPool::new(),
            op_queue: TransrportOpQueue::new(),
        };
        engine
    }
}

impl TransportEngine {
    fn progress_ops(&mut self) {
        let removed_agents = self
            .op_queue
            .progress_ops(|agent_id, op| self.resources.progress_op(agent_id, op));
        for agent_id in removed_agents.drain(..) {
            self.resources
                .global_registry
                .transport_delegator
                .register_agent_shutdown(self.id);
            self.resources.agent_connected.remove(&agent_id);
            let reply = TransportEngineReply::AgentShutdown(agent_id);
            log::info!("shutdown {:?}", agent_id);
            self.resources.proxy_chan[agent_id.client_cuda_dev as usize]
                .tx
                .send(reply)
                .unwrap();
        }
    }

    fn progress_async_tasks(&mut self) {
        self.async_tasks
            .progress(|x| self.resources.progress_async_task(x));
    }

    fn check_proxy_requests(&mut self) {
        for chan in self.resources.proxy_chan.iter_mut() {
            match chan.rx.try_recv() {
                Ok(request) => {
                    match request {
                        TransportEngineRequest::AgentSetup(transporter, agent_id, request) => {
                            let task = new_setup_task(
                                transporter,
                                agent_id,
                                request,
                                &self.resources.global_registry,
                            );
                            self.async_tasks.enqueue(task);
                        }
                        TransportEngineRequest::AgentConnect(transporter, agent_id, request) => {
                            let setup_resources = self.resources.agent_setup.remove(&agent_id);
                            let task =
                                new_connect_task(transporter, agent_id, request, setup_resources);
                            self.async_tasks.enqueue(task);
                        }
                        TransportEngineRequest::AgentTransportOp(agent_id, tx_op) => {
                            self.op_queue.submit_op(agent_id, tx_op);
                        }
                        TransportEngineRequest::AgentShutdown(agent_id) => {
                            if self.op_queue.remove_agent(&agent_id) {
                                self.resources
                                    .global_registry
                                    .transport_delegator
                                    .register_agent_shutdown(self.id);
                                self.resources.agent_connected.remove(&agent_id);
                                let reply = TransportEngineReply::AgentShutdown(agent_id);
                                log::info!("shutdown {:?}", agent_id);
                                chan.tx.send(reply).unwrap();
                            }
                        }
                    };
                }
                Err(TryRecvError::Empty) => (),
                Err(TryRecvError::Disconnected) => {
                    panic!("Proxy engines shall never shutdown")
                }
            }
        }
    }
}

impl Engine for TransportEngine {
    fn progress(&mut self) -> EngineStatus {
        self.check_proxy_requests();
        if fastrand::usize(..10) < 1 {
            self.progress_async_tasks();
        }
        self.progress_ops();
        // TODO: implement transport engine shutdown

        EngineStatus::Progressed
    }
}
