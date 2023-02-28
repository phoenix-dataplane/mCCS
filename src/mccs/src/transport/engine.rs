use std::collections::HashMap;
use std::task::{Context, Poll};

use crossbeam::channel::{Sender, Receiver, TryRecvError};
use futures::FutureExt;
use futures::future::BoxFuture;

use crate::utils::pool::WorkPool;

use super::message::{TransportEngineRequest, TransportEngineReply};
use super::queue::TransrportOpQueue;
use super::task::TransportOp;
use super::transporter::{TransportAgentId, AnyResources, Transporter, AgentMessage};
use super::channel::ConnType;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransportEngineId {
    pub cuda_device_idx: usize,
    pub index: u32,
}

struct TransportAgent {
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
        reply: AgentMessage
    },
}

struct AsyncTask {
    agent_id: TransportAgentId,
    transporter: &'static dyn Transporter,
    task: BoxFuture<'static, AsyncTaskResult>,
}

fn new_setup_task(
    transporter: &'static dyn Transporter, 
    id: TransportAgentId, 
    request: AgentMessage
) -> AsyncTask {
    let setup = match id.peer_conn.conn_type {
        ConnType::Send => {
            transporter.agent_send_setup(id, request)
        },
        ConnType::Recv => {
            transporter.agent_recv_setup(id, request)
        },
    };
    let task = setup.map(|(resources, reply)| {
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
        ConnType::Send => {
            transporter.agent_send_connect(
                id,
                request,
                setup_resources
            )
        },
        ConnType::Recv => {
            transporter.agent_recv_connect(
                id,
                request,
                setup_resources
            )
        }
    };
    let task = connect.map(|(resources,  reply)| {
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
    
struct TrasnportEngineResources {
    agent_setup: HashMap<TransportAgentId, AnyResources>,
    agent_connected: HashMap<TransportAgentId, TransportAgent>,
    proxy_tx: Vec<Sender<TransportEngineReply>>,
    proxy_rx: Vec<Receiver<TransportEngineRequest>>,
}

impl TrasnportEngineResources {
    fn progress_op(&mut self, op: &mut TransportOp) -> bool {
        // TODO
        true
    }
    
    fn progress_async_task(&mut self, task: &mut AsyncTask) -> bool {
        let waker = futures::task::noop_waker_ref();
        let mut cx = Context::from_waker(&waker);
        let poll = task.task.as_mut().poll(&mut cx);
        match poll {
            Poll::Ready(result) => {
                match result {
                    AsyncTaskResult::Setup { 
                        setup_resources, 
                        reply 
                    } => {
                        let reply = TransportEngineReply::AgentSendSetup(
                            task.agent_id,
                            reply
                        );
                        self.proxy_tx[task.agent_id.client_cuda_dev].send(reply).unwrap();
                        self.agent_setup.insert(task.agent_id, setup_resources);
                    },
                    AsyncTaskResult::Connect { 
                        agent_resources, 
                        reply
                    } => {
                        let connected = TransportAgent {
                            transporter: task.transporter,
                            agent_resources,
                        };
                        let reply = TransportEngineReply::AgentSendConnect(
                            task.agent_id,
                            reply
                        );
                        self.proxy_tx[task.agent_id.client_cuda_dev].send(reply).unwrap();
                        self.agent_connected.insert(task.agent_id, connected);
                    },
                }
                true
            },
            Poll::Pending => false,
        }
    }
}


pub struct TransportEngine {
    id: TransportEngineId,
    resources: TrasnportEngineResources,
    async_tasks: WorkPool<AsyncTask>,
    op_queue: TransrportOpQueue,
}

impl TransportEngine {
    fn progress_ops(&mut self) {
        self.op_queue.progress_ops(|op| self.resources.progress_op(op));
    }

    fn progress_async_tasks(&mut self) {
        self.async_tasks.progress(|x| self.resources.progress_async_task(x));
    }

    fn check_proxy_requests(&mut self) {
        for rx in self.resources.proxy_rx.iter_mut() {
            match rx.try_recv() {
                Ok(request) => {
                    let task = match request {
                        TransportEngineRequest::AgentSendSetup(
                            transporter, 
                            agent_id, 
                            request,
                        ) | TransportEngineRequest::AgentRecvSetup(
                            transporter,
                            agent_id,
                            request,
                        ) => {
                            new_setup_task(transporter, agent_id, request)
                        },
                        TransportEngineRequest::AgentSendConnect(
                            transporter,
                            agent_id,
                            request,
                        ) | TransportEngineRequest::AgentRecvConnect(
                            transporter,
                            agent_id,
                            request,
                        ) => {
                            let setup_resources = self.resources.agent_setup.remove(&agent_id);
                            new_connect_task(
                                transporter,
                                agent_id,
                                request,
                                setup_resources
                            )
                        },
                    };
                    self.async_tasks.enqueue(task);
                },
                Err(TryRecvError::Empty) => (),
                Err(TryRecvError::Disconnected) => {
                    unreachable!("Proxy engines shall never shutdown")
                }
            }
        }
    }
}

impl TransportEngine {
    pub fn mainloop(&mut self) {
        self.check_proxy_requests();
        self.progress_async_tasks();
        self.progress_ops();
    }
}