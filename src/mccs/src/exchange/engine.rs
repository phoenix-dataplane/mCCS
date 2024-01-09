use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::task::{Context, Poll};

use byteorder::{ByteOrder, LittleEndian};
use crossbeam::channel::{Receiver, Sender, TryRecvError};
use futures::future::BoxFuture;
use futures::FutureExt;
use smol::io::{AsyncReadExt, AsyncWriteExt};
use smol::net::TcpListener;

use super::command::{ExchangeCommand, ExchangeCompletion};
use super::message::ExchangeMessage;
use super::ExchangeError;
use crate::bootstrap::BootstrapHandle;
use crate::comm::CommunicatorId;
use crate::utils::pool::WorkPool;
use crate::utils::tcp;

struct CommunicatorInfo {
    bootstrap_handle: BootstrapHandle,
}

enum OutstandingRequest {
    BootstrapHandleSend((CommunicatorId, SocketAddr)),
    BootstrapHandleRecv((CommunicatorId, i32)),
}

const EXCHANGE_MAGIC: u64 = 0x424ab9f2fc4b9d6e;

struct ExchangeEngineResources {
    listener: Arc<TcpListener>,
    proxy_tx: Vec<Sender<ExchangeCompletion>>,
    proxy_rx: Vec<Receiver<ExchangeCommand>>,
    comm_info: HashMap<CommunicatorId, CommunicatorInfo>,
    outstanding_requests: Vec<OutstandingRequest>,
    submit_pool: Vec<AsyncTask>,
}

async fn send_message(msg: ExchangeMessage, addr: SocketAddr) -> Result<(), ExchangeError> {
    let encode = bincode::serialize(&msg).unwrap();
    let mut stream = tcp::async_connect(&addr, EXCHANGE_MAGIC).await?;
    log::trace!("Exchange engine send {:?} to {:?}", msg, addr);
    let mut buf = [0u8; 4];
    LittleEndian::write_u32(&mut buf, encode.len() as u32);
    stream.write_all(&buf).await?;
    stream.write_all(encode.as_slice()).await?;
    Ok(())
}

async fn recv_message(listener: Arc<TcpListener>) -> Result<ExchangeMessage, ExchangeError> {
    let mut stream = tcp::async_accept(&listener, EXCHANGE_MAGIC).await?;
    let mut buf = [0u8; 4];
    stream.read_exact(&mut buf).await?;
    let len = LittleEndian::read_u32(&buf);
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await?;
    let decode = bincode::deserialize(buf.as_slice()).unwrap();
    let remote_addr = stream.peer_addr()?;
    log::trace!("Exchange engine recv {:?} from {:?}", decode, remote_addr);
    Ok(decode)
}

enum AsyncTaskOutput {
    Send,
    Recv(ExchangeMessage),
}

type AsyncTask = BoxFuture<'static, Result<AsyncTaskOutput, ExchangeError>>;

impl ExchangeEngineResources {
    fn progress_async_task(&mut self, task: &mut AsyncTask) -> bool {
        let waker = futures::task::noop_waker_ref();
        let mut cx = Context::from_waker(waker);
        let poll = task.as_mut().poll(&mut cx);
        match poll {
            Poll::Ready(result) => {
                let output = result.unwrap();
                let mut new_recv = false;
                let additional_task = match output {
                    AsyncTaskOutput::Send => None,
                    AsyncTaskOutput::Recv(msg) => {
                        new_recv = true;
                        match msg {
                            ExchangeMessage::BootstrapHandle(comm_id, handle) => {
                                let requests =
                                    self.outstanding_requests.drain_filter(|x| match x {
                                        OutstandingRequest::BootstrapHandleRecv((id, _)) => {
                                            id == &comm_id
                                        }
                                        _ => false,
                                    });
                                for req in requests {
                                    match req {
                                        OutstandingRequest::BootstrapHandleRecv((id, cuda_dev)) => {
                                            let reply = ExchangeCompletion::RecvBootstrapHandle(
                                                id,
                                                handle.clone(),
                                            );
                                            self.proxy_tx[cuda_dev as usize].send(reply).unwrap();
                                        }
                                        _ => unreachable!(),
                                    }
                                }
                                let info = CommunicatorInfo {
                                    bootstrap_handle: handle,
                                };
                                self.comm_info.insert(comm_id, info);
                                None
                            }
                            ExchangeMessage::BootstrapHandleRequest(comm_id, reply_addr) => {
                                if let Some(info) = self.comm_info.get(&comm_id) {
                                    let msg = ExchangeMessage::BootstrapHandle(
                                        comm_id,
                                        info.bootstrap_handle.clone(),
                                    );
                                    let task = send_message(msg, reply_addr)
                                        .map(|x| x.map(|_| AsyncTaskOutput::Send));
                                    let fut = Box::pin(task) as AsyncTask;
                                    Some(fut)
                                } else {
                                    let request = OutstandingRequest::BootstrapHandleSend((
                                        comm_id, reply_addr,
                                    ));
                                    self.outstanding_requests.push(request);
                                    None
                                }
                            }
                        }
                    }
                };
                let recv_task = if new_recv {
                    let task = recv_message(self.listener.clone())
                        .map(|x| x.map(|msg| AsyncTaskOutput::Recv(msg)));
                    let fut = Box::pin(task) as AsyncTask;
                    Some(fut)
                } else {
                    None
                };
                if let Some(task) = additional_task {
                    self.submit_pool.push(task);
                }
                if let Some(task) = recv_task {
                    self.submit_pool.push(task);
                }
                true
            }
            Poll::Pending => false,
        }
    }
}

pub struct ExchangeEngine {
    resources: ExchangeEngineResources,
    async_tasks: WorkPool<AsyncTask>,
}

impl ExchangeEngine {
    fn progress_async_tasks(&mut self) {
        self.async_tasks
            .progress(|x| self.resources.progress_async_task(x));
        for task in self.resources.submit_pool.drain(..) {
            self.async_tasks.enqueue(task);
        }
    }

    fn check_proxy_requests(&mut self) {
        for (cuda_dev, rx) in self.resources.proxy_rx.iter().enumerate() {
            match rx.try_recv() {
                Ok(cmd) => match cmd {
                    ExchangeCommand::RegisterBootstrapHandle(comm_id, handle) => {
                        let requests =
                            self.resources
                                .outstanding_requests
                                .drain_filter(|x| match x {
                                    OutstandingRequest::BootstrapHandleSend((id, _)) => {
                                        id == &comm_id
                                    }
                                    _ => false,
                                });
                        for req in requests {
                            match req {
                                OutstandingRequest::BootstrapHandleSend((comm_id, addr)) => {
                                    let msg =
                                        ExchangeMessage::BootstrapHandle(comm_id, handle.clone());
                                    let task = send_message(msg, addr)
                                        .map(|x| x.map(|_| AsyncTaskOutput::Send));
                                    let fut = Box::pin(task);
                                    self.async_tasks.enqueue(fut);
                                }
                                _ => unreachable!(),
                            }
                        }
                        let info = CommunicatorInfo {
                            bootstrap_handle: handle,
                        };
                        self.resources.comm_info.insert(comm_id, info);
                    }
                    ExchangeCommand::RecvBootstrapHandle(comm_id, root_addr) => {
                        if let Some(info) = self.resources.comm_info.get(&comm_id) {
                            let msg = ExchangeCompletion::RecvBootstrapHandle(
                                comm_id,
                                info.bootstrap_handle.clone(),
                            );
                            self.resources.proxy_tx[cuda_dev].send(msg).unwrap();
                        } else {
                            let reply_addr = self.resources.listener.local_addr().unwrap();
                            let msg = ExchangeMessage::BootstrapHandleRequest(comm_id, reply_addr);
                            let task = send_message(msg, root_addr)
                                .map(|x| x.map(|_| AsyncTaskOutput::Send));
                            let fut = Box::pin(task);
                            self.async_tasks.enqueue(fut);
                            self.resources.outstanding_requests.push(
                                OutstandingRequest::BootstrapHandleRecv((comm_id, cuda_dev as i32)),
                            );
                        }
                    }
                },
                Err(TryRecvError::Empty) => (),
                Err(TryRecvError::Disconnected) => {
                    unreachable!("Proxy engines shall never shutdown")
                }
            }
        }
    }
}

impl ExchangeEngine {
    pub fn new(
        listen_addr: SocketAddr,
        proxy_tx: Vec<Sender<ExchangeCompletion>>,
        proxy_rx: Vec<Receiver<ExchangeCommand>>,
    ) -> Self {
        if listen_addr.port() == 0 {
            panic!("Listen port must be specified");
        }
        let listener = tcp::async_listen(&listen_addr).unwrap();
        let resources = ExchangeEngineResources {
            listener: Arc::new(listener),
            proxy_tx,
            proxy_rx,
            comm_info: HashMap::new(),
            outstanding_requests: Vec::new(),
            submit_pool: Vec::new(),
        };
        let mut work_pool = WorkPool::new();
        let task = recv_message(resources.listener.clone())
            .map(|x| x.map(|msg| AsyncTaskOutput::Recv(msg)));
        let fut = Box::pin(task) as AsyncTask;
        work_pool.enqueue(fut);
        ExchangeEngine {
            resources,
            async_tasks: work_pool,
        }
    }

    pub fn mainloop(&mut self) {
        loop {
            self.check_proxy_requests();
            self.progress_async_tasks();
        }
    }
}