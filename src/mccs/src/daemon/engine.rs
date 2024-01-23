use std::collections::HashMap;
use std::os::raw::c_void;

use cuda_runtime_sys::cudaError;
use cuda_runtime_sys::cudaFree;
use cuda_runtime_sys::cudaIpcMemHandle_t;
use cuda_runtime_sys::{cudaIpcGetMemHandle, cudaMalloc, cudaSetDevice};

use ipc::customer::ShmCustomer;
use ipc::mccs::command;
use ipc::mccs::command::MccsDeviceMemoryHandle;
use ipc::mccs::dp;
use ipc::mccs::handle::{CommunicatorHandle, CudaMemHandle};

use super::{DaemonId, Error};
use crate::comm::CommunicatorId;
use crate::cuda_warning;
use crate::engine::{Engine, EngineStatus};
use crate::proxy::command::AllReduceRequest;
use crate::proxy::command::{
    AllGatherRequest, CollRequest, InitCommunicator, ProxyCommand, ProxyCompletion,
};
use crate::utils::duplex_chan::DuplexChannel;

pub type CustomerType =
    ShmCustomer<command::Command, command::Completion, dp::WorkRequestSlot, dp::CompletionSlot>;

pub struct CommunicatorDelegation {
    pub comm_id: u32,
    pub cuda_device_idx: i32,
}

pub(crate) struct DeviceMemory {
    addr: usize,
    device_idx: i32,
}

pub struct DaemonEngine {
    pub(crate) id: DaemonId,
    pub(crate) customer: CustomerType,
    pub(crate) proxy_chan: Vec<DuplexChannel<ProxyCommand, ProxyCompletion>>,
    pub(crate) device_mem: HashMap<u64, DeviceMemory>,
    pub(crate) comm_delegation: HashMap<CommunicatorHandle, CommunicatorDelegation>,
    pub(crate) mem_counter: u64,
}

impl DaemonEngine {
    pub fn new(
        id: DaemonId,
        customer: CustomerType,
        proxy_chan: Vec<DuplexChannel<ProxyCommand, ProxyCompletion>>,
    ) -> Self {
        DaemonEngine {
            id,
            customer,
            proxy_chan,
            device_mem: HashMap::new(),
            comm_delegation: HashMap::new(),
            mem_counter: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Progress(usize),
    Disconnected,
}

use Status::Progress;

impl DaemonEngine {
    fn process_cmd(
        &mut self,
        req: &command::Command,
    ) -> Result<Option<command::CompletionKind>, Error> {
        use ipc::mccs::command::{Command, CompletionKind};
        match req {
            Command::CudaMalloc(dev_idx, size) => {
                unsafe {
                    let error = cudaSetDevice(*dev_idx as _);
                    if error != cudaError::cudaSuccess {
                        panic!("cudaSetDevice");
                    }
                }
                let mut dev_ptr: *mut c_void = std::ptr::null_mut();
                let err = unsafe { cudaMalloc(&mut dev_ptr as *mut _, *size) };
                if err != cudaError::cudaSuccess {
                    panic!("cudaMalloc failed")
                }
                let mut handle = cudaIpcMemHandle_t::default();
                let err = unsafe { cudaIpcGetMemHandle(&mut handle as *mut _, dev_ptr) };
                if err != cudaError::cudaSuccess {
                    panic!("cudaIpcGetMemHandle failed")
                }
                let this_cnt = self.mem_counter;
                self.mem_counter += 1;
                self.device_mem.insert(
                    0,
                    DeviceMemory {
                        addr: dev_ptr as usize,
                        device_idx: *dev_idx,
                    },
                );
                let return_handle = CudaMemHandle(handle.reserved);
                let return_mem = MccsDeviceMemoryHandle {
                    id: this_cnt,
                    offset: 0,
                    len: *size,
                };
                log::debug!(
                    "[Daemon-{}] cudaMalloc {} bytes at {:p} on GPU {}",
                    self.id.0,
                    size,
                    dev_ptr,
                    dev_idx
                );
                Ok(Some(CompletionKind::CudaMalloc((
                    return_handle,
                    return_mem,
                ))))
            }
            Command::InitCommunicator(init) => {
                log::debug!(
                    "[Daemon-{}] initCommunicator {} ({}/{}) on GPU {}",
                    self.id.0,
                    init.id,
                    init.rank,
                    init.num_ranks,
                    init.cuda_device_idx
                );
                let proxy_init = InitCommunicator {
                    communicator_id: CommunicatorId(init.id),
                    rank: init.rank,
                    num_ranks: init.num_ranks,
                    root_mccs_addr: init.root_addr.clone(),
                };
                let proxy_cmd = ProxyCommand::InitCommunicator(proxy_init);
                self.proxy_chan[init.cuda_device_idx as usize]
                    .tx
                    .send(proxy_cmd)
                    .unwrap();
                let res = self.proxy_chan[init.cuda_device_idx as usize]
                    .rx
                    .recv()
                    .unwrap();
                let event_handle = match res {
                    ProxyCompletion::InitCommunicator(event_handle) => event_handle,
                    _ => panic!("unexpected result"),
                };
                let comm_handle = CommunicatorHandle(((init.id as u64) << 32) + init.rank as u64);
                let comm = CommunicatorDelegation {
                    comm_id: init.id,
                    cuda_device_idx: init.cuda_device_idx,
                };
                self.comm_delegation.insert(comm_handle, comm);

                Ok(Some(CompletionKind::InitCommunicator((
                    comm_handle,
                    event_handle,
                ))))
            }
            Command::AllGather(all_gather) => {
                // prepare arguments
                let comm = self.comm_delegation.get(&all_gather.comm).unwrap();
                let send_buf_addr = (*self.device_mem.get(&all_gather.send_buf.id).unwrap()).addr
                    + all_gather.send_buf.offset;
                let recv_buf_addr = (*self.device_mem.get(&all_gather.recv_buf.id).unwrap()).addr
                    + all_gather.recv_buf.offset;
                let proxy_all_gather = AllGatherRequest {
                    communicator_id: CommunicatorId(comm.comm_id),
                    send_buf_addr,
                    recv_buf_addr,
                    size: all_gather.size,
                    user_stream: all_gather.user_stream,
                };
                log::debug!(
                    "[Daemon-{}] try to issue allGather ({:p},{:p}) on communicator {}@{}",
                    self.id.0,
                    send_buf_addr as *const c_void,
                    recv_buf_addr as *const c_void,
                    comm.cuda_device_idx,
                    comm.comm_id,
                );
                // send command
                let proxy_cmd = ProxyCommand::AllGather(proxy_all_gather);
                self.proxy_chan[comm.cuda_device_idx as usize]
                    .tx
                    .send(proxy_cmd)
                    .unwrap();
                let res = self.proxy_chan[comm.cuda_device_idx as usize]
                    .rx
                    .recv()
                    .unwrap();
                match res {
                    ProxyCompletion::AllGather => {}
                    _ => panic!("unexpected result"),
                };
                log::debug!(
                    "[Daemon-{}] SUCCESS for issuing allGather on communicator {}@{}",
                    self.id.0,
                    comm.cuda_device_idx,
                    comm.comm_id,
                );
                Ok(Some(CompletionKind::AllGather))
            }
            Command::AllReduce(all_reduce) => {
                // prepare arguments
                let comm = self.comm_delegation.get(&all_reduce.comm).unwrap();
                let send_buf_addr = (*self.device_mem.get(&all_reduce.send_buf.id).unwrap()).addr
                    + all_reduce.send_buf.offset;
                let recv_buf_addr = (*self.device_mem.get(&all_reduce.recv_buf.id).unwrap()).addr
                    + all_reduce.recv_buf.offset;
                let proxy_all_reduce = AllReduceRequest {
                    communicator_id: CommunicatorId(comm.comm_id),
                    send_buf_addr,
                    recv_buf_addr,
                    size: all_reduce.size,
                    data_type: all_reduce.data_type.into(),
                    op_type: all_reduce.op_type.into(),
                    user_stream: all_reduce.user_stream,
                };
                log::debug!(
                    "[Daemon-{}] try to issue allReduce ({:p},{:p}) on communicator {}@{}",
                    self.id.0,
                    send_buf_addr as *const c_void,
                    recv_buf_addr as *const c_void,
                    comm.cuda_device_idx,
                    comm.comm_id,
                );
                // send command
                let proxy_cmd = ProxyCommand::AllReduce(proxy_all_reduce);
                self.proxy_chan[comm.cuda_device_idx as usize]
                    .tx
                    .send(proxy_cmd)
                    .unwrap();
                let res = self.proxy_chan[comm.cuda_device_idx as usize]
                    .rx
                    .recv()
                    .unwrap();
                match res {
                    ProxyCompletion::AllReduce => {}
                    _ => panic!("unexpected result"),
                };
                log::debug!(
                    "[Daemon-{}] SUCCESS for issuing allReduce on communicator {}@{}",
                    self.id.0,
                    comm.cuda_device_idx,
                    comm.comm_id,
                );
                Ok(Some(CompletionKind::AllReduce))
            }
            Command::GroupCall(colls) => {
                let mut requests = Vec::with_capacity(colls.len());
                let comm_handle = match &colls[0] {
                    command::CollOperation::AllGather(all_gather) => all_gather.comm,
                    command::CollOperation::AllReduce(all_reduce) => all_reduce.comm,
                };
                let comm = self.comm_delegation.get(&comm_handle).unwrap();
                for coll in colls.iter() {
                    match coll {
                        command::CollOperation::AllGather(all_gather) => {
                            // prepare arguments
                            let send_buf_addr =
                                (*self.device_mem.get(&all_gather.send_buf.id).unwrap()).addr
                                    + all_gather.send_buf.offset;
                            let recv_buf_addr =
                                (*self.device_mem.get(&all_gather.recv_buf.id).unwrap()).addr
                                    + all_gather.recv_buf.offset;
                            let proxy_all_gather = AllGatherRequest {
                                communicator_id: CommunicatorId(comm.comm_id),
                                send_buf_addr,
                                recv_buf_addr,
                                size: all_gather.size,
                                user_stream: all_gather.user_stream,
                            };
                            let coll_op = CollRequest::AllGather(proxy_all_gather);
                            requests.push(coll_op);
                        }
                        command::CollOperation::AllReduce(all_reduce) => {
                            // prepare arguments
                            let send_buf_addr =
                                (*self.device_mem.get(&all_reduce.send_buf.id).unwrap()).addr
                                    + all_reduce.send_buf.offset;
                            let recv_buf_addr =
                                (*self.device_mem.get(&all_reduce.recv_buf.id).unwrap()).addr
                                    + all_reduce.recv_buf.offset;
                            let proxy_all_reduce = AllReduceRequest {
                                communicator_id: CommunicatorId(comm.comm_id),
                                send_buf_addr,
                                recv_buf_addr,
                                size: all_reduce.size,
                                data_type: all_reduce.data_type.into(),
                                op_type: all_reduce.op_type.into(),
                                user_stream: all_reduce.user_stream,
                            };
                            let coll_op = CollRequest::AllReduce(proxy_all_reduce);
                            requests.push(coll_op);
                        }
                    }
                }
                let proxy_cmd = ProxyCommand::GroupCall(requests);
                self.proxy_chan[comm.cuda_device_idx as usize]
                    .tx
                    .send(proxy_cmd)
                    .unwrap();
                let res = self.proxy_chan[comm.cuda_device_idx as usize]
                    .rx
                    .recv()
                    .unwrap();
                match res {
                    ProxyCompletion::GroupCall => {}
                    _ => panic!("unexpected result"),
                };
                Ok(Some(CompletionKind::GroupCall))
            }
            Command::RegisterStream(cuda_dev, stream, event_handle) => {
                let proxy_cmd = ProxyCommand::RegisterStream(*stream, event_handle.clone());
                self.proxy_chan[*cuda_dev as usize]
                    .tx
                    .send(proxy_cmd)
                    .unwrap();
                let res = self.proxy_chan[*cuda_dev as usize].rx.recv().unwrap();
                match res {
                    ProxyCompletion::RegisterStream => {}
                    _ => panic!("unexpected result"),
                };
                Ok(Some(CompletionKind::RegisterStream))
            }
        }
    }

    fn check_cmd(&mut self) -> Result<Status, Error> {
        match self.customer.try_recv_cmd() {
            Ok(req) => {
                let result = self.process_cmd(&req);
                match result {
                    Ok(Some(res)) => self.customer.send_comp(command::Completion(Ok(res)))?,
                    Ok(None) => return Ok(Progress(0)),
                    Err(_e) => panic!(),
                }
                log::trace!("Send completion back successfully");
                Ok(Progress(1))
            }
            Err(ipc::TryRecvError::Empty) => Ok(Progress(0)),
            Err(ipc::TryRecvError::Disconnected) => Ok(Status::Disconnected),
            Err(ipc::TryRecvError::Other(_e)) => Err(Error::IpcTryRecv),
        }
    }
}

impl Engine for DaemonEngine {
    fn progress(&mut self) -> EngineStatus {
        let status = self.check_cmd().unwrap();
        match status {
            Progress(x) => {
                if x != 0 {
                    EngineStatus::Progressed
                } else {
                    EngineStatus::Idle
                }
            }
            Status::Disconnected => EngineStatus::Completed,
        }
    }
}

impl Drop for DaemonEngine {
    fn drop(&mut self) {
        for mem in self.device_mem.drain().map(|(_, v)| v) {
            cuda_warning!(unsafe { cudaSetDevice(mem.device_idx) });
            cuda_warning!(unsafe { cudaFree(mem.addr as *mut c_void) });
        }
    }
}
