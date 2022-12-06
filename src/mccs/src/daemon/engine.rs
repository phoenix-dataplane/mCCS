use std::os::raw::c_void;

use cuda_runtime_sys::{cudaMalloc, cudaIpcGetMemHandle};
use cuda_runtime_sys::cudaError;
use cuda_runtime_sys::cudaIpcMemHandle_t;

use ipc::customer::ShmCustomer;
use ipc::mccs::command;
use ipc::mccs::dp;
use ipc::mccs::handle::CudaMemHandle;

use super::Error;

pub type CustomerType =
    ShmCustomer<command::Command, command::Completion, dp::WorkRequestSlot, dp::CompletionSlot>;

pub struct DaemonEngine {
    pub(crate) customer: CustomerType
}

impl DaemonEngine {
    pub fn mainloop(&mut self) {
        loop {
            self.check_cmd().unwrap();
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
            Command::CudaMalloc(size) => {
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
                let return_handle = CudaMemHandle(handle.reserved);
                Ok(Some(CompletionKind::CudaMalloc(return_handle)))
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
                    Err(_e) => todo!(),
                }
                Ok(Progress(1))
            }
            Err(ipc::TryRecvError::Empty) => {
                Ok(Progress(0))
            }
            Err(ipc::TryRecvError::Disconnected) => Ok(Status::Disconnected),
            Err(ipc::TryRecvError::Other(_e)) => Err(Error::IpcTryRecv),
        }
    }
}