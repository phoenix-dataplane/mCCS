#![feature(strict_provenance)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

use ipc::mccs::handle::CommunicatorHandle;
use thiserror::Error;

use cuda_runtime_sys::{cudaEvent_t, cudaStream_t};
use ipc::mccs::command;
use ipc::mccs::dp;
use ipc::service::ShmService;

pub mod collectives;
pub mod communicator;
pub mod memory;

pub use collectives::{all_gather, all_reduce};
pub use communicator::{init_communicator_rank, register_stream};
pub use memory::cuda_malloc;

pub use ipc::mccs::command::{AllReduceDataType, AllReduceOpType};

const DEFAULT_MCCS_PREFIX: &str = "/tmp/mccs";
const DEFAULT_MCCS_CONTROL: &str = "control.sock";

lazy_static::lazy_static! {
    pub static ref MCCS_PREFIX: PathBuf = {
        env::var("MCCS_PREFIX").map_or_else(|_| PathBuf::from(DEFAULT_MCCS_PREFIX), |p| {
            let path = PathBuf::from(p);
            assert!(path.is_dir(), "{path:?} is not a directly");
            path
        })
    };

    pub static ref MCCS_CONTROL_SOCK: PathBuf = {
        env::var("MCCS_CONTROL")
            .map_or_else(|_| PathBuf::from(DEFAULT_MCCS_CONTROL), PathBuf::from)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _rx_recv_impl {
    ($srv:expr, $resp:path) => {
        match $srv.recv_comp()?.0 {
            Ok($resp) => Ok(()),
            Err(e) => Err(Error::Control(stringify!($resp), e)),
            otherwise => panic!("Expect {}, found {:?}", stringify!($resp), otherwise),
        }
    };
    ($srv:expr, $resp:path, $ok_block:block) => {
        #[allow(unreachable_patterns)]
        match $srv.recv_comp()?.0 {
            Ok($resp) => $ok_block,
            Err(e) => Err(Error::Control(stringify!($resp), e)),
            otherwise => panic!("Expect {}, found {:?}", stringify!($resp), otherwise),
        }
    };
    ($srv:expr, $resp:path, $inst:ident, $ok_block:block) => {
        #[allow(unreachable_patterns)]
        match $srv.recv_comp()?.0 {
            Ok($resp($inst)) => $ok_block,
            Err(e) => Err(Error::Control(stringify!($resp), e)),
            otherwise => panic!("Expect {}, found {:?}", stringify!($resp), otherwise),
        }
    };
    ($srv:expr, $resp:path, $ok_block:block, $err:ident, $err_block:block) => {
        #[allow(unreachable_patterns)]
        match $srv.recv_comp()?.0 {
            Ok($resp) => $ok_block,
            Err($err) => $err_block,
            otherwise => panic!("Expect {}, found {:?}", stringify!($resp), otherwise),
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _checked_cuda {
    ($call:expr) => {
        match $call {
            cuda_runtime_sys::cudaError::cudaSuccess => {}
            e => return Err(Error::Cuda(e)),
        }
    };
}

#[doc(hidden)]
pub(crate) use _checked_cuda as checked_cuda;
#[doc(hidden)]
pub(crate) use _rx_recv_impl as rx_recv_impl;
use ipc::mccs::command::MccsDeviceMemoryHandle;

#[derive(Debug, Clone, Copy)]
pub struct MccsCommunicatorHandle {
    pub(crate) comm_handle: CommunicatorHandle,
    pub(crate) backend_event: cudaEvent_t,
}

thread_local! {
    pub(crate) static MCCS_CTX: Context = Context::register().expect("mCCS register failed");
    pub(crate) static MCCS_STREAM_SYNC: RefCell<HashMap<cudaStream_t, cudaEvent_t>> = RefCell::new(HashMap::new());
}

pub(crate) struct Context {
    service:
        ShmService<command::Command, command::Completion, dp::WorkRequestSlot, dp::CompletionSlot>,
}

impl Context {
    fn register() -> Result<Context, Error> {
        let service = ShmService::register(&*MCCS_PREFIX, &*MCCS_CONTROL_SOCK)?;
        Ok(Self { service })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DevicePtr {
    pub ptr: *mut std::os::raw::c_void,
    backup_mem: MccsDeviceMemoryHandle,
}

impl DevicePtr {
    pub fn add(&self, size: usize) -> Result<Self, ()> {
        let new_mem = self.backup_mem.add(size)?;
        Ok(Self {
            ptr: unsafe { self.ptr.clone().add(size) },
            backup_mem: new_mem,
        })
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("Service error: {0}")]
    Service(#[from] ipc::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("mCCS control plane: {0}")]
    Control(&'static str, ipc::control::Error),
    #[error("CUDA")]
    Cuda(cuda_runtime_sys::cudaError),
}
