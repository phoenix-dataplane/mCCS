#![feature(strict_provenance)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::ffi::CString;
use std::path::PathBuf;

use ipc::mccs::handle::CommunicatorHandle;
use thiserror::Error;

use cuda_runtime_sys::cudaDeviceGetPCIBusId;
use cuda_runtime_sys::{cudaEvent_t, cudaStream_t};
use ipc::mccs::command;
use ipc::mccs::dp;
use ipc::service::ShmService;
use nvml_sys::{nvmlDeviceGetCpuAffinity, nvmlDeviceGetHandleByPciBusId_v2};

pub mod collectives;
pub mod communicator;
pub mod memory;

pub use collectives::{all_gather, all_reduce};
pub use communicator::{init_communicator_rank, register_stream};
pub use memory::cuda_malloc;

pub use ipc::mccs::command::{AllReduceDataType, AllReduceOpType};

const DEFAULT_MCCS_PREFIX: &str = "/tmp/mccs-${USER}";
const DEFAULT_MCCS_CONTROL: &str = "control.sock";

macro_rules! nvml_warning {
    ($nvml_op:expr) => {{
        let e = $nvml_op;
        if e != nvml_sys::nvmlReturn_enum::NVML_SUCCESS {
            panic!("NVML failed with {:?} at {}:{}.", e, file!(), line!())
        }
    }};
}

lazy_static::lazy_static! {
    pub static ref MCCS_PREFIX: PathBuf = {
        env::var("MCCS_PREFIX").map_or_else(|_| match std::env::var("USER") {
            Ok(user) => PathBuf::from(DEFAULT_MCCS_PREFIX.replace("${USER}", &user)),
            Err(_) => PathBuf::from(DEFAULT_MCCS_PREFIX),
        }, |p| {
            let path = PathBuf::from(p);
            assert!(path.is_dir(), "{path:?} is not a directly");
            path
        })
    };

    pub static ref MCCS_CONTROL_SOCK: PathBuf = {
        env::var("MCCS_CONTROL")
            .map_or_else(|_| PathBuf::from(DEFAULT_MCCS_CONTROL), PathBuf::from)
    };

    pub static ref MCCS_DEVICE_AFFINITY: i32 = {
        let affinity = env::var("MCCS_DEVICE_AFFINITY")
            .map_or_else(|_| -1, |s| s.parse().expect("MCCS_DEVICE_AFFINITY is not a number"));
        if affinity >= 0 {
            unsafe {
                nvml_warning!(nvml_sys::nvmlInit_v2());
                let bus_id = CString::new(b"00000000:00:00.0").unwrap();
                let raw_bus_id = bus_id.as_c_str();
                // including the null terminator
                let len = raw_bus_id.to_bytes().len() + 1;
                let cpu_set = {
                    cudaDeviceGetPCIBusId(raw_bus_id.as_ptr() as *mut _, len as i32, affinity);
                    let mut handle = std::ptr::null_mut();
                    nvml_warning!(nvmlDeviceGetHandleByPciBusId_v2(raw_bus_id.as_ptr() as *mut _, &mut handle));
                    let mut cpu_set = 0u64;
                    nvml_warning!(nvmlDeviceGetCpuAffinity(handle, 1, &mut cpu_set));
                    cpu_set
                };

                use libnuma::masks::indices::CpuIndex;
                use libnuma::masks::{Mask, CpuMask};
                let cpu_mask = CpuMask::allocate();
                for i in 0..64 {
                    if cpu_set & (1 << i) != 0 {
                        cpu_mask.set(CpuIndex::new(i as _));
                    }
                }
                println!("Setting CPU affinity to {:#066b}", cpu_set);
                if !cpu_mask.sched_set_affinity_for_current_thread() {
                    panic!("Failed to set CPU affinity for current thread");
                }
            }
        }
        affinity
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
        let device_affnity = match *MCCS_DEVICE_AFFINITY {
            -1 => None,
            idx => Some(idx),
        };
        println!(
            "register ctx: prefix={:?}, control={:?}, affinity={:?}",
            *MCCS_PREFIX, *MCCS_CONTROL_SOCK, device_affnity
        );
        let service = ShmService::register(&*MCCS_PREFIX, &*MCCS_CONTROL_SOCK, device_affnity)?;
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
