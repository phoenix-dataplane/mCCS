use std::env;
use std::path::PathBuf;

use thiserror::Error;

use ipc::service::ShmService;
use ipc::mccs::command;
use ipc::mccs::dp;

pub mod memory;

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
        #[allow(unreachable_patterns)]
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
pub(crate) use _rx_recv_impl as rx_recv_impl;


thread_local! {
    pub(crate) static MCCS_CTX: Context = Context::register().expect("mCCS register failed");
}

pub(crate) struct Context {
    service: ShmService<command::Command, command::Completion, dp::WorkRequestSlot, dp::CompletionSlot>,
}

impl Context {
    fn register() -> Result<Context, Error> {
        let service = ShmService::register(
            &*MCCS_PREFIX,
            &*MCCS_CONTROL_SOCK,
        )?;
        Ok(Self { service })
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
    Cuda(cuda_runtime_sys::cudaError)
}
