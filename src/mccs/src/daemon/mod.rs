use thiserror::Error;

pub mod engine;

#[derive(Debug, Error)]
pub(crate) enum Error {
    #[error("ipc-channel TryRecvError")]
    IpcTryRecv,
    #[error("Customer error: {0}")]
    Customer(#[from] ipc::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct DaemonId(pub u32);
