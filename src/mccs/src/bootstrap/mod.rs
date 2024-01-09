pub mod task;

use std::net::SocketAddr;

use serde::{Deserialize, Serialize};
use smol::lock::Mutex;
use smol::net::{TcpListener, TcpStream};
use thiserror::Error;

pub use task::{bootstrap_create_root, bootstrap_root};

#[derive(Debug, Error)]
pub enum BootstrapError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Bootstrap root received inconsistent rank count of {0} vs {1}")]
    NumRanksMismatch(usize, usize),
    #[error("Bootstrap root received duplicate check-in from rank {0}")]
    DuplicatedCheckIn(usize),
    #[error("Bootstrap root received incorrect rank number {0}")]
    RankOverflow(usize),
    #[error("Received {0} bytes instead of {1} bytes")]
    RecvSizeMismatch(u32, u32),
    #[error(
        "Could not acquire Mutex in bootstrap state, only a single outstanding task is allowed"
    )]
    MutexAcquire,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapHandle {
    pub addr: SocketAddr,
    pub magic: u64,
}

pub struct UnexpectedConn {
    pub stream: TcpStream,
    pub peer: usize,
    pub tag: u32,
}

pub struct BootstrapRing {
    pub ring_recv: TcpStream,
    pub ring_send: TcpStream,
}

pub struct BootstrapState {
    pub listener: TcpListener,
    pub ring: Mutex<BootstrapRing>,
    pub peer_addrs: Vec<SocketAddr>,
    // Mutex is not necessary as proxy engine will ensure that
    // only a single outstanding recv task will access this field
    pub unexpected_connections: Mutex<Vec<UnexpectedConn>>,
    pub rank: usize,
    pub num_ranks: usize,
    pub magic: u64,
}
