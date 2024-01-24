use std::path::PathBuf;

pub use libc::pid_t;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum Error {
    #[error("{0}")]
    Generic(String),
}

type IResult<T> = Result<T, Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    // New client with device affinity,
    NewClient(Option<i32>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseKind {
    /// path of the engine's domain socket
    NewClient(PathBuf),
    /// .0: the requested scheduling mode
    /// .1: name of the OneShotServer
    /// .2: data path work queue capacity in bytes
    ConnectEngine {
        one_shot_name: String,
        wq_cap: usize,
        cq_cap: usize,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Response(pub IResult<ResponseKind>);
