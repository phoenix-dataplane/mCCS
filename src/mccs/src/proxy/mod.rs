use std::net::SocketAddr;

pub mod command;
pub mod engine;
pub mod init;
pub mod message;
pub mod op;
pub mod plan;
pub mod task;


pub struct DeviceInfo {
    pub host: SocketAddr,
    pub cuda_device_idx: i32,
}
