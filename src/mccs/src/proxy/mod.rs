use std::net::IpAddr;

pub mod command;
pub mod engine;
pub mod init;
pub mod message;
pub mod op;
pub mod plan;
pub mod task;

pub struct DeviceInfo {
    pub host: IpAddr,
    pub listen_port: u16,
    pub cuda_device_idx: i32,
}
