pub mod engine;
pub mod ops;
pub mod command;

pub struct DeviceInfo {
    pub cuda_device_idx: usize,
    pub cuda_comp_cap: u32,
}