use std::any::Any;

pub type AnyNetComm = Box<dyn Any + Send>;

pub struct NetProperties {
    name: String,
    pci_path: String,
    guid: u64,
    
    speed: u32,
    port: u16,
    latency: f32,
    max_comms: usize,
    max_recvs: usize,
}


pub trait NetProvider {
    type NetHandle;


    fn init() -> Self;
    fn get_num_devices() -> usize;

}