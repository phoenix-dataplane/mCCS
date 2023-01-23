use crate::transport::NUM_PROTOCOLS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemLocality {
    SenderSide,
    ReceiverSide,
}

#[derive(Clone, Debug)]
pub struct HostMemTptConfig {
    pub buff_sizes: [usize; NUM_PROTOCOLS],
    pub locality: MemLocality,
}