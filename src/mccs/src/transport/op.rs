use super::Protocol;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportOpState {
    Init,
    InProgress,
    Completed,
}

// NCCL supports groupping multiple ncclProxySubArgs
// inside a ncclProxyArgs
// each ncclProxySubArgs could correspond to a different connection
// (each connection is identified by channel, peer, connIndex)
// we can also have TransportSubOp, and put AgentResources in each sub op
pub struct TransportOp {
    pub num_steps: u32,

    pub slice_steps: u32,
    pub chunk_steps: u32,

    pub protocol: Protocol,

    pub state: TransportOpState,

    pub base: u64,
    pub posted: u64,
    pub receive: u64,
    pub flushed: u64,
    pub transmitted: u64,
    pub done: u64,
}

impl TransportOp {
    pub fn new(num_steps: u32, slice_steps: u32, chunk_steps: u32, protocol: Protocol) -> Self {
        Self {
            num_steps,
            slice_steps,
            chunk_steps,
            protocol,
            state: TransportOpState::Init,
            base: 0,
            posted: 0,
            receive: 0,
            flushed: 0,
            transmitted: 0,
            done: 0,
        }
    }
}
