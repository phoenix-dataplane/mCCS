use crate::daemon::DaemonId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerType {
    Local,
    IntraNode,
    InterNode,
}

#[derive(Clone)]
pub struct PeerInfo {
    pub peer_type: PeerType,
    pub rank: usize,
    pub cuda_device_idx: usize,
    pub cuda_comp_cap: u32,
}

#[derive(Clone)]
pub struct RankInfo {
    pub rank: usize,
    // TBD
    pub host: u64, 
    pub cuda_device_idx: usize,
    pub cuda_comp_cap: u32,
}

// used for bootstraping
// perhaps more
pub struct CommunicatorGlobalInfo {
    pub communicator_id: u32,
    pub num_ranks: usize,
    pub ranks_info: Vec<Option<RankInfo>>,
}


pub struct LocalCommunicator {
    pub communicator_id: u32,
    pub daemon_id: DaemonId,

    pub rank: usize,
    pub n_ranks: usize,
    pub cuda_device_idx: usize,

    pub local_rank: usize,
    pub local_ranks: usize,
    
    pub channels: Vec<CommChannel>,
    pub peers_info: Vec<PeerInfo>,
}

pub struct CommChannel {
    // struct ncclChannelPeer* peers;
    // struct ncclDevChannelPeer* devPeers;
    // int* devRingUserRanks;
    // struct ncclTree tree;
    // struct ncclTree collnetChain;
    // struct ncclDirect collnetDirect;
    // uint32_t workFifoSent;
    // uint64_t p2pOpCount;
    pub id: u64,
    pub ring: RingPattern,
    
}

pub struct RingPattern {
    pub prev_rank: usize,
    pub next_rank: usize,
}

// Following are from MSCCL, which allows custom collective allgorithm
// https://github.com/microsoft/msccl/blob/58b5006ded3983c17c317ba9e330e0b19e2325f1/src/include/msccl.h
#[allow(unused)]
pub struct CustomCollectiveAlgorithm {
    // placeholder, TBD
    send_schedule: Vec<Vec<u32>>,
    // placeholder, TBD
    recv_schedule: Vec<Vec<u32>>,
    channels_info: Vec<CustomCollectiveChannelInfo>,
    thread_blocks: Vec<CustomCollectiveThreadBlock>,
}

#[allow(unused)]
pub struct CustomCollectiveChannelInfo {
    channel_id: u32,
    send_peers_info: Vec<CustomCollectiveChannelPeerInfo>,
    recv_peers_info: Vec<CustomCollectiveChannelPeerInfo>,
}

#[allow(unused)]
pub struct CustomCollectiveChannelPeerInfo {
    peer: u32,
    n_chunks_for_peer: Vec<u32>,
    n_count_exists: u32,
    counts: Vec<u32>,
}

#[allow(unused)]
pub struct CustomCollectiveThreadBlock {
    send_peer: i16,
    recv_peer: i16,
    n_steps: u16,
    channel_id: i8,
    dependency_bit: Vec<i8>,
    dependency_step: Vec<i16>,
    reduction_src_offsets: Vec<i16>,
    transfers: Vec<CustomCollectiveTransferSchedule>,
    pad: i64,
}

#[allow(unused)]
pub struct CustomCollectiveTransferSchedule {
    src_offset: i16,
    dst_offset: i16,
    src_buffer: u8,
    dst_buffer: u8,
    dependency_pointer: i16,
    num_dependencies: i8,
    has_dependency: bool,
    num_reductions: i16,
    reduction_pointer: i64,
    ty: u8,
    count: u8,
}