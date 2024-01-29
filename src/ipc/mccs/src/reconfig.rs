use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelPattern {
    pub channel_id: u32,
    pub ring: Vec<usize>,
    // (send_rank, recv_rank) -> port
    pub udp_sport: Option<Vec<(usize, usize, u16)>>,
    pub net_dev: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct CommunicatorId(pub u32);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommPatternReconfig {
    pub communicator_id: CommunicatorId,
    pub channels: Vec<ChannelPattern>,
    pub ib_traffic_class: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExchangeReconfigCommand {
    CommPatternReconfig(Vec<CommPatternReconfig>),
}
