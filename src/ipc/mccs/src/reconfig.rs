use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelPattern {
    pub channel_id: u32,
    pub ring: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct CommunicatorId(pub u32);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommPatternReconfig {
    pub communicator_id: CommunicatorId,
    pub channels: Vec<ChannelPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExchangeReconfigCommand {
    CommPatternReconfig(CommPatternReconfig),
}
