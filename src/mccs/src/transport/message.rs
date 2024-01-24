use super::{
    op::TransportOp,
    transporter::{AgentMessage, TransportAgentId, Transporter},
};

pub enum TransportEngineRequest {
    AgentSetup(&'static dyn Transporter, TransportAgentId, AgentMessage),
    AgentConnect(&'static dyn Transporter, TransportAgentId, AgentMessage),
    AgentTransportOp(TransportAgentId, TransportOp),
    AgentShutdown(TransportAgentId),
}

pub enum TransportEngineReply {
    AgentSetup(TransportAgentId, AgentMessage),
    AgentConnect(TransportAgentId, AgentMessage),
    AgentShutdown(TransportAgentId),
}
