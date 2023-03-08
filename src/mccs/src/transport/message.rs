use super::transporter::{TransportAgentId, AgentMessage, Transporter};

pub enum TransportEngineRequest {
    AgentSetup(&'static dyn Transporter, TransportAgentId, AgentMessage),
    AgentConnect(&'static dyn Transporter, TransportAgentId, AgentMessage),
}

pub enum TransportEngineReply {
    AgentSetup(TransportAgentId, AgentMessage),
    AgentConnect(TransportAgentId, AgentMessage),
}