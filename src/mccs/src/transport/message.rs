use super::transporter::{TransportAgentId, AgentMessage, Transporter};

pub enum TransportEngineRequest {
    AgentSendSetup(&'static dyn Transporter, TransportAgentId, AgentMessage),
    AgentSendConnect(&'static dyn Transporter, TransportAgentId, AgentMessage),
    AgentRecvSetup(&'static dyn Transporter, TransportAgentId, AgentMessage),
    AgentRecvConnect(&'static dyn Transporter, TransportAgentId, AgentMessage),
}

pub enum TransportEngineReply {
    AgentSendSetup(TransportAgentId, AgentMessage),
    AgentSendConnect(TransportAgentId, AgentMessage),
    AgentRecvSetup(TransportAgentId, AgentMessage),
    AgentRecvConnect(TransportAgentId, AgentMessage),
}