use super::{op::TransportOp, transporter::TransportAgentId};

pub struct TransportTask {
    pub(crate) agent_id: TransportAgentId,
    pub(crate) op: TransportOp,
}
