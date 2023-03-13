use crate::{comm::CommunicatorId, daemon::DaemonId};


pub enum ProxyOp {
    InitCommunicator(DaemonId, CommunicatorId),
    PollCudaEvent(DaemonId, CommunicatorId),
}
