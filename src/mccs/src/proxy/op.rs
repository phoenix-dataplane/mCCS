use crate::communicator::CommunicatorId;


pub enum ProxyOp {
    InitCommunicator(CommunicatorId),
}
