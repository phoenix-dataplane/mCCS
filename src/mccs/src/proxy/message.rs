use crate::transport::channel::PeerConnId;
use crate::transport::transporter::ConnectInfo;
use crate::comm::CommunicatorId;

pub enum ProxyPeerMessage {
    ConnectInfoExchange(CommunicatorId, PeerConnId, ConnectInfo),
}