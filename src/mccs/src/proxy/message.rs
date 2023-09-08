use crate::comm::CommunicatorId;
use crate::transport::channel::PeerConnId;
use crate::transport::transporter::ConnectInfo;

pub enum ProxyPeerMessage {
    ConnectInfoExchange(CommunicatorId, PeerConnId, ConnectInfo),
}
