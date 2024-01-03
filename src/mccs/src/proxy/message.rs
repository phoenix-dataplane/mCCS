use crate::comm::CommunicatorId;
use crate::transport::channel::PeerConnId;
use crate::transport::transporter::ConnectHandle;

pub enum ProxyPeerMessage {
    ConnectInfoExchange(CommunicatorId, PeerConnId, ConnectHandle),
}
