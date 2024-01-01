use crate::comm::CommProfile;
use crate::transport::channel::PeerConnId;
use crate::transport::transporter::{Transporter, ConnectInfo, AnyResources, TransportSetup, TransportConnect};
use crate::transport::catalog::TransportCatalog;

pub struct NetTransport;


impl Transporter for NetTransport {
    fn send_setup(&self, profile: &CommProfile,conn_id: &PeerConnId,catalog: &TransportCatalog) -> TransportSetup {
        todo!()
    }

    fn send_connect(&self,conn_id: &PeerConnId,connect_info:ConnectInfo,setup_resources:Option<AnyResources> ,) -> TransportConnect {
        todo!()
    }

    fn recv_setup(&self,profile: &CommProfile,conn_id: &PeerConnId,catalog: &TransportCatalog,) -> TransportSetup {
        todo!()
    }

    fn recv_connect(&self,conn_id: &PeerConnId,connect_info:ConnectInfo,setup_resources:Option<AnyResources> ,) -> TransportConnect {
        todo!()
    }
}