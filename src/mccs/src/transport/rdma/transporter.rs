use async_trait::async_trait;

use crate::comm::CommProfile;
use crate::transport::catalog::TransportCatalog;
use crate::transport::channel::PeerConnId;
use crate::transport::op::TransportOp;
use crate::transport::transporter::{
    AnyResources, ConnectInfo, TransportConnect, TransportSetup, Transporter,
};

pub struct RdmaTransporter {}

#[async_trait]
impl Transporter for RdmaTransporter {
    fn send_setup(
        &self,
        profile: &CommProfile,
        conn_id: &PeerConnId,
        catalog: &TransportCatalog,
    ) -> TransportSetup {
        todo!()
    }

    fn send_connect(
        &self,
        conn_id: &PeerConnId,
        connect_info: ConnectInfo,
        setup_resources: Option<AnyResources>,
    ) -> TransportConnect {
        todo!()
    }

    // Setup receiver transport
    fn recv_setup(
        &self,
        profile: &CommProfile,
        conn_id: &PeerConnId,
        catalog: &TransportCatalog,
    ) -> TransportSetup {
        todo!()
    }

    // Connect receiver transport
    fn recv_connect(
        &self,
        conn_id: &PeerConnId,
        connect_info: ConnectInfo,
        setup_resources: Option<AnyResources>,
    ) -> TransportConnect {
        todo!()
    }

    // Progress transport op for send connection
    fn agent_send_progress_op(&self, op: &mut TransportOp, resources: &mut AnyResources) {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Progress transport op for recv connection
    fn agent_recv_progress_op(&self, op: &mut TransportOp, resources: &mut AnyResources) {
        unimplemented!("Transport agent is not implemented for this transport");
    }
}
