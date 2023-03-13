use std::any::Any;
use async_trait::async_trait;

use crate::comm::{CommunicatorId, CommProfile};
use super::channel::{PeerConnId, PeerConnInfo};
use super::op::TransportOp;
use super::catalog::TransportCatalog;

pub type AgentMessage = Option<Box<dyn Any + Send>>;
pub type AnyResources = Box<dyn Any + Send>;
pub type ConnectInfo = Box<dyn Any + Send>;

pub enum TransportSetup {
    PreAgentCb {
        agent_request: AgentMessage,
        setup_resources: Option<AnyResources>,
    },
    Setup {
        peer_connect_info: ConnectInfo,
        setup_resources: Option<AnyResources>,
    },
}

pub enum TransportConnect {
    PreAgentCb {
        agent_request: AgentMessage,
        transport_resources: Option<AnyResources>,
    },
    Connect {
        conn_info: PeerConnInfo,
        transport_resources: AnyResources,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransportAgentId {
    pub communicator_id: CommunicatorId,
    pub client_rank: usize,
    pub client_cuda_dev: i32,
    pub peer_conn: PeerConnId,
}

#[async_trait]
pub trait Transporter: Send + Sync {
    // Setup sender transport, prepare any sender-side resources
    // that hold by sender rank,
    // returns PreAgentCb variant of TransportSetup
    // if transport agent setup is required,
    // otherwise, returns Setup variant
    fn send_setup(
        &self, 
        profile: &CommProfile,
        conn_id: &PeerConnId,
        catalog: &TransportCatalog,
    ) -> TransportSetup;

    // If agent setup is requested, then this function will be invoked,
    // to finish up remaining work
    // optional `setup_resources` setup during `send_setup` will be passed in,
    // must return Setup variant of TransportSetup
    fn send_setup_agent_callback(
        &self, 
        _conn_id: &PeerConnId, 
        _agent_reply: AgentMessage, 
        _setup_resources: Option<AnyResources>,
    ) -> TransportSetup {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Connect sender transport to receiver,
    // `conn_resources` that is set by the receiver setup is passed in
    // if transport agent connect is required,
    // returns PreAgentCb variant of TransportConnect
    // otherwise return Connect variant
    fn send_connect(
        &self,
        conn_id: &PeerConnId,
        connect_info: ConnectInfo,
        setup_resources: Option<AnyResources>,
    ) -> TransportConnect;

    // If agent connect is requested, then this function is invoked
    // after agent replies
    // must returns Connect variant of TransportConnect
    fn send_connect_agent_callback(
        &self,
        _conn_id: &PeerConnId,
        _agent_reply: AgentMessage,
        _transport_resources: Option<AnyResources>,
    ) -> TransportConnect {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Setup receiver transport
    fn recv_setup(
        &self, 
        profile: &CommProfile,
        conn_id: &PeerConnId,
        catalog: &TransportCatalog,
    ) -> TransportSetup;

    // Complete receiver transport setup with transport agent,
    // after agent completes setup and replies
    fn recv_setup_agent_callback(
        &self, 
        _conn_id: &PeerConnId, 
        _agent_reply: AgentMessage, 
        _setup_resources: Option<AnyResources>
    ) -> TransportSetup {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Connect receiver transport
    fn recv_connect(
        &self,
        conn_id: &PeerConnId,
        connect_info: ConnectInfo,
        setup_resources: Option<AnyResources>,
    ) -> TransportConnect;

    // Complete receiver transport connect with transport agent
    // after agent completes connect and replies
    fn recv_connect_agent_callback(
        &self,
        _conn_id: &PeerConnId,
        _agent_reply: AgentMessage,
        _transport_resources: Option<AnyResources>,
    ) -> TransportConnect {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Transport agent sender setup
    async fn agent_send_setup(
        &self,
        _id: TransportAgentId,
        _agent_request: AgentMessage,
    ) -> (AnyResources, AgentMessage) {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Transport agent sender connect
    async fn agent_send_connect(
        &self,
        _id: TransportAgentId,
        _agent_request: AgentMessage,
        _setup_resources: Option<AnyResources>,
    ) -> (AnyResources, AgentMessage) {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Transport agent receiver setup
    async fn agent_recv_setup(
        &self,
        _id: TransportAgentId,
        _agent_request: AgentMessage,
    ) -> (AnyResources, AgentMessage) {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Transport agent receiver connect
    async fn agent_recv_connect(
        &self,
        _id: TransportAgentId,
        _agent_request: AgentMessage,
        _setup_resources: Option<AnyResources>,
    ) -> (AnyResources, AgentMessage) {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Progress transport op for send connection
    fn agent_send_progress_op(
        &self,
        _op: &mut TransportOp,
        _resources: &mut AnyResources,
    ) {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Progress transport op for recv connection
    fn agent_recv_progress_op(
        &self,
        _op: &mut TransportOp,
        _resources: &mut AnyResources,
    ) {
        unimplemented!("Transport agent is not implemented for this transport");
    }
}
