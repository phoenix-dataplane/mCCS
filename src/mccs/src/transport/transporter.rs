use std::any::Any;
use std::mem::MaybeUninit;
use std::sync::Arc;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;

use qos_service::QosSchedule;

use super::catalog::TransportCatalog;
use super::channel::{PeerConnId, PeerConnInfo};
use super::op::TransportOp;
use crate::comm::{CommProfile, CommunicatorId, PeerInfo};

pub type AgentMessage = Option<Box<dyn Any + Send>>;
pub type AnyResources = Box<dyn Any + Send>;
pub type TransporterError = anyhow::Error;

pub const CONNECT_HANDLE_SIZE: usize = 128;

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct ConnectHandle(pub [u8; CONNECT_HANDLE_SIZE]);

#[derive(Debug, Error)]
pub enum ConnectHandleError {
    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("Required size {0} exceeds maximum of {}", CONNECT_HANDLE_SIZE)]
    ExceedMaxSize(usize),
}

impl ConnectHandle {
    pub fn serialize_from<T: Serialize>(handle: T) -> Result<Self, ConnectHandleError> {
        let mut serialized = [0u8; CONNECT_HANDLE_SIZE];
        let required_size = bincode::serialized_size(&handle)?;
        if required_size as usize > CONNECT_HANDLE_SIZE {
            return Err(ConnectHandleError::ExceedMaxSize(required_size as usize));
        }
        bincode::serialize_into(serialized.as_mut_slice(), &handle)?;
        let serialized_handle = ConnectHandle(serialized);
        Ok(serialized_handle)
    }

    pub fn deserialize_to<T: DeserializeOwned>(&self) -> Result<T, ConnectHandleError> {
        let handle = bincode::deserialize::<T>(self.0.as_slice())?;
        Ok(handle)
    }
}

pub enum TransportSetup {
    PreAgentCb {
        agent_cuda_dev: i32,
        agent_request: AgentMessage,
        setup_resources: Option<AnyResources>,
    },
    Setup {
        peer_connect_handle: ConnectHandle,
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
    },
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
    #[inline]
    // Determine whether this transporter needs TransportOp
    fn need_op(&self) -> bool {
        true
    }

    #[inline]
    // Determine whether two peers can communicate
    fn can_connect(
        &self,
        _send_peer: &PeerInfo,
        _recv_peer: &PeerInfo,
        _profile: &CommProfile,
        _catalog: &TransportCatalog,
    ) -> bool {
        false
    }

    // Setup sender transport, prepare any sender-side resources
    // that hold by sender rank,
    // returns PreAgentCb variant of TransportSetup
    // if transport agent setup is required,
    // otherwise, returns Setup variant
    fn send_setup(
        &self,
        conn_id: &PeerConnId,
        my_info: &PeerInfo,
        peer_info: &PeerInfo,
        profile: &CommProfile,
        catalog: &TransportCatalog,
    ) -> Result<TransportSetup, TransporterError>;

    // If agent setup is requested, then this function will be invoked,
    // to finish up remaining work
    // optional `setup_resources` setup during `send_setup` will be passed in,
    // must return Setup variant of TransportSetup
    fn send_setup_agent_callback(
        &self,
        _rank: usize,
        _conn_id: &PeerConnId,
        _agent_reply: AgentMessage,
        _setup_resources: Option<AnyResources>,
    ) -> Result<TransportSetup, TransporterError> {
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
        connect_handle: ConnectHandle,
        setup_resources: Option<AnyResources>,
    ) -> Result<TransportConnect, TransporterError>;

    // If agent connect is requested, then this function is invoked
    // after agent replies
    // must returns Connect variant of TransportConnect
    fn send_connect_agent_callback(
        &self,
        _conn_id: &PeerConnId,
        _agent_reply: AgentMessage,
        _transport_resources: Option<AnyResources>,
    ) -> Result<TransportConnect, TransporterError> {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Setup receiver transport
    fn recv_setup(
        &self,
        conn_id: &PeerConnId,
        my_info: &PeerInfo,
        peer_info: &PeerInfo,
        profile: &CommProfile,
        catalog: &TransportCatalog,
    ) -> Result<TransportSetup, TransporterError>;

    // Complete receiver transport setup with transport agent,
    // after agent completes setup and replies
    fn recv_setup_agent_callback(
        &self,
        _rank: usize,
        _conn_id: &PeerConnId,
        _agent_reply: AgentMessage,
        _setup_resources: Option<AnyResources>,
    ) -> Result<TransportSetup, TransporterError> {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Connect receiver transport
    fn recv_connect(
        &self,
        conn_id: &PeerConnId,
        connect_handle: ConnectHandle,
        setup_resources: Option<AnyResources>,
    ) -> Result<TransportConnect, TransporterError>;

    // Complete receiver transport connect with transport agent
    // after agent completes connect and replies
    fn recv_connect_agent_callback(
        &self,
        _conn_id: &PeerConnId,
        _agent_reply: AgentMessage,
        _transport_resources: Option<AnyResources>,
    ) -> Result<TransportConnect, TransporterError> {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Transport agent sender setup
    async fn agent_send_setup(
        &self,
        _id: TransportAgentId,
        _agent_request: AgentMessage,
        _catalog: Arc<TransportCatalog>,
    ) -> Result<(AnyResources, AgentMessage), TransporterError> {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Transport agent sender connect
    async fn agent_send_connect(
        &self,
        _id: TransportAgentId,
        _agent_request: AgentMessage,
        _setup_resources: Option<AnyResources>,
    ) -> Result<(AnyResources, AgentMessage), TransporterError> {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Transport agent receiver setup
    async fn agent_recv_setup(
        &self,
        _id: TransportAgentId,
        _agent_request: AgentMessage,
        _catalog: Arc<TransportCatalog>,
    ) -> Result<(AnyResources, AgentMessage), TransporterError> {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Transport agent receiver connect
    async fn agent_recv_connect(
        &self,
        _id: TransportAgentId,
        _agent_request: AgentMessage,
        _setup_resources: Option<AnyResources>,
    ) -> Result<(AnyResources, AgentMessage), TransporterError> {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Progress transport op for send connection
    fn agent_send_progress_op(
        &self,
        _op: &mut TransportOp,
        _resources: &mut AnyResources,
        _schedule: &QosSchedule,
    ) -> Result<(), TransporterError> {
        unimplemented!("Transport agent is not implemented for this transport");
    }

    // Progress transport op for recv connection
    fn agent_recv_progress_op(
        &self,
        _op: &mut TransportOp,
        _resources: &mut AnyResources,
        _schedule: &QosSchedule,
    ) -> Result<(), TransporterError> {
        unimplemented!("Transport agent is not implemented for this transport");
    }
}
