use dashmap::DashMap;

use crate::communicator::CommunicatorGlobalInfo;
use crate::transport::registry::TransportSetupRegistry;

pub struct GlobalResources {
    pub communicators: DashMap<u32, CommunicatorGlobalInfo>,
    pub transport_setup: TransportSetupRegistry,
}