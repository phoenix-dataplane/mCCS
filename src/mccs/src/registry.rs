use std::collections::HashMap;
use std::sync::Arc;

use crate::comm::CommunicatorId;
use crate::config::CommPatternConfig;
use crate::config::DefaultCommConfig;
use crate::transport::catalog::TransportCatalog;
use crate::transport::delegator::TransportDelegator;

#[derive(Clone)]
pub struct GlobalRegistry {
    pub default_comm_config: DefaultCommConfig,
    pub comm_pattern_override: HashMap<CommunicatorId, CommPatternConfig>,
    pub transport_delegator: Arc<TransportDelegator>,
    pub transport_catalog: Arc<TransportCatalog>,
}
