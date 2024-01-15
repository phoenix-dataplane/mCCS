use std::sync::Arc;

use crate::config::DefaultCommConfig;
use crate::transport::catalog::TransportCatalog;
use crate::transport::delegator::TransportDelegator;

#[derive(Clone)]
pub struct GlobalRegistry {
    pub default_comm_config: DefaultCommConfig,
    pub transport_delegator: Arc<TransportDelegator>,
    pub transport_catalog: Arc<TransportCatalog>,
}
