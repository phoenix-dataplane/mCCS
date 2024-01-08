use std::sync::Arc;

use crate::transport::catalog::TransportCatalog;
use crate::transport::delegator::TransportDelegator;

pub struct GlobalRegistry {
    pub transport_delegator: TransportDelegator,
    pub transport_catalog: Arc<TransportCatalog>,
}
