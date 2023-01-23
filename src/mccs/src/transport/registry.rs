use dashmap::DashMap;
use thiserror::Error;

use super::connector::ConnectorIdentifier;
use super::hmem::resources::{HostMemTptSetupSender, HostMemTptSetupReceiver};

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("Resource not found in the table")]
    NotFound,
    #[error("Resource exists in the table")]
    Exists,
}


// TBD: use sender/receiver instead of global registry to setup
pub struct TransportSetupRegistry {
    pub hmem_senders: DashMap<ConnectorIdentifier, HostMemTptSetupSender>,
    pub hmem_receivers: DashMap<ConnectorIdentifier, HostMemTptSetupReceiver>,
}

impl TransportSetupRegistry {
    pub fn new() -> Self {
        TransportSetupRegistry {
            hmem_senders: DashMap::new(),
            hmem_receivers: DashMap::new(),
        }
    }
}

impl TransportSetupRegistry {
    pub fn insert_hmem_sender(&self, id: ConnectorIdentifier, sender: HostMemTptSetupSender) -> Result<(), Error> {
        match self.hmem_senders.insert(id, sender) {
            Some(_) => Err(Error::Exists),
            None => Ok(()),
        }
    }

    pub fn insert_hmem_receiver(&self, id: ConnectorIdentifier, receiver: HostMemTptSetupReceiver) -> Result<(), Error> {
        match self.hmem_receivers.insert(id, receiver) {
            Some(_) => Err(Error::Exists),
            None => Ok(()),
        }
    }
}