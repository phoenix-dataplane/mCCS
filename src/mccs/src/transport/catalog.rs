use std::any::Any;

use dashmap::mapref::one::{MappedRef, MappedRefMut};
use dashmap::DashMap;
use thiserror::Error;

pub type AnyConfig = Box<dyn Any + Send + Sync>;
pub type ConfigRef<'a, T> = MappedRef<'a, String, AnyConfig, T>;
pub type ConfigRefMut<'a, T> = MappedRefMut<'a, String, AnyConfig, T>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Fail to downcast to a concrete type")]
    Downcast,
    #[error("Resources not found")]
    NotFound,
}

// TODO: temporary solution for async agent setup
pub struct TransportCatalog {
    config: DashMap<String, AnyConfig>,
}

impl TransportCatalog {
    pub fn new() -> Self {
        TransportCatalog {
            config: DashMap::new(),
        }
    }
}

impl Default for TransportCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl TransportCatalog {
    pub fn register_config<T>(&self, name: String, config: T)
    where
        T: Any + Send + Sync,
    {
        let boxed = Box::new(config);
        self.config.insert(name, boxed);
    }

    pub fn remove_config<T>(&self, name: &str) {
        self.config.remove(name);
    }

    pub fn get_config<T>(&self, name: &str) -> Result<ConfigRef<T>, Error>
    where
        T: Any + Send + Sync,
    {
        let config = self.config.get(name);
        if let Some(entry) = config {
            let concrete = entry
                .try_map(|x| x.downcast_ref::<T>())
                .map_err(|_| Error::Downcast)?;
            Ok(concrete)
        } else {
            Err(Error::NotFound)
        }
    }

    pub fn get_config_mut<T>(&self, name: &str) -> Result<ConfigRefMut<T>, Error>
    where
        T: Any + Send + Sync,
    {
        let config = self.config.get_mut(name);
        if let Some(entry) = config {
            let concrete = entry
                .try_map(|x| x.downcast_mut::<T>())
                .map_err(|_| Error::Downcast)?;
            Ok(concrete)
        } else {
            Err(Error::NotFound)
        }
    }
}
