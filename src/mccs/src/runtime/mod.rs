pub mod affinity;
pub mod executor;
pub mod manager;

pub use affinity::CoreMask;
pub use executor::{Runtime, RuntimeId, RuntimeMode};
pub use manager::RuntimeManager;

use crate::engine::Engine;
pub type EngineContainer = Box<dyn Engine>;
