pub mod executor;
pub mod manager;

pub use executor::{Runtime, RuntimeMode};
pub use manager::RuntimeManager;

use crate::engine::Engine;
pub type EngineContainer = Box<dyn Engine>;
