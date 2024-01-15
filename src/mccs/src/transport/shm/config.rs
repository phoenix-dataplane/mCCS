use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShmTransportConfig {
    pub locality: ShmLocality,
    #[serde(rename = "memcpy_send")]
    pub use_memcpy_send: bool,
    #[serde(rename = "memcpy_recv")]
    pub use_memcpy_recv: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShmLocality {
    Sender,
    Receiver,
}

impl Default for ShmTransportConfig {
    fn default() -> Self {
        ShmTransportConfig {
            locality: ShmLocality::Sender,
            use_memcpy_send: false,
            use_memcpy_recv: false,
        }
    }
}
