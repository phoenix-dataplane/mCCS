use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetTransportConfig {
    pub gdr_enable: bool,
    pub gdr_copy_sync_enable: bool,
    pub gdr_copy_flush_enable: bool,
}

impl Default for NetTransportConfig {
    fn default() -> Self {
        NetTransportConfig {
            gdr_enable: false,
            gdr_copy_sync_enable: false,
            gdr_copy_flush_enable: false,
        }
    }
}
