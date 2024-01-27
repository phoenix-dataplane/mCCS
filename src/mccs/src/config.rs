use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use qos_service::QosScheduleDef;

use crate::transport::net::config::NetTransportConfig;
use crate::transport::net::provider::RdmaTransportConfig;
use crate::transport::shm::config::ShmTransportConfig;
use crate::transport::NUM_PROTOCOLS;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultCommConfig {
    #[serde(rename = "buffer_sizes")]
    pub buf_sizes: [usize; NUM_PROTOCOLS],
    pub channel_count: u32,
    // TODO: specify number of channels and ring for each channel
}

impl Default for DefaultCommConfig {
    fn default() -> Self {
        DefaultCommConfig {
            // 4MB
            buf_sizes: [1 << 22],
            channel_count: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelPattern {
    pub channel_id: u32,
    pub ring: Vec<usize>,
    // (send_rank, recv_rank) -> port
    pub udp_sport: Option<Vec<(usize, usize, u16)>>,
    pub net_dev: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommPatternConfig {
    pub communicator_id: u32,
    pub channels: Vec<ChannelPattern>,
    pub ib_traffic_class: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommGlobalConfig {
    #[serde(rename = "net_rdma", default)]
    pub rdma_config: RdmaTransportConfig,
    #[serde(rename = "net", default)]
    pub net_config: NetTransportConfig,
    #[serde(rename = "shm", default)]
    pub shm_config: ShmTransportConfig,
}

impl Default for CommGlobalConfig {
    fn default() -> Self {
        CommGlobalConfig {
            rdma_config: Default::default(),
            net_config: Default::default(),
            shm_config: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Control {
    pub prefix: PathBuf,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub control: Control,
    #[serde(default)]
    pub comm_global_config: CommGlobalConfig,
    #[serde(default)]
    pub comm_default_config: DefaultCommConfig,
    pub addrs: Vec<IpAddr>,
    pub listen_port: u16,
    pub mccs_daemon_basename: String,
    pub mccs_daemon_prefix: PathBuf,
    pub qos_schedule: Option<QosScheduleDef>,
    pub comm_patterns_override: Option<Vec<CommPatternConfig>>,
}

impl Config {
    pub fn from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config = toml::from_str(&content)?;
        Ok(config)
    }
}
