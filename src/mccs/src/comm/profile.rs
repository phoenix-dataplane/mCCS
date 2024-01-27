use std::collections::HashMap;

use crate::transport::channel::{ChannelId, PeerConnId};
use crate::transport::net::provider::RDMA_TRANSPORT;
use crate::transport::net::provider::{NetProperties, NetProvierWrap};
use crate::transport::NUM_PROTOCOLS;

// comm profile, setting and
pub struct CommProfile {
    pub buff_sizes: [usize; NUM_PROTOCOLS],
    pub udp_sport_map: HashMap<PeerConnId, u16>,
    pub channel_net_device_map: HashMap<ChannelId, String>,
    pub tc: Option<u8>,
}

impl CommProfile {
    // (net_device, proxy_rank)
    // TODO: choose net dev that is closest to the specified GPU
    // and allow admins to specify the set of allowed net devs
    #[inline]
    pub fn get_network_device(
        &self,
        channel_id: ChannelId,
        my_rank: usize,
        _peer_rank: usize,
    ) -> (usize, usize) {
        let prefix = self.channel_net_device_map.get(&channel_id);
        let num_devices = RDMA_TRANSPORT.get_num_devices().unwrap();
        if let Some(prefix) = prefix {
            for dev in 0..num_devices {
                let props = RDMA_TRANSPORT.get_properties(dev).unwrap();
                if props.name.starts_with(prefix) {
                    return (dev, my_rank);
                }
            }
        }
        (0, my_rank)
    }

    #[inline]
    pub fn get_udp_sport(&self, peer_conn_id: &PeerConnId) -> Option<u16> {
        self.udp_sport_map.get(peer_conn_id).copied()
    }

    #[inline]
    pub fn get_tc(&self) -> Option<u8> {
        self.tc
    }

    #[inline]
    pub fn check_gdr(&self, _rank: usize, _net_dev: usize, _read: bool) -> bool {
        false
    }

    #[inline]
    pub fn check_gdr_need_flush(&self, _rank: usize) -> bool {
        false
    }

    #[inline]
    pub fn get_net_provider(&self) -> &'static dyn NetProvierWrap {
        &RDMA_TRANSPORT
    }
}
