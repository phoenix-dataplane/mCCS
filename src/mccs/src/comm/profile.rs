use crate::transport::net::provider::RDMA_TRANSPORT;
use crate::transport::net::provider::{NetProperties, NetProvierWrap};
use crate::transport::NUM_PROTOCOLS;

// comm profile, setting and
pub struct CommProfile {
    pub buff_sizes: [usize; NUM_PROTOCOLS],
}

impl CommProfile {
    // (net_device, proxy_rank)
    // TODO: choose net dev that is closest to the specified GPU
    // and allow admins to specify the set of allowed net devs
    #[inline]
    pub fn get_network_device(&self, rank: usize, _peer_rank: usize) -> (usize, usize) {
        (0, rank)
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
