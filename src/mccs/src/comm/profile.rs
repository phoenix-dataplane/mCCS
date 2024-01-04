use crate::transport::net::provider::RDMA_TRANSPORT;
use crate::transport::net::provider::{NetProperties, NetProvierWrap};
use crate::transport::NUM_PROTOCOLS;

// comm profile, setting and
pub struct CommProfile {
    pub buff_sizes: [usize; NUM_PROTOCOLS],
    pub peers_local_rank: Vec<usize>,
    pub peers_cuda_device_idx: Vec<i32>,
    pub network_devices: Vec<NetProperties>,
}

impl CommProfile {
    #[inline]
    pub fn get_local_rank(&self, rank: usize) -> usize {
        self.peers_local_rank[rank]
    }

    #[inline]
    pub fn get_cuda_device_idx(&self, rank: usize) -> i32 {
        self.peers_cuda_device_idx[rank]
    }

    // (net_device, proxy_rank)
    #[inline]
    pub fn get_network_device(&self, rank: usize, peer_rank: usize) -> (usize, usize) {
        (0, rank)
    }

    #[inline]
    pub fn check_gdr(&self, rank: usize, net_dev: usize, read: bool) -> bool {
        false
    }

    #[inline]
    pub fn check_gdr_need_flush(&self, rank: usize) -> bool {
        false
    }

    #[inline]
    pub fn get_net_provider(&self) -> &'static dyn NetProvierWrap {
        todo!()
    }
}
