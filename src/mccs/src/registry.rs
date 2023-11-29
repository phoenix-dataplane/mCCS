use std::collections::HashMap;

use dashmap::DashMap;

use crate::comm::{ChannelCommPattern, CommunicatorId, HostIdent, PeerInfo, PeerType};
use crate::pattern;
use crate::transport::catalog::TransportCatalog;
use crate::transport::channel::{ChannelId, PeerConnId};
use crate::transport::delegator::TransportDelegator;
use crate::transport::transporter::Transporter;
use crate::transport::SHM_TRANSPORTER;

#[derive(Clone)]
pub struct RankInfo {
    exchanged: bool,
    host: HostIdent,
    cuda_device_idx: i32,
}

pub struct CommunicatorInfo {
    num_ranks: usize,
    ranks_info: Vec<RankInfo>,
}

impl CommunicatorInfo {
    pub fn new(num_ranks: usize) -> Self {
        let sock_addr =
            std::net::SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0)), 0);
        let host_ident = HostIdent(sock_addr);
        let rank_info = RankInfo {
            exchanged: false,
            host: host_ident,
            cuda_device_idx: 0,
        };
        let ranks_info = vec![rank_info; num_ranks];
        CommunicatorInfo {
            num_ranks,
            ranks_info,
        }
    }
}

pub struct GlobalRegistry {
    pub communicators: DashMap<CommunicatorId, CommunicatorInfo>,
    pub transport_delegator: TransportDelegator,
    pub transport_catalog: TransportCatalog,
}

impl GlobalRegistry {
    pub fn register_communicator_rank(
        &self,
        comm_id: CommunicatorId,
        rank: usize,
        num_ranks: usize,
        info: &PeerInfo,
    ) {
        let mut comm = self
            .communicators
            .entry(comm_id)
            .or_insert_with(|| CommunicatorInfo::new(num_ranks));
        let peer_info = &mut comm.ranks_info[rank];
        peer_info.host = info.host;
        peer_info.cuda_device_idx = info.cuda_device_idx;
        peer_info.exchanged = true;
    }

    pub fn query_communicator_peers(
        &self,
        comm_id: CommunicatorId,
        rank: usize,
        peers_to_query: &mut Vec<usize>,
        peers_info: &mut HashMap<usize, PeerInfo>,
    ) -> usize {
        use dashmap::try_result::TryResult;

        let comm = match self.communicators.try_get(&comm_id) {
            TryResult::Present(comm) => comm,
            TryResult::Absent => panic!("communicator is not registered"),
            TryResult::Locked => return 0,
        };
        let mut idx = 0;
        let mut count = 0;
        while idx < peers_to_query.len() {
            let peer = peers_to_query[idx];
            if comm.ranks_info[peer].exchanged {
                let peer_rank_info = &comm.ranks_info[rank];
                let intra_host = comm.ranks_info[rank].host == peer_rank_info.host;
                let peer_type = match intra_host {
                    true => PeerType::IntraNode,
                    false => PeerType::InterNode,
                };
                let peer_info = PeerInfo {
                    peer_type,
                    host: peer_rank_info.host,
                    cuda_device_idx: peer_rank_info.cuda_device_idx,
                };
                peers_info.insert(peer, peer_info);
                peers_to_query.swap_remove(idx);
                count += 1;
            } else {
                idx += 1;
            }
        }
        count
    }

    pub fn arbitrate_comm_patterns(
        &self,
        comm_id: CommunicatorId,
        rank: usize,
    ) -> Option<Vec<ChannelCommPattern>> {
        use dashmap::try_result::TryResult;

        let comm = match self.communicators.try_get(&comm_id) {
            TryResult::Present(comm) => comm,
            TryResult::Absent => panic!("communicator is not registered"),
            TryResult::Locked => return None,
        };
        let ring_next = (rank + 1) % comm.num_ranks;
        let ring_prev = (rank + comm.num_ranks - 1) % comm.num_ranks;
        // in current implentation, ring follows the same ordering of ranks
        let ring_index = rank;

        let mut user_ranks = Vec::with_capacity(comm.num_ranks);
        for idx in 0..comm.num_ranks {
            let ring_rank = (rank + idx) % comm.num_ranks;
            user_ranks.push(ring_rank);
        }
        let ring_pattern = pattern::RingPattern {
            prev: ring_prev,
            next: ring_next,
            user_ranks,
            index: ring_index,
        };

        let channel = ChannelCommPattern {
            channel: ChannelId(0),
            ring: ring_pattern,
        };
        let channels = vec![channel];

        Some(channels)
    }

    // Decide which type of transporter
    // shall be used for a peer connection
    pub fn arbitrate_conn_transporter(
        &self,
        _comm_id: CommunicatorId,
        _rank: usize,
        _peer: &PeerConnId,
    ) -> &'static dyn Transporter {
        &SHM_TRANSPORTER
    }
}
