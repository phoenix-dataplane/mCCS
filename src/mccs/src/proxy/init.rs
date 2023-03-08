use std::collections::{HashMap, HashSet, VecDeque};

use crate::transport::engine::TransportEngineId;
use crate::transport::transporter::{Transporter, AgentMessage, AnyResources, ConnectInfo}; 
use crate::transport::channel::{PeerConnId, PeerConnector, ConnType, CommChannel, ChannelPeerConn};
use crate::communicator::{CommunicatorId, PeerInfo, ChannelCommPattern, Communicator, CommProfile};

pub struct PeerConnConstruct {
    pub transporter: &'static dyn Transporter,
    pub resources: Option<AnyResources>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommInitStage {
    RegisterRank,
    PeerInfoExchange,
    ConnectChannel,
}

pub struct CommInitState {
    pub id: CommunicatorId,
    pub stage: CommInitStage,

    pub rank: usize,
    pub num_ranks: usize,
    pub profile: CommProfile,
    pub peers_info: HashMap<usize, PeerInfo>,
    pub peers_await_exchange: Vec<usize>,

    pub comm_patterns: Vec<ChannelCommPattern>,

    pub to_setup: VecDeque<PeerConnId>,
    pub to_setup_agent_cb: VecDeque<(PeerConnId, AgentMessage)>,
    pub to_connect: VecDeque<(PeerConnId, ConnectInfo)>,
    pub to_connect_agent_cb: VecDeque<(PeerConnId, AgentMessage)>,
    pub peer_transport_assigned: HashMap<PeerConnId, TransportEngineId>,

    pub peer_setup_pre_agent: HashMap<PeerConnId, PeerConnConstruct>,
    pub peer_setup: HashMap<PeerConnId, PeerConnConstruct>,
    pub peer_connect_pre_agent: HashMap<PeerConnId, PeerConnConstruct>,

    pub peer_connected: HashMap<PeerConnId, PeerConnector>,
    pub await_connections: usize,

    pub pending_conn: HashSet<PeerConnId>,
}

impl CommInitState {
    pub fn enqueue_channels_setup(&mut self) {
        for pattern in self.comm_patterns.iter() {
            let ring_send = PeerConnId {
                peer_rank: pattern.ring.next,
                channel: pattern.channel,
                conn_index: 0,
                conn_type: ConnType::Send,
            };
            let ring_recv = PeerConnId {
                peer_rank: pattern.ring.prev,
                channel: pattern.channel,
                conn_index: 0,
                conn_type: ConnType::Recv,
            };
            self.to_setup.push_back(ring_send);
            self.to_setup.push_back(ring_recv);
        }
    }

    pub fn finalize_communicator(self) -> Communicator {
        let mut send_peer_conns = HashMap::new();
        let mut recv_peer_conns = HashMap::new();
        for (peer_conn, peer_connector) in self.peer_connected {
            let peer_conns = match peer_conn.conn_type {
                ConnType::Send => &mut send_peer_conns,
                ConnType::Recv => &mut recv_peer_conns,
            };
            let channel = peer_conns
                .entry(peer_conn.channel)
                .or_insert_with(HashMap::new);
            let chan_peer = channel
                .entry(peer_conn.peer_rank)
                .or_insert_with(Vec::new);
            chan_peer.push(peer_connector);
        }

        let mut channels = Vec::new();
        for chan_pattern in self.comm_patterns {
            let send_peers = send_peer_conns
                .remove(&chan_pattern.channel)
                .unwrap_or_else(HashMap::new);
            let mut recv_peers = recv_peer_conns
                .remove(&chan_pattern.channel)
                .unwrap_or_else(HashMap::new);
            let mut chan_peers = Vec::new();
            for (peer_rank, send) in send_peers {
                let recv = recv_peers.remove(&peer_rank).unwrap_or_else(Vec::new);
                let peer = ChannelPeerConn {
                    send,
                    recv,
                    peer_rank,
                };
                chan_peers.push(peer);
            }

            let channel = CommChannel {
                id: chan_pattern.channel,
                peers: chan_peers,
                ring: chan_pattern.ring,
            };
            channels.push(channel);
        }
        todo!()
    }
}