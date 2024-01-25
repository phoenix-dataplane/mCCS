use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::sync::Arc;

use thiserror::Error;

use super::catalog::TransportCatalog;
use super::channel::{ChannelId, ConnType, PeerConnId, PeerConnInfo};
use super::transporter::TransporterError;
use super::transporter::CONNECT_HANDLE_SIZE;
use super::transporter::{AgentMessage, AnyResources, ConnectHandle, Transporter};
use super::ALL_TRANSPORTERS;
use crate::bootstrap::{BootstrapError, BootstrapState};
use crate::comm::{CommProfile, PeerInfo};

#[derive(Debug, Error)]
pub enum TransportConnectError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Transporter error: {0}")]
    Transporter(#[from] TransporterError),
    #[error("Bootstrap error: {0}")]
    Bootstrap(#[from] BootstrapError),
    #[error("No transport found for rank {0} -> rank {1}")]
    NoTransportFound(usize, usize),
    #[error("Connection {0:?} not found")]
    ConnectionNotFound(PeerConnId),
    #[error("Current transport connect phase is still in progress")]
    ConnectPhaseInProgress,
    #[error("Connection index mismatch: {0} vs {1}")]
    ConnIndexMismatch(u32, u32),
}

pub struct PeerTransportSetupTask {
    pub id: PeerConnId,
    pub transporter: &'static dyn Transporter,
}

pub struct PeerTransportSetupAgentCbTask {
    pub id: PeerConnId,
    pub transporter: &'static dyn Transporter,
    pub agent_message: AgentMessage,
    pub setup_resources: Option<AnyResources>,
}

pub struct PeerTransportConnectTask {
    pub id: PeerConnId,
    pub transporter: &'static dyn Transporter,
    // connect handle received from peer
    pub peer_connect_handle: ConnectHandle,
    pub setup_resources: Option<AnyResources>,
}

pub struct PeerTransportConnectAgentCbTask {
    pub id: PeerConnId,
    pub transporter: &'static dyn Transporter,
    pub agent_message: AgentMessage,
    pub transport_resources: Option<AnyResources>,
}

pub struct PeerTransportHandleExchangeTask {
    pub rank: usize,
    pub num_ranks: usize,
    pub num_channels: usize,
    pub conn_index: u32,
    pub connect_recv: Vec<u64>,
    pub connect_send: Vec<u64>,
    pub handles: VecDeque<HashMap<PeerConnId, ConnectHandle>>,
}

pub enum TransportConnectTask {
    Idle,
    WaitingOutstandingTask,
    PeerTransportSetup(PeerTransportSetupTask),
    PeerTransportSetupAgentCb(PeerTransportSetupAgentCbTask),
    PeerTransportConnect(PeerTransportConnectTask),
    PeerTransportConnectAgentCb(PeerTransportConnectAgentCbTask),
    PeerTransportConnectHandleExchange(PeerTransportHandleExchangeTask),
}

pub struct PeerConnConstructor {
    pub transporter: &'static dyn Transporter,
    pub resources: Option<AnyResources>,
}

pub struct PeerConnected {
    pub conn_info: PeerConnInfo,
    pub transporter: &'static dyn Transporter,
    pub resources: AnyResources,
}

pub struct TransportConnectState {
    pub num_channels: usize,
    pub rank: usize,
    pub num_ranks: usize,

    pub to_setup: VecDeque<PeerConnId>,
    pub to_setup_agent_cb: VecDeque<(PeerConnId, AgentMessage)>,
    pub to_connect: VecDeque<(PeerConnId, ConnectHandle)>,
    pub to_connect_agent_cb: VecDeque<(PeerConnId, AgentMessage)>,

    pub handle_to_exchange: VecDeque<HashMap<PeerConnId, ConnectHandle>>,

    pub peer_setup_pre_agent: HashMap<PeerConnId, PeerConnConstructor>,
    pub peer_setup: HashMap<PeerConnId, PeerConnConstructor>,
    pub peer_connect_pre_agent: HashMap<PeerConnId, PeerConnConstructor>,

    pub transporter_map: HashMap<PeerConnId, &'static dyn Transporter>,
    pub peer_connected: HashMap<PeerConnId, PeerConnected>,

    conn_index: Option<u32>,
    // bitmask for which channels to connect to for each peer
    recv_connect_mask: Vec<u64>,
    send_connect_mask: Vec<u64>,
}

impl TransportConnectState {
    pub fn new(rank: usize, num_ranks: usize, num_channels: usize) -> Self {
        let recv_connect_mask = vec![0; num_ranks];
        let send_connect_mask = vec![0; num_ranks];
        TransportConnectState {
            num_channels,
            rank,
            num_ranks,
            to_setup: VecDeque::new(),
            to_setup_agent_cb: VecDeque::new(),
            to_connect: VecDeque::new(),
            to_connect_agent_cb: VecDeque::new(),
            handle_to_exchange: VecDeque::new(),
            peer_setup_pre_agent: HashMap::new(),
            peer_setup: HashMap::new(),
            peer_connect_pre_agent: HashMap::new(),
            transporter_map: HashMap::new(),
            peer_connected: HashMap::new(),
            conn_index: None,
            recv_connect_mask,
            send_connect_mask,
        }
    }
}

// Exchange connection handles with send/recv peers
// At each round, exchange handles with recv peer (rank - i)
// and send peer (rank + i)
// Connect handles received from peers are returned
pub async fn exchange_connect_handle(
    bootstrap_state: Arc<BootstrapState>,
    graph_tag: u8,
    task: PeerTransportHandleExchangeTask,
) -> Result<HashMap<PeerConnId, ConnectHandle>, TransportConnectError> {
    log::debug!("Prepare to exchange connect handle");
    let rank = task.rank;
    let num_ranks = task.num_ranks;
    let num_channels = task.num_channels;
    let conn_index = task.conn_index;
    let mut all_handles = task.handles;
    let connect_recv = task.connect_recv;
    let connect_send = task.connect_send;

    let mut all_peer_handles = HashMap::new();
    for i in 1..num_ranks {
        let bootstrap_tag = (i << 8) as u32 + graph_tag as u32;

        let mut round_handles = all_handles.pop_front().unwrap();
        let mut recv_handles = Vec::new();
        let mut send_handles = Vec::new();

        let recv_peer = (rank + num_ranks - i) % num_ranks;
        let send_peer = (rank + i) % num_ranks;
        let recv_mask = connect_recv[recv_peer];
        let send_mask = connect_send[send_peer];

        log::trace!(
            "rank={}, send_peer={}, recv_peer={}, send_mask={}, recv_mask={}, round_handle={:?}",
            rank,
            send_peer,
            recv_peer,
            send_mask,
            recv_mask,
            round_handles.keys().collect::<Vec<_>>()
        );

        for c in 0..num_channels as u32 {
            if recv_mask & (1u64 << c) > 0 {
                let conn_id = PeerConnId {
                    peer_rank: recv_peer,
                    conn_type: ConnType::Recv,
                    channel: ChannelId(c as u32),
                    conn_index,
                };
                let handle = round_handles
                    .remove(&conn_id)
                    .ok_or_else(|| TransportConnectError::ConnectionNotFound(conn_id))?;
                recv_handles.push(handle);
            }
            if send_mask & (1u64 << c) > 0 {
                let conn_id = PeerConnId {
                    peer_rank: send_peer,
                    conn_type: ConnType::Send,
                    channel: ChannelId(c as u32),
                    conn_index,
                };
                let handle = round_handles
                    .remove(&conn_id)
                    .ok_or_else(|| TransportConnectError::ConnectionNotFound(conn_id))?;
                send_handles.push(handle);
            }
        }

        let (mut peer_recv_handles, mut peer_send_handles) = if send_peer == recv_peer {
            let mut send_data = Vec::new();
            let recv_channels = recv_handles.len();
            let send_channels = send_handles.len();
            for handle in recv_handles.into_iter() {
                send_data.write_all(handle.0.as_slice())?;
            }
            for handle in send_handles.into_iter() {
                send_data.write_all(handle.0.as_slice())?;
            }
            assert_eq!(
                send_data.len() / CONNECT_HANDLE_SIZE,
                recv_channels + send_channels
            );
            bootstrap_state
                .bootstrap_send_internal(recv_peer, bootstrap_tag, send_data.as_slice())
                .await?;
            let mut recv_data = vec![0u8; CONNECT_HANDLE_SIZE * (recv_channels + send_channels)];
            bootstrap_state
                .bootstrap_recv_internal(recv_peer, bootstrap_tag, recv_data.as_mut_slice())
                .await?;

            let mut peer_recv_handles = Vec::new();
            let mut peer_send_handles = Vec::new();
            for idx in 0..send_channels {
                let data = &recv_data.as_slice()
                    [idx * CONNECT_HANDLE_SIZE..(idx + 1) * CONNECT_HANDLE_SIZE];
                let handle = ConnectHandle(data.try_into().unwrap());
                peer_send_handles.push(handle);
            }
            for idx in send_channels..(send_channels + recv_channels) {
                let data = &recv_data.as_slice()
                    [idx * CONNECT_HANDLE_SIZE..(idx + 1) * CONNECT_HANDLE_SIZE];
                let handle = ConnectHandle(data.try_into().unwrap());
                peer_recv_handles.push(handle);
            }
            (peer_recv_handles, peer_send_handles)
        } else {
            let mut send_data_recv_handles = Vec::new();
            let mut send_data_send_handles = Vec::new();
            let recv_channels = recv_handles.len();
            let send_channels = send_handles.len();
            for handle in recv_handles.into_iter() {
                send_data_recv_handles.write_all(handle.0.as_slice())?;
            }
            for handle in send_handles.into_iter() {
                send_data_send_handles.write_all(handle.0.as_slice())?;
            }
            assert_eq!(
                send_data_recv_handles.len() / CONNECT_HANDLE_SIZE,
                recv_channels
            );
            assert_eq!(
                send_data_send_handles.len() / CONNECT_HANDLE_SIZE,
                send_channels
            );
            bootstrap_state
                .bootstrap_send_internal(
                    recv_peer,
                    bootstrap_tag,
                    send_data_recv_handles.as_slice(),
                )
                .await?;
            bootstrap_state
                .bootstrap_send_internal(
                    send_peer,
                    bootstrap_tag,
                    send_data_send_handles.as_slice(),
                )
                .await?;

            let mut recv_data_send_handles = vec![0u8; CONNECT_HANDLE_SIZE * send_channels];
            let mut recv_data_recv_handles = vec![0u8; CONNECT_HANDLE_SIZE * recv_channels];
            bootstrap_state
                .bootstrap_recv_internal(
                    send_peer,
                    bootstrap_tag,
                    recv_data_send_handles.as_mut_slice(),
                )
                .await?;
            bootstrap_state
                .bootstrap_recv_internal(
                    recv_peer,
                    bootstrap_tag,
                    recv_data_recv_handles.as_mut_slice(),
                )
                .await?;

            let mut peer_recv_handles = Vec::new();
            let mut peer_send_handles = Vec::new();
            for idx in 0..recv_channels {
                let data = &recv_data_recv_handles.as_slice()
                    [idx * CONNECT_HANDLE_SIZE..(idx + 1) * CONNECT_HANDLE_SIZE];
                let handle = ConnectHandle(data.try_into().unwrap());
                peer_recv_handles.push(handle);
            }
            for idx in 0..send_channels {
                let data = &recv_data_send_handles.as_slice()
                    [idx * CONNECT_HANDLE_SIZE..(idx + 1) * CONNECT_HANDLE_SIZE];
                let handle = ConnectHandle(data.try_into().unwrap());
                peer_send_handles.push(handle);
            }
            (peer_recv_handles, peer_send_handles)
        };

        for c in 0..num_channels as u32 {
            if recv_mask & (1u64 << c) > 0 {
                let conn_id = PeerConnId {
                    peer_rank: recv_peer,
                    conn_type: ConnType::Recv,
                    channel: ChannelId(c as u32),
                    conn_index,
                };
                let handle = peer_recv_handles.remove(0);
                all_peer_handles.insert(conn_id, handle);
            }
            if send_mask & (1u64 << c) > 0 {
                let conn_id = PeerConnId {
                    peer_rank: send_peer,
                    conn_type: ConnType::Send,
                    channel: ChannelId(c as u32),
                    conn_index,
                };
                let handle = peer_send_handles.remove(0);
                all_peer_handles.insert(conn_id, handle);
            }
        }
    }
    log::trace!(
        "Rank {} of {} complete ConnectHandle exchange",
        rank,
        num_ranks
    );
    Ok(all_peer_handles)
}

fn select_transport(
    send_peer: &PeerInfo,
    recv_peer: &PeerInfo,
    profile: &CommProfile,
    catalog: &TransportCatalog,
) -> Result<&'static dyn Transporter, TransportConnectError> {
    for transporter in ALL_TRANSPORTERS.iter() {
        if transporter.can_connect(send_peer, recv_peer, profile, catalog) {
            return Ok(*transporter);
        }
    }
    Err(TransportConnectError::NoTransportFound(
        send_peer.rank,
        recv_peer.rank,
    ))
}

impl TransportConnectState {
    fn is_idle(&self) -> bool {
        self.to_setup.is_empty()
            && self.to_setup_agent_cb.is_empty()
            && self.to_connect.is_empty()
            && self.to_connect_agent_cb.is_empty()
            && self.peer_setup_pre_agent.is_empty()
            && self.peer_setup.is_empty()
            && self.peer_connect_pre_agent.is_empty()
    }

    pub fn register_connect(&mut self, conn_id: &PeerConnId) -> Result<(), TransportConnectError> {
        if !self.is_idle() {
            Err(TransportConnectError::ConnectPhaseInProgress)?;
        }
        if let Some(conn_index) = self.conn_index {
            if conn_index != conn_id.conn_index {
                return Err(TransportConnectError::ConnIndexMismatch(
                    conn_index,
                    conn_id.conn_index,
                ));
            }
        } else {
            self.conn_index = Some(conn_id.conn_index);
        }
        if self.peer_connected.contains_key(conn_id) {
            return Ok(());
        }
        match conn_id.conn_type {
            ConnType::Send => {
                self.send_connect_mask[conn_id.peer_rank] |= 1u64 << conn_id.channel.0;
            }
            ConnType::Recv => {
                self.recv_connect_mask[conn_id.peer_rank] |= 1u64 << conn_id.channel.0;
            }
        }
        Ok(())
    }

    pub fn post_setup_tasks(
        &mut self,
        peers_info: &[PeerInfo],
        profile: &CommProfile,
        catalog: &TransportCatalog,
    ) -> Result<(), TransportConnectError> {
        self.handle_to_exchange.clear();
        self.handle_to_exchange
            .resize_with(self.num_ranks - 1, || HashMap::new());
        for i in 1..self.num_ranks {
            let recv_peer = (self.rank + self.num_ranks - i) % self.num_ranks;
            let send_peer = (self.rank + i) % self.num_ranks;
            let recv_mask = self.recv_connect_mask[recv_peer];
            let send_mask = self.send_connect_mask[send_peer];

            for c in 0..self.num_channels {
                if recv_mask & (1u64 << c) > 0 {
                    let transporter = select_transport(
                        &peers_info[recv_peer],
                        &peers_info[self.rank],
                        profile,
                        catalog,
                    )?;
                    let conn_id = PeerConnId {
                        peer_rank: recv_peer,
                        conn_type: ConnType::Recv,
                        channel: ChannelId(c as u32),
                        conn_index: self.conn_index.unwrap(),
                    };
                    self.transporter_map.insert(conn_id, transporter);
                    self.to_setup.push_back(conn_id);
                }
                if send_mask & (1u64 << c) > 0 {
                    let transporter = select_transport(
                        &peers_info[self.rank],
                        &peers_info[send_peer],
                        profile,
                        catalog,
                    )?;
                    let conn_id = PeerConnId {
                        peer_rank: send_peer,
                        conn_type: ConnType::Send,
                        channel: ChannelId(c as u32),
                        conn_index: self.conn_index.unwrap(),
                    };
                    self.transporter_map.insert(conn_id, transporter);
                    self.to_setup.push_back(conn_id);
                }
            }
        }
        Ok(())
    }

    // Each synchronized task must be immediately processed by the caller
    // i.e., proxy engine<
    // and corresponding results must be registered via
    // put_peer_setup, put_peer_setup_pre_agent, ...
    pub fn get_task(&mut self) -> Result<TransportConnectTask, TransportConnectError> {
        let task = if let Some(conn_id) = self.to_setup.pop_front() {
            // First, setup tasks are posted
            // we check whether there is still setup tasks to be issued
            // some of these tasks directly completed, some require agent calls
            let transporter = *self
                .transporter_map
                .get(&conn_id)
                .ok_or_else(|| TransportConnectError::ConnectionNotFound(conn_id))?;
            let task = PeerTransportSetupTask {
                id: conn_id,
                transporter,
            };
            TransportConnectTask::PeerTransportSetup(task)
        } else if let Some((conn_id, agent_message)) = self.to_setup_agent_cb.pop_front() {
            // Check if there are setup tasks that agent has completed
            let constructor = self
                .peer_setup_pre_agent
                .remove(&conn_id)
                .ok_or_else(|| TransportConnectError::ConnectionNotFound(conn_id))?;
            let task = PeerTransportSetupAgentCbTask {
                id: conn_id,
                transporter: constructor.transporter,
                agent_message,
                setup_resources: constructor.resources,
            };
            TransportConnectTask::PeerTransportSetupAgentCb(task)
        } else if !self.peer_setup_pre_agent.is_empty() {
            // There are no outstanding setup tasks to issue,
            // but there could be setup tasks that are still waiting for agent to complete
            // we need to wait for these agent calls to complete
            // they are then enqueued to self.to_agent_cv
            TransportConnectTask::WaitingOutstandingTask
        } else if !self.handle_to_exchange.is_empty() {
            // all setup tasks have completed, we need to exchange handles
            let handles = std::mem::take(&mut self.handle_to_exchange);
            let task = PeerTransportHandleExchangeTask {
                rank: self.rank,
                num_ranks: self.num_ranks,
                num_channels: self.num_channels,
                conn_index: self.conn_index.unwrap(),
                connect_recv: self.recv_connect_mask.clone(),
                connect_send: self.send_connect_mask.clone(),
                handles,
            };
            TransportConnectTask::PeerTransportConnectHandleExchange(task)
        } else if let Some((conn_id, peer_handle)) = self.to_connect.pop_front() {
            let constructor = self
                .peer_setup
                .remove(&conn_id)
                .ok_or_else(|| TransportConnectError::ConnectionNotFound(conn_id))?;
            let task = PeerTransportConnectTask {
                id: conn_id,
                transporter: constructor.transporter,
                peer_connect_handle: peer_handle,
                setup_resources: constructor.resources,
            };
            TransportConnectTask::PeerTransportConnect(task)
        } else if let Some((conn_id, agent_message)) = self.to_connect_agent_cb.pop_front() {
            let constructor = self
                .peer_connect_pre_agent
                .remove(&conn_id)
                .ok_or_else(|| TransportConnectError::ConnectionNotFound(conn_id))?;
            let task = PeerTransportConnectAgentCbTask {
                id: conn_id,
                transporter: constructor.transporter,
                agent_message,
                transport_resources: constructor.resources,
            };
            TransportConnectTask::PeerTransportConnectAgentCb(task)
        } else if !self.peer_connect_pre_agent.is_empty()
            || !self.peer_setup.is_empty()
            || !self.peer_setup_pre_agent.is_empty()
        {
            TransportConnectTask::WaitingOutstandingTask
        } else {
            TransportConnectTask::Idle
        };
        Ok(task)
    }
}

impl TransportConnectState {
    pub fn put_peer_setup(
        &mut self,
        conn_id: &PeerConnId,
        handle: ConnectHandle,
        setup_resources: Option<AnyResources>,
    ) -> Result<(), TransportConnectError> {
        let round_idx = match conn_id.conn_type {
            ConnType::Send => (self.num_ranks + conn_id.peer_rank - self.rank) % self.num_ranks,
            ConnType::Recv => (self.num_ranks + self.rank - conn_id.peer_rank) % self.num_ranks,
        } - 1;
        self.handle_to_exchange[round_idx].insert(*conn_id, handle);
        let transporter = *self
            .transporter_map
            .get(conn_id)
            .ok_or_else(|| TransportConnectError::ConnectionNotFound(*conn_id))?;
        let constructor = PeerConnConstructor {
            transporter,
            resources: setup_resources,
        };
        self.peer_setup.insert(*conn_id, constructor);
        Ok(())
    }

    pub fn put_peer_setup_pre_agent(
        &mut self,
        conn_id: &PeerConnId,
        setup_resources: Option<AnyResources>,
    ) -> Result<(), TransportConnectError> {
        let transporter = *self
            .transporter_map
            .get(conn_id)
            .ok_or_else(|| TransportConnectError::ConnectionNotFound(*conn_id))?;
        let constructor = PeerConnConstructor {
            transporter,
            resources: setup_resources,
        };
        self.peer_setup_pre_agent.insert(*conn_id, constructor);
        Ok(())
    }

    pub fn put_peer_connect(
        &mut self,
        conn_id: &PeerConnId,
        conn_info: PeerConnInfo,
        transport_resources: AnyResources,
    ) -> Result<(), TransportConnectError> {
        let transporter = *self
            .transporter_map
            .get(conn_id)
            .ok_or_else(|| TransportConnectError::ConnectionNotFound(*conn_id))?;
        let connected = PeerConnected {
            conn_info,
            transporter,
            resources: transport_resources,
        };
        self.peer_connected.insert(*conn_id, connected);
        if self.peer_connect_pre_agent.is_empty()
            && self.to_connect.is_empty()
            && self.to_connect_agent_cb.is_empty()
        {
            // No more outstanding connect tasks, reset connect masks
            self.recv_connect_mask.clear();
            self.send_connect_mask.clear();
            self.recv_connect_mask.resize(self.num_ranks, 0);
            self.send_connect_mask.resize(self.num_ranks, 0);
        }
        Ok(())
    }

    pub fn put_peer_connect_pre_agent(
        &mut self,
        conn_id: &PeerConnId,
        transport_resources: Option<AnyResources>,
    ) -> Result<(), TransportConnectError> {
        let transporter = *self
            .transporter_map
            .get(conn_id)
            .ok_or_else(|| TransportConnectError::ConnectionNotFound(*conn_id))?;
        let constructor = PeerConnConstructor {
            transporter,
            resources: transport_resources,
        };
        self.peer_connect_pre_agent.insert(*conn_id, constructor);
        Ok(())
    }

    pub fn put_peer_agent_setup_message(
        &mut self,
        conn_id: &PeerConnId,
        agent_message: AgentMessage,
    ) -> Result<(), TransportConnectError> {
        self.to_setup_agent_cb.push_back((*conn_id, agent_message));
        Ok(())
    }

    pub fn put_peer_agent_connect_message(
        &mut self,
        conn_id: &PeerConnId,
        agent_message: AgentMessage,
    ) -> Result<(), TransportConnectError> {
        self.to_connect_agent_cb
            .push_back((*conn_id, agent_message));
        Ok(())
    }

    pub fn put_peer_connect_handles(&mut self, handles: HashMap<PeerConnId, ConnectHandle>) {
        for (conn_id, handle) in handles.into_iter() {
            self.to_connect.push_back((conn_id, handle));
        }
    }
}
