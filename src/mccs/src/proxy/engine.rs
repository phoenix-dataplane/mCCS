use std::collections::{LinkedList, HashMap};
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crossbeam::channel::{Sender, Receiver, TryRecvError};
use cuda_runtime_sys::{cudaMemcpy, cudaError};

use crate::communicator::{LocalCommunicator, PeerInfo, PeerType, CommunicatorGlobalInfo, RankInfo, CommChannel, RingPattern};
use crate::daemon::DaemonId;
use crate::proxy::ops::SetupTransport;
use crate::resources::GlobalResources;
use crate::transport::connector::ConnectorIdentifier;
use crate::transport::hmem::config::HostMemTptConfig;
use crate::transport::hmem::ops::{HostMemTptEndpoint, hmem_sender_setup, hmem_receiver_setup, hmem_sender_connect, hmem_receiver_connect};
use super::command::{ProxyCommand, ProxyCompletion, CommandEndpointProxy};
use super::ops::{ProxyOp, LocalCommunicatorBootstrap};
use super::DeviceInfo;

pub struct ProxyEngine {
    pub device_info: DeviceInfo,
    pub outstanding_ops: LinkedList<(ProxyOp, OpStatus)>,
    pub daemon_endpoint_rx: Receiver<CommandEndpointProxy>,
    pub daemon_command_rx: Vec<(DaemonId, Receiver<ProxyCommand>)>,
    pub daemon_completion_tx: Vec<(DaemonId, Sender<ProxyCompletion>)>,
    pub communicators: HashMap<u32, LocalCommunicator>,
    pub global_resources: Arc<GlobalResources>,
    pub hmem_senders: HashMap<ConnectorIdentifier, HostMemTptEndpoint>,
    pub hmem_receivers: HashMap<ConnectorIdentifier, HostMemTptEndpoint>,
}

impl ProxyEngine {
    pub fn mainloop(&mut self) {
        loop {
            if rand::random::<u8>() < 10 {
                self.check_new_endpoint();
            }
            self.check_command();
            self.process_ops();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineStatus {
    Progress(usize),
}

use EngineStatus::Progress;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpStatus {
    Init,
    InProgress,
    Completed,
}

impl ProxyEngine {
    fn check_new_endpoint(&mut self) {
        match self.daemon_endpoint_rx.try_recv() {
            Ok(endpoint) => {
                self.daemon_command_rx.push((endpoint.daemon_id, endpoint.command_rx));
                self.daemon_completion_tx.push((endpoint.daemon_id, endpoint.completion_tx));
            },
            Err(_) => {},
        }
    }

    fn check_command(&mut self) -> EngineStatus {
        let mut progress = 0;
        self.daemon_command_rx.drain_filter(|(_, cmd_rx)| {
            match cmd_rx.try_recv() {
                Ok(cmd) => {
                    match cmd {
                        ProxyCommand::InitCommunicator(init) => {
                            let init = LocalCommunicatorBootstrap {
                                daemon_id: init.daemon_id,
                                communicator_id: init.communicator_id,
                                rank: init.rank,
                                peers_info: HashMap::with_capacity(init.num_ranks),
                                num_ranks: init.num_ranks,
                            };
                            let op = ProxyOp::InitCommunicator(init);
                            self.outstanding_ops.push_back((op, OpStatus::Init));
                        },
                        ProxyCommand::AllGather(all_gather) => {
                            let all_gather = super::ops::AllGather {
                                daemon_id: all_gather.daemon_id,
                                communicator_id: all_gather.communicator_id,
                                send_buf: all_gather.send_buf_addr as *mut u8,
                                recv_buf: all_gather.recv_buf_addr as *mut u8,
                                size: all_gather.size,
                                step: 0,
                                recv_completed: false,
                                send_completed: false,
                            };
                            let op = ProxyOp::AllGather(all_gather);
                            self.outstanding_ops.push_back((op, OpStatus::Init));
                        },
                    }
                    progress += 1;
                    false
                },
                Err(TryRecvError::Empty) => false,
                Err(TryRecvError::Disconnected) => true,
            }
        });
        Progress(progress)
    }

    fn process_ops(&mut self) -> EngineStatus {
        // TBD: process only one op per engine mainloop?
        // minimize amount of work per op?
        let mut progress = 0;
        let mut new_ops = Vec::new();

        self.outstanding_ops.drain_filter(|(op, status)| {
            assert_ne!(*status, OpStatus::Completed);
            match op {
                ProxyOp::InitCommunicator(bootstrap) => {
                    let mut comm_global = self.global_resources.communicators.entry(bootstrap.communicator_id) 
                        .or_insert_with(|| {
                            CommunicatorGlobalInfo {
                                communicator_id: bootstrap.communicator_id,
                                num_ranks: bootstrap.num_ranks,
                                ranks_info: vec![None; bootstrap.num_ranks],
                            }
                        });
                    if *status == OpStatus::Init {
                        let local_peer_info = PeerInfo {
                            peer_type: PeerType::Local,
                            rank: bootstrap.rank,
                            cuda_device_idx: self.device_info.cuda_device_idx,
                            cuda_comp_cap: self.device_info.cuda_comp_cap,
                        };
                        let local_rank_info = RankInfo {
                            rank: bootstrap.rank,
                            // TBD
                            host: 0,
                            cuda_device_idx: local_peer_info.cuda_device_idx,
                            cuda_comp_cap: local_peer_info.cuda_comp_cap,
                        };
                        comm_global.ranks_info[bootstrap.rank] = Some(local_rank_info);
                        bootstrap.peers_info.insert(bootstrap.rank, local_peer_info);
                        *status = OpStatus::InProgress;
                    }
                    for rank_info in comm_global.ranks_info.iter().filter_map(|x| x.as_ref()) {
                        match bootstrap.peers_info.entry(rank_info.rank) {
                            Entry::Occupied(_) => { },
                            Entry::Vacant(entry) => { 
                                let peer_info = PeerInfo {
                                    peer_type: PeerType::IntraNode,
                                    rank: rank_info.rank,
                                    cuda_device_idx: rank_info.cuda_device_idx,
                                    cuda_comp_cap: rank_info.cuda_comp_cap,
                                };
                                entry.insert(peer_info);
                            },
                        }
                    }
                    if bootstrap.peers_info.len() == bootstrap.num_ranks {
                        let mut local_rank = 0;
                        let mut local_num_ranks = 0;
                        let mut peers = vec![None; bootstrap.num_ranks];
                        for (peer_rank, peer_info) in bootstrap.peers_info.iter_mut() {
                            peers[*peer_rank] = Some(peer_info.clone());
                        }
                        let peers = peers.into_iter().map(|x| x.unwrap()).collect::<Vec<_>>();
                        for peer in peers.iter() {
                            if peer.rank == bootstrap.rank {
                                local_rank = local_num_ranks;
                            } 
                            if peer.peer_type == PeerType::IntraNode {
                                local_num_ranks += 1;
                            }
                        }

                        let ring = RingPattern {
                            prev_rank: (bootstrap.rank - 1 + bootstrap.num_ranks) % bootstrap.num_ranks,
                            next_rank: (bootstrap.rank + 1) % bootstrap.num_ranks,
                        };
                        let channel = CommChannel {
                            id: 0,
                            ring,
                        };
                        let comm = LocalCommunicator {
                            communicator_id: bootstrap.communicator_id,
                            daemon_id: bootstrap.daemon_id,
                            rank: bootstrap.rank,
                            n_ranks: bootstrap.num_ranks,
                            cuda_device_idx: self.device_info.cuda_device_idx,
                            local_rank: local_rank,
                            local_ranks: local_num_ranks,
                            channels: vec![channel],
                            peers_info: peers,
                        };
                        self.communicators.insert(comm.communicator_id, comm);
                        let setup_transport = ProxyOp::InitCommSetupTransport(
                            SetupTransport {
                                daemon_id: bootstrap.daemon_id,
                                communicator_id: bootstrap.communicator_id,
                            }
                        );
                        new_ops.push(setup_transport);
                        *status = OpStatus::Completed;
                    }
                },
                ProxyOp::InitCommSetupTransport(setup) => {
                    let comm = self.communicators.get(&setup.communicator_id).unwrap();
                    let prev = comm.channels[0].ring.prev_rank;
                    let next = comm.channels[0].ring.next_rank;
                    let send_id = ConnectorIdentifier {
                        communicator_id: comm.communicator_id,
                        sender_rank: comm.rank,
                        receiver_rank: next,
                        channel: 0,
                    };
                    let recv_id = ConnectorIdentifier {
                        communicator_id: comm.communicator_id,
                        sender_rank: prev,
                        receiver_rank: comm.rank,
                        channel: 0,
                    };
                    if *status == OpStatus::Init {
                        let config = HostMemTptConfig {
                            buff_sizes: [8192],
                            locality: crate::transport::hmem::config::MemLocality::SenderSide,
                        };
                        hmem_sender_setup(&self.global_resources.transport_setup, send_id, &config);
                        hmem_receiver_setup(&self.global_resources.transport_setup, recv_id, &config);
                        *status = OpStatus::InProgress;
                    }
                    let mut completed = true;
                    if !self.hmem_senders.contains_key(&send_id) {
                        match hmem_sender_connect(&self.global_resources.transport_setup, &send_id) {
                            Ok(sender) => { self.hmem_senders.insert(send_id, sender); }, 
                            Err(_) => { completed = false; },
                        }
                    }
                    if !self.hmem_receivers.contains_key(&recv_id) {
                        match hmem_receiver_connect(&self.global_resources.transport_setup, &recv_id) {
                            Ok(receiver) => { self.hmem_receivers.insert(recv_id, receiver); }, 
                            Err(_) => { completed = false; },
                        }
                    }
                    if completed {
                        self.daemon_completion_tx.iter_mut()
                            .find(|(id, _)| *id == setup.daemon_id)
                            .unwrap()
                            .1
                            .send(ProxyCompletion::InitCommunicator).unwrap();
                        *status = OpStatus::Completed;
                    }
                },
                ProxyOp::AllGather(all_gather) => {
                    assert!(all_gather.size == 4096);
                    assert!(all_gather.step < 2);
                    *status = OpStatus::InProgress;
                    let comm = self.communicators.get(&all_gather.communicator_id).unwrap();
                    let send_chunk = (comm.rank - all_gather.step as usize + comm.n_ranks) % comm.n_ranks;
                    let recv_chunk = (send_chunk - 1 + comm.n_ranks) % comm.n_ranks;
                    let prev = comm.channels[0].ring.prev_rank;
                    let next = comm.channels[0].ring.next_rank;
                    let send_id = ConnectorIdentifier {
                        communicator_id: comm.communicator_id,
                        sender_rank: comm.rank,
                        receiver_rank: next,
                        channel: 0,
                    };
                    let recv_id = ConnectorIdentifier {
                        communicator_id: comm.communicator_id,
                        sender_rank: prev,
                        receiver_rank: comm.rank,
                        channel: 0,
                    }; 
                    let sender = self.hmem_senders.get(&send_id).unwrap();
                    let receiver = self.hmem_receivers.get(&recv_id).unwrap();

                    let size = all_gather.size;
                    if !all_gather.send_completed {
                        let src = unsafe { all_gather.send_buf.add(send_chunk * size) };
                        let dst = unsafe { sender.connector.info.bufs[0].as_ptr().add(send_chunk * size) };
                        unsafe { 
                            let err = cudaMemcpy(dst as *mut _, src as *mut _, size, cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost);
                            if err != cudaError::cudaSuccess {
                                panic!("cudaMemcpy failed")
                            }
                            (*sender.connector.info.tail).fetch_add(1, Ordering::Relaxed);
                        };
                        all_gather.send_completed = true;
                    }
                    if !all_gather.recv_completed {
                        unsafe { 
                            let tail = (*receiver.connector.info.tail).load(Ordering::Relaxed);
                            if tail > all_gather.step as u64 {
                                let src = receiver.connector.info.bufs[0].as_ptr().add(recv_chunk * size);
                                let dst = all_gather.recv_buf.add(recv_chunk * size);
                                let err = cudaMemcpy(dst as *mut _, src as *mut _, size, cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice);
                                if err != cudaError::cudaSuccess {
                                    panic!("cudaMemcpy failed")
                                }
                                (*receiver.connector.info.head).fetch_add(1, Ordering::Relaxed);
                                all_gather.recv_completed = true;
                            }
                        }
                    }
                    if all_gather.send_completed && all_gather.recv_completed {
                        all_gather.step += 1;
                        all_gather.send_completed = false;
                        all_gather.recv_completed = false;
                        if all_gather.step as usize == comm.n_ranks {
                            self.daemon_completion_tx.iter_mut()
                                .find(|(id, _)| *id == all_gather.daemon_id)
                                .unwrap()
                                .1
                                .send(ProxyCompletion::AllGather).unwrap();
                            *status = OpStatus::Completed;
                        }
                    }
                },
            }
            if *status == OpStatus::Completed { progress += 1; }
            *status == OpStatus::Completed
            
        });

        for op in new_ops {
            self.outstanding_ops.push_back((op, OpStatus::Init));
        }

        Progress(progress)
    }
}