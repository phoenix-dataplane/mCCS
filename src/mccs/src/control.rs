use std::collections::HashMap;
use std::fs;
use std::io;
use std::iter::zip;
use std::net::{IpAddr, Ipv4Addr};
use std::os::unix::net::UCred;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use std::time::{Duration, Instant};

use anyhow::anyhow;
use cuda_runtime_sys::cudaMalloc;
use cuda_runtime_sys::cudaMemcpy;
use cuda_runtime_sys::cudaMemcpyKind;
use dashmap::DashMap;
use nix::libc;

use cuda_runtime_sys::cudaError;
use cuda_runtime_sys::cudaGetDeviceCount;
use cuda_runtime_sys::cudaSetDevice;
use ipc::customer::ShmCustomer;
use ipc::unix::DomainSocket;

use crate::comm::CommunicatorId;
use crate::config::Config;
use crate::cuda_warning;
use crate::daemon::DaemonId;
use crate::exchange::command::{ExchangeCommand, ExchangeCompletion};
use crate::exchange::engine::ExchangeEngine;
use crate::message::{ControlCommand, ControlRequest};
use crate::proxy::command::AllGatherRequest;
use crate::proxy::command::InitCommunicator;
use crate::proxy::command::ProxyCommand;
use crate::proxy::command::ProxyCompletion;
use crate::proxy::engine::ProxyEngine;
use crate::proxy::engine::ProxyResources;
use crate::proxy::message::ProxyPeerMessage;
use crate::proxy::DeviceInfo;
use crate::registry::GlobalRegistry;
use crate::transport;
use crate::transport::catalog::TransportCatalog;
use crate::transport::delegator::TransportDelegator;
use crate::transport::engine::TransportEngine;
use crate::transport::engine::TransportEngineId;
use crate::transport::engine::TransportEngineResources;
use crate::transport::message::TransportEngineReply;
use crate::transport::message::TransportEngineRequest;
use crate::transport::net::config::NetTransportConfig;
use crate::transport::queue::TransrportOpQueue;
use crate::transport::shm::config::ShmConfig;
use crate::utils::duplex_chan::DuplexChannel;
use crate::utils::pool::WorkPool;

pub struct Control {
    sock: DomainSocket,
    config: Config,
    daemon_cnt: DaemonId,
    proxy_channels: Vec<DuplexChannel<ControlCommand, ControlRequest>>,
}

impl Control {
    pub fn new(config: Config) -> Self {
        let mccs_prefix = &config.control.prefix;
        fs::create_dir_all(mccs_prefix)
            .unwrap_or_else(|e| panic!("Failed to create directory for {mccs_prefix:?}: {e}"));

        let mccs_path = mccs_prefix.join(&config.control.path);
        if mccs_path.exists() {
            fs::remove_file(&mccs_path).expect("remove_file");
        }
        let sock = DomainSocket::bind(&mccs_path)
            .unwrap_or_else(|e| panic!("Cannot bind domain socket at {mccs_path:?}: {e}"));

        sock.set_read_timeout(Some(Duration::from_millis(1)))
            .expect("set_read_timeout");
        sock.set_write_timeout(Some(Duration::from_millis(1)))
            .expect("set_write_timeout");

        let transport_delegator = TransportDelegator::new();
        let transport_catalog = Arc::new(TransportCatalog::new());
        let shm_config = ShmConfig {
            locality: crate::transport::shm::config::ShmLocality::Sender,
            use_memcpy_send: false,
            use_memcpy_recv: false,
        };
        transport_catalog.register_config(String::from("ShmTransport"), shm_config);
        let registry = GlobalRegistry {
            transport_delegator,
            transport_catalog,
        };
        let registry = Arc::new(registry);

        // FIXME: problematic, should be checked whenever encountered a bug with inter-host
        // let sock_addr = "127.0.0.1:8000".parse().unwrap();

        // let chan = Self::create_proxies(registry, sock_addr).expect("Create proxies failed");

        // Control {
        //     sock,
        //     config,
        //     daemon_cnt: 0,
        //     proxy_channels: chan,
        // }
        todo!()
    }

    // pub fn mainloop(&mut self, exit_flag: &AtomicBool) -> anyhow::Result<()> {
    //     let mut buf = vec![0u8; 65536];
    //     while !exit_flag.load(Ordering::Relaxed) {
    //         match self.sock.recv_with_credential_from(buf.as_mut_slice()) {
    //             Ok((size, sender, cred)) => {
    //                 if let Some(cred) = cred {
    //                     if let Err(_e) = self.dispatch(&mut buf[..size], &sender, &cred) {
    //                         // log
    //                     }
    //                 } else {
    //                     // log
    //                 }
    //             }
    //             Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {}
    //             Err(_e) => {
    //                 if exit_flag.load(Ordering::Relaxed) {
    //                     break;
    //                 }
    //                 // log
    //             }
    //         }
    //     }
    //     Ok(())
    // }

    // fn create_proxies(
    //     registry: Arc<GlobalRegistry>,
    //     sock_addr: std::net::SocketAddr,
    // ) -> anyhow::Result<Vec<DuplexChannel<ControlCommand, ControlRequest>>> {
    //     // we don't allow device hot-plug, so we create connected proxies in advance
    //     let device_cnt = {
    //         let mut i = 0;
    //         cuda_warning!(unsafe { cudaGetDeviceCount(&mut i) });
    //         i as i32 as usize
    //     };
    //     // proxies <--> proxies
    //     let mut inter_senders = vec![vec![]; device_cnt];
    //     let mut inter_receivers = vec![];
    //     for _ in 0..device_cnt {
    //         let (sender, receiver) = crossbeam::channel::unbounded();
    //         inter_receivers.push(receiver);
    //         inter_senders
    //             .iter_mut()
    //             .for_each(|x| x.push(sender.clone()));
    //     }
    //     // control <--> proxies
    //     let mut control_command_local = vec![];
    //     let mut control_command_proxy = vec![];
    //     for _ in 0..device_cnt {
    //         let (local_side, proxy_side) = DuplexChannel::new_unbound_pair();
    //         control_command_local.push(local_side);
    //         control_command_proxy.push(proxy_side)
    //     }

    //     zip(control_command_proxy, zip(inter_senders, inter_receivers))
    //         .enumerate()
    //         .map(
    //             |(idx, (chan, (inter_sender, inter_receiver)))| ProxyResources {
    //                 device_info: DeviceInfo {
    //                     host: HostIdent(sock_addr),
    //                     cuda_device_idx: idx as i32,
    //                 },
    //                 control_chan: chan,
    //                 daemon_tx: Default::default(),
    //                 daemon_rx: vec![],
    //                 proxy_peer_tx: inter_sender,
    //                 proxy_peer_rx: inter_receiver,
    //                 comms_init: Default::default(),
    //                 communicators: Default::default(),
    //                 global_registry: registry.clone(),
    //                 transport_engines_tx: Default::default(),
    //                 transport_engines_rx: vec![],
    //                 transport_submission_pool: Default::default(),
    //             },
    //         )
    //         .enumerate()
    //         .for_each(|(cuda_idx, res)| {
    //             let mut proxy = ProxyEngine {
    //                 resources: res,
    //                 ops: WorkPool::new(),
    //             };
    //             std::thread::spawn(move || {
    //                 unsafe {
    //                     let error = cudaSetDevice(cuda_idx as i32);
    //                     if error != cudaError::cudaSuccess {
    //                         panic!("cudaSetDevice");
    //                     }
    //                 }
    //                 proxy.mainloop();
    //             });
    //         });

    //     Ok(control_command_local)
    // }

    // fn dispatch(
    //     &mut self,
    //     buf: &mut [u8],
    //     sender: &SocketAddr,
    //     _cred: &UCred,
    // ) -> anyhow::Result<()> {
    //     use ipc::control;
    //     let msg: control::Request = bincode::deserialize(buf).unwrap();
    //     match msg {
    //         control::Request::NewClient => {
    //             let client_path = sender
    //                 .as_pathname()
    //                 .ok_or_else(|| anyhow!("peer is unnamed, something is wrong"))?;

    //             let uuid = uuid::Uuid::new_v4();
    //             let instance_name = format!("{}-{}.sock", self.config.mccs_daemon_basename, uuid);
    //             let engine_path = self.config.mccs_daemon_prefix.join(instance_name);

    //             // create customer stub
    //             let customer = ShmCustomer::accept(&self.sock, client_path, engine_path)?;

    //             let daemon_id = self.daemon_cnt;
    //             let num_devices = self.proxy_channels.len();
    //             let mut daemon_channels = Vec::with_capacity(num_devices);

    //             for device_idx in 0..num_devices {
    //                 let endpoint_tx = &mut self.proxy_channels[device_idx].tx;
    //                 let (daemon_side, proxy_side) = DuplexChannel::new_unbound_pair();
    //                 let proxy_endpoint = ControlCommand::NewDaemon {
    //                     id: daemon_id,
    //                     chan: proxy_side,
    //                 };
    //                 endpoint_tx.send(proxy_endpoint).unwrap();
    //                 daemon_channels.push(daemon_side);
    //             }

    //             let mut engine = crate::daemon::engine::DaemonEngine {
    //                 id: daemon_id,
    //                 proxy_chan: daemon_channels,
    //                 device_mem: HashMap::new(),
    //                 comm_delegation: HashMap::new(),
    //                 customer,
    //                 mem_counter: 0,
    //             };
    //             std::thread::spawn(move || {
    //                 engine.mainloop();
    //             });
    //             self.daemon_cnt += 1;

    //             Ok(())
    //         }
    //     }
    // }
}

impl Control {
    // #[allow(dead_code)]
    // fn test(&mut self) -> anyhow::Result<()> {
    //     let start_test = Instant::now();
    //     let mut num_devices = 0;
    //     unsafe {
    //         let error = cudaGetDeviceCount(&mut num_devices as *mut _);
    //         if error != cudaError::cudaSuccess {
    //             panic!("cudaGetDeviceCount");
    //         }
    //     }
    //     let transport_delegator = TransportDelegator::new();
    //     let transport_catalog = Arc::new(TransportCatalog::new());
    //     let shm_config = ShmConfig {
    //         locality: crate::transport::shm::config::ShmLocality::Sender,
    //         use_memcpy_send: false,
    //         use_memcpy_recv: false,
    //     };
    //     transport_catalog.register_config(String::from("ShmTransport"), shm_config);
    //     let registry = GlobalRegistry {
    //         communicators: DashMap::new(),
    //         transport_delegator,
    //         transport_catalog,
    //     };
    //     let registry = Arc::new(registry);
    //     let (proxy_0_tx, proxy_0_rx) = crossbeam::channel::unbounded();
    //     let (proxy_1_tx, proxy_1_rx) = crossbeam::channel::unbounded();
    //     let (daemon_0_cmd_tx, daemon_0_cmd_rx) = crossbeam::channel::unbounded();
    //     let (daemon_0_comp_tx, daemon_0_comp_rx) = crossbeam::channel::unbounded();
    //     let (daemon_1_cmd_tx, daemon_1_cmd_rx) = crossbeam::channel::unbounded();
    //     let (daemon_1_comp_tx, daemon_1_comp_rx) = crossbeam::channel::unbounded();

    //     let sock_addr = std::net::SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8000);
    //     let dev_info = DeviceInfo {
    //         host: HostIdent(sock_addr),
    //         cuda_device_idx: 0,
    //     };
    //     let (control_req_tx, _control_req_rx) = crossbeam::channel::unbounded();
    //     let (_control_notify_tx, control_notify_rx) = crossbeam::channel::unbounded();
    //     let mut daemon_tx = HashMap::new();
    //     daemon_tx.insert(0, daemon_0_comp_tx);
    //     let daemon_rx = vec![(0, daemon_0_cmd_rx)];
    //     let proxy_peer_tx = vec![proxy_0_tx.clone(), proxy_1_tx.clone()];
    //     let proxy_0_resources = ProxyResources {
    //         device_info: dev_info,
    //         control_chan: DuplexChannel {
    //             tx: control_req_tx,
    //             rx: control_notify_rx,
    //         },
    //         daemon_tx,
    //         daemon_rx,
    //         proxy_peer_tx,
    //         proxy_peer_rx: proxy_0_rx,
    //         comms_init: HashMap::new(),
    //         communicators: HashMap::new(),
    //         global_registry: Arc::clone(&registry),
    //         transport_engines_tx: HashMap::new(),
    //         transport_engines_rx: Vec::new(),
    //         transport_submission_pool: HashMap::new(),
    //     };
    //     let mut proxy_0 = ProxyEngine {
    //         resources: proxy_0_resources,
    //         ops: WorkPool::new(),
    //     };

    //     let (control_req_tx, _control_req_rx) = crossbeam::channel::unbounded();
    //     let (_control_notify_tx, control_notify_rx) = crossbeam::channel::unbounded();
    //     let mut daemon_tx = HashMap::new();
    //     daemon_tx.insert(0, daemon_1_comp_tx);
    //     let daemon_rx = vec![(0, daemon_1_cmd_rx)];
    //     let proxy_peer_tx = vec![proxy_0_tx, proxy_1_tx];
    //     let dev_info = DeviceInfo {
    //         host: HostIdent(sock_addr),
    //         cuda_device_idx: 1,
    //     };
    //     let proxy_1_resources = ProxyResources {
    //         device_info: dev_info,
    //         control_chan: DuplexChannel {
    //             tx: control_req_tx,
    //             rx: control_notify_rx,
    //         },
    //         daemon_tx,
    //         daemon_rx,
    //         proxy_peer_tx,
    //         proxy_peer_rx: proxy_1_rx,
    //         comms_init: HashMap::new(),
    //         communicators: HashMap::new(),
    //         global_registry: Arc::clone(&registry),
    //         transport_engines_tx: HashMap::new(),
    //         transport_engines_rx: Vec::new(),
    //         transport_submission_pool: HashMap::new(),
    //     };
    //     let mut proxy_1 = ProxyEngine {
    //         resources: proxy_1_resources,
    //         ops: WorkPool::new(),
    //     };
    //     std::thread::spawn(move || {
    //         unsafe {
    //             let error = cudaSetDevice(0);
    //             if error != cudaError::cudaSuccess {
    //                 panic!("cudaSetDevice");
    //             }
    //         }
    //         proxy_0.mainloop();
    //     });
    //     std::thread::spawn(move || {
    //         unsafe {
    //             let error = cudaSetDevice(1);
    //             if error != cudaError::cudaSuccess {
    //                 panic!("cudaSetDevice");
    //             }
    //         }
    //         proxy_1.mainloop();
    //     });
    //     let cmd = InitCommunicator {
    //         communicator_id: CommunicatorId(0),
    //         rank: 0,
    //         num_ranks: 2,
    //     };
    //     let cmd = ProxyCommand::InitCommunicator(cmd);
    //     daemon_0_cmd_tx.send(cmd).unwrap();
    //     let cmd = InitCommunicator {
    //         communicator_id: CommunicatorId(0),
    //         rank: 1,
    //         num_ranks: 2,
    //     };
    //     let cmd = ProxyCommand::InitCommunicator(cmd);
    //     daemon_1_cmd_tx.send(cmd).unwrap();
    //     let comp = daemon_0_comp_rx.recv().unwrap();
    //     match comp {
    //         ProxyCompletion::InitCommunicator => (),
    //         ProxyCompletion::AllGather(_) => panic!(),
    //     }
    //     let comp = daemon_1_comp_rx.recv().unwrap();
    //     match comp {
    //         ProxyCompletion::InitCommunicator => (),
    //         ProxyCompletion::AllGather(_) => panic!(),
    //     }

    //     unsafe {
    //         let error = cudaSetDevice(0);
    //         if error != cudaError::cudaSuccess {
    //             panic!("cudaSetDevice");
    //         }
    //     }
    //     const BUFFER_SIZE: usize = 1024 * 1024 * 512;
    //     let dev_buf_0 = unsafe {
    //         let mut dev_ptr = std::ptr::null_mut();
    //         cudaMalloc(&mut dev_ptr, BUFFER_SIZE);
    //         dev_ptr
    //     };
    //     log::info!("dev_buf_0: {:p} of size {BUFFER_SIZE} bytes", dev_buf_0);
    //     let mut buf = vec![1883i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()];
    //     buf.extend(vec![0i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]);
    //     log::info!(
    //         "Initialize resource for {:.3} KB: {} ms",
    //         (BUFFER_SIZE as f64) / 1024.0,
    //         start_test.elapsed().as_millis()
    //     );
    //     let before_memcpy = Instant::now();
    //     unsafe {
    //         cudaMemcpy(
    //             dev_buf_0,
    //             buf.as_ptr() as *const _,
    //             BUFFER_SIZE,
    //             cudaMemcpyKind::cudaMemcpyHostToDevice,
    //         )
    //     };

    //     unsafe {
    //         let error = cudaSetDevice(1);
    //         if error != cudaError::cudaSuccess {
    //             panic!("cudaSetDevice");
    //         }
    //     }
    //     let dev_buf_1 = unsafe {
    //         let mut dev_ptr = std::ptr::null_mut();
    //         cudaMalloc(&mut dev_ptr, BUFFER_SIZE);
    //         dev_ptr
    //     };
    //     log::info!("dev_buf_1: {:p} of size {BUFFER_SIZE} bytes", dev_buf_1);
    //     let mut buf = vec![0i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()];
    //     buf.extend(vec![2042i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]);
    //     unsafe {
    //         cudaMemcpy(
    //             dev_buf_1,
    //             buf.as_ptr() as *const _,
    //             BUFFER_SIZE,
    //             cudaMemcpyKind::cudaMemcpyHostToDevice,
    //         )
    //     };
    //     log::info!("Memory copy: {} ms", before_memcpy.elapsed().as_millis());
    //     let before_allgather = Instant::now();
    //     let cmd = AllGatherRequest {
    //         communicator_id: CommunicatorId(0),
    //         send_buf_addr: dev_buf_0 as usize,
    //         recv_buf_addr: dev_buf_0 as usize,
    //         size: BUFFER_SIZE / 2,
    //         app_ipc_event_handle: todo!(),
    //     };
    //     let cmd = ProxyCommand::AllGather(cmd);
    //     daemon_0_cmd_tx.send(cmd).unwrap();
    //     let cmd = AllGatherRequest {
    //         communicator_id: CommunicatorId(0),
    //         send_buf_addr: dev_buf_1 as usize + BUFFER_SIZE / 2,
    //         recv_buf_addr: dev_buf_1 as usize,
    //         size: BUFFER_SIZE / 2,
    //         app_ipc_event_handle: todo!(),
    //     };
    //     let cmd = ProxyCommand::AllGather(cmd);
    //     daemon_1_cmd_tx.send(cmd).unwrap();

    //     let comp = daemon_0_comp_rx.recv().unwrap();
    //     match comp {
    //         ProxyCompletion::InitCommunicator => panic!(),
    //         ProxyCompletion::AllGather(_) => (),
    //     }
    //     let comp = daemon_1_comp_rx.recv().unwrap();
    //     match comp {
    //         ProxyCompletion::InitCommunicator => panic!(),
    //         ProxyCompletion::AllGather(_) => (),
    //     }
    //     log::info!("All Gather: {} ms", before_allgather.elapsed().as_millis());

    //     let mut buf = vec![0; BUFFER_SIZE];
    //     unsafe {
    //         let err = cudaMemcpy(
    //             buf.as_mut_ptr() as *mut _,
    //             dev_buf_1,
    //             BUFFER_SIZE,
    //             cudaMemcpyKind::cudaMemcpyDeviceToHost,
    //         );
    //         if err != cudaError::cudaSuccess {
    //             panic!("cudaMemcpy failed");
    //         }
    //     };
    //     assert_eq!(buf[0], 1883);
    //     assert_eq!(buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()], 2042);
    //     log::info!("Pass data check");
    //     Ok(())
    // }

    fn start_test_proxy(
        cuda_device: i32,
        sock_addr: std::net::SocketAddr,
        daemon_tx: HashMap<DaemonId, crossbeam::channel::Sender<ProxyCompletion>>,
        daemon_rx: Vec<(DaemonId, crossbeam::channel::Receiver<ProxyCommand>)>,
        exchange_tx: crossbeam::channel::Sender<ExchangeCommand>,
        exchange_rx: crossbeam::channel::Receiver<ExchangeCompletion>,
        transport_engines_tx: HashMap<TransportEngineId, crossbeam::channel::Sender<TransportEngineRequest>>,
        transport_engines_rx: Vec<(TransportEngineId, crossbeam::channel::Receiver<TransportEngineReply>)>,
        global_registry: Arc<GlobalRegistry>,
    ) -> anyhow::Result<()> {
        let dev_info = DeviceInfo {
            host: sock_addr,
            cuda_device_idx: cuda_device,
        };
        let (control_req_tx, _control_req_rx) = crossbeam::channel::unbounded();
        let (_control_notify_tx, control_notify_rx) = crossbeam::channel::unbounded();

        let proxy_resources = ProxyResources {
            device_info: dev_info,
            control_chan: DuplexChannel {
                tx: control_req_tx,
                rx: control_notify_rx,
            },
            daemon_tx,
            daemon_rx,
            comms_init: HashMap::new(),
            communicators: HashMap::new(),
            global_registry,
            transport_engines_tx,
            transport_engines_rx,
            transport_submission_pool: HashMap::new(),
            exchange_tx,
            exchange_rx,
            task_submit_pool: Vec::new(),
        };
        let mut proxy = ProxyEngine {
            resources: proxy_resources,
            ops: WorkPool::new(),
            async_tasks: WorkPool::new(),
        };
        std::thread::spawn(move || {
            unsafe {
                let error = cudaSetDevice(cuda_device);
                if error != cudaError::cudaSuccess {
                    panic!("cudaSetDevice");
                }
            }
            proxy.mainloop();
        });
        Ok(())
    }
    
    pub fn dist_test(host: usize) {
        use crate::transport::net::provider::NetProvierWrap;
        crate::transport::net::provider::RDMA_TRANSPORT.init().unwrap();
        let transport_delegator = TransportDelegator::new();
        transport_delegator.active_connections.insert(0, vec![(0, 0)]);
        let transport_engine_id = TransportEngineId {
            cuda_device_idx: 0,
            index: 0,
        };
        let (transport_cmd_tx, transport_cmd_rx) = crossbeam::channel::unbounded();
        let (transport_comp_tx, transport_comp_rx) = crossbeam::channel::unbounded();
        let mut transport_engines_tx = HashMap::new();
        transport_engines_tx.insert(transport_engine_id, transport_cmd_tx);
        let transport_engines_rx = vec![(transport_engine_id, transport_comp_rx)];

        let transport_catalog = TransportCatalog::new();
        let net_config = NetTransportConfig {
            gdr_enable: false,
            gdr_copy_sync_enable: false,
            gdr_copy_flush_enable: false,
        };
        transport_catalog.register_config(String::from("NetTransport"), net_config);
        let registry = GlobalRegistry {
            transport_delegator,
            transport_catalog: Arc::new(transport_catalog),
        };
        let registry = Arc::new(registry);
        
        let resources = TransportEngineResources {
            agent_setup: HashMap::new(),
            agent_connected: HashMap::new(),
            proxy_chan: vec![DuplexChannel {
                tx: transport_comp_tx,
                rx: transport_cmd_rx,
            }],
            global_registry: Arc::clone(&registry),
        };
        let mut transport_engine = TransportEngine {
            id: transport_engine_id,
            resources,
            async_tasks: WorkPool::new(),
            op_queue: TransrportOpQueue::new(),
        };
        std::thread::spawn(move || {
            unsafe {
                let error = cudaSetDevice(0);
                if error != cudaError::cudaSuccess {
                    panic!("cudaSetDevice");
                }
            }
            transport_engine.mainloop();
        }); 

        let (daemon_cmd_tx, daemon_cmd_rx) = crossbeam::channel::unbounded();
        let (daemon_comp_tx, daemon_comp_rx) = crossbeam::channel::unbounded();
        let (exchange_cmd_tx, exchange_cmd_rx) = crossbeam::channel::unbounded();
        let (exchange_comp_tx, exchange_comp_rx) = crossbeam::channel::unbounded();

        let sock_addr = if host == 0 {
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 211, 66)), 5000)
        } else {
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 211, 194)), 5000)
        };
        let mut daemon_tx = HashMap::new();
        daemon_tx.insert(0, daemon_comp_tx);
        let daemon_rx = vec![(0, daemon_cmd_rx)];
        Self::start_test_proxy(
            0,
            sock_addr.clone(),
            daemon_tx,
            daemon_rx,
            exchange_cmd_tx,
            exchange_comp_rx,
            transport_engines_tx,
            transport_engines_rx,
            registry.clone(),
        ).unwrap();

        let proxy_tx = vec![exchange_comp_tx];
        let proxy_rx = vec![exchange_cmd_rx];
        let mut exchange_engine = ExchangeEngine::new(
            sock_addr,
            proxy_tx,
            proxy_rx,
        );
        std::thread::spawn(move || {
            exchange_engine.mainloop();
        });

        let root_sock_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 211, 66)), 5000);
        let rank = host;
        let init_comm = InitCommunicator {
            communicator_id: CommunicatorId(1042),
            root_mccs_addr: root_sock_addr,
            rank,
            num_ranks: 2,
        };
        let cmd = ProxyCommand::InitCommunicator(init_comm);
        daemon_cmd_tx.send(cmd).unwrap();
        let comp = daemon_comp_rx.recv().unwrap();
        match comp {
            ProxyCompletion::InitCommunicator => (),
            _ => panic!("unexpected"),
        }
        log::info!("Init communicator done");
    }

    // fn initialize_test_region(
    //     buf_size: usize,
    //     first_content: i32,
    //     second_content: i32,
    // ) -> (*mut libc::c_void, *mut libc::c_void) {
    //     unsafe {
    //         let error = cudaSetDevice(0);
    //         if error != cudaError::cudaSuccess {
    //             panic!("cudaSetDevice");
    //         }
    //     }
    //     // Inference
    //     let dev_buf_0 = unsafe {
    //         let mut dev_ptr = std::ptr::null_mut();
    //         cuda_warning!(cudaMalloc(&mut dev_ptr, buf_size));
    //         dev_ptr
    //     };
    //     let mut buf = vec![first_content; buf_size / 2 / std::mem::size_of::<i32>()];
    //     buf.extend(vec![0i32; buf_size / 2 / std::mem::size_of::<i32>()]);

    //     unsafe {
    //         cudaMemcpy(
    //             dev_buf_0,
    //             buf.as_ptr() as *const _,
    //             buf_size,
    //             cudaMemcpyKind::cudaMemcpyHostToDevice,
    //         )
    //     };

    //     unsafe {
    //         let error = cudaSetDevice(1);
    //         if error != cudaError::cudaSuccess {
    //             panic!("cudaSetDevice");
    //         }
    //     }
    //     let dev_buf_1 = unsafe {
    //         let mut dev_ptr = std::ptr::null_mut();
    //         cuda_warning!(cudaMalloc(&mut dev_ptr, buf_size));
    //         dev_ptr
    //     };
    //     let mut buf = vec![0i32; buf_size / 2 / std::mem::size_of::<i32>()];
    //     buf.extend(vec![
    //         second_content;
    //         buf_size / 2 / std::mem::size_of::<i32>()
    //     ]);
    //     unsafe {
    //         cudaMemcpy(
    //             dev_buf_1,
    //             buf.as_ptr() as *const _,
    //             buf_size,
    //             cudaMemcpyKind::cudaMemcpyHostToDevice,
    //         )
    //     };
    //     (dev_buf_0, dev_buf_1)
    // }

    /*
    #[allow(dead_code)]
    fn test2(&mut self) -> anyhow::Result<()> {
        let initial_timer = Instant::now();
        let inference_comm_id = 0;
        let training_comm_id = 1;
        let mut num_devices = 0;
        unsafe {
            let error = cudaGetDeviceCount(&mut num_devices as *mut _);
            if error != cudaError::cudaSuccess {
                panic!("cudaGetDeviceCount");
            }
        }
        let transport_delegator = TransportDelegator::new();
        let transport_catalog = TransportCatalog::new();
        let shm_config = ShmConfig {
            locality: crate::transport::shm::config::ShmLocality::Sender,
            use_memcpy_send: false,
            use_memcpy_recv: false,
        };
        transport_catalog.register_config(String::from("ShmTransport"), shm_config);
        let registry = GlobalRegistry {
            communicators: DashMap::new(),
            transport_delegator,
            transport_catalog,
        };
        let registry = Arc::new(registry);
        let (proxy_0_tx, proxy_0_rx) = crossbeam::channel::unbounded();
        let (proxy_1_tx, proxy_1_rx) = crossbeam::channel::unbounded();
        let (daemon_0_cmd_tx, daemon_0_cmd_rx) = crossbeam::channel::unbounded();
        let (daemon_0_comp_tx, daemon_0_comp_rx) = crossbeam::channel::unbounded();
        let (daemon_1_cmd_tx, daemon_1_cmd_rx) = crossbeam::channel::unbounded();
        let (daemon_1_comp_tx, daemon_1_comp_rx) = crossbeam::channel::unbounded();

        let (daemon2_0_cmd_tx, daemon2_0_cmd_rx) = crossbeam::channel::unbounded();
        let (daemon2_0_comp_tx, daemon2_0_comp_rx) = crossbeam::channel::unbounded();
        let (daemon2_1_cmd_tx, daemon2_1_cmd_rx) = crossbeam::channel::unbounded();
        let (daemon2_1_comp_tx, daemon2_1_comp_rx) = crossbeam::channel::unbounded();

        let sock_addr = std::net::SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8000);

        {
            let mut daemon_tx = HashMap::new();
            daemon_tx.insert(0, daemon_0_comp_tx);
            daemon_tx.insert(1, daemon2_0_comp_tx);
            let daemon_rx = vec![(0, daemon_0_cmd_rx), (1, daemon2_0_cmd_rx)];
            let proxy_peer_tx = vec![proxy_0_tx.clone(), proxy_1_tx.clone()];
            Self::start_test_proxy(
                0,
                sock_addr,
                daemon_tx,
                daemon_rx,
                proxy_peer_tx,
                proxy_0_rx,
                registry.clone(),
            )?;
        }
        {
            let mut daemon_tx = HashMap::new();
            daemon_tx.insert(0, daemon_1_comp_tx);
            daemon_tx.insert(1, daemon2_1_comp_tx);
            let daemon_rx = vec![(0, daemon_1_cmd_rx), (1, daemon2_1_cmd_rx)];
            let proxy_peer_tx = vec![proxy_0_tx, proxy_1_tx];
            Self::start_test_proxy(
                1,
                sock_addr,
                daemon_tx,
                daemon_rx,
                proxy_peer_tx,
                proxy_1_rx,
                registry.clone(),
            )?;
        }

        // Proxy Engine initialization finished

        {
            daemon_0_cmd_tx
                .send(ProxyCommand::InitCommunicator(InitCommunicator {
                    communicator_id: CommunicatorId(inference_comm_id),
                    rank: 0,
                    num_ranks: 2,
                }))
                .unwrap();
            daemon_1_cmd_tx
                .send(ProxyCommand::InitCommunicator(InitCommunicator {
                    communicator_id: CommunicatorId(inference_comm_id),
                    rank: 1,
                    num_ranks: 2,
                }))
                .unwrap();
            match daemon_0_comp_rx.recv() {
                Ok(ProxyCompletion::InitCommunicator) => (),
                _ => panic!(),
            };
            match daemon_1_comp_rx.recv() {
                Ok(ProxyCompletion::InitCommunicator) => (),
                _ => panic!(),
            };

            daemon2_0_cmd_tx
                .send(ProxyCommand::InitCommunicator(InitCommunicator {
                    communicator_id: CommunicatorId(training_comm_id),
                    rank: 0,
                    num_ranks: 2,
                }))
                .unwrap();
            daemon2_1_cmd_tx
                .send(ProxyCommand::InitCommunicator(InitCommunicator {
                    communicator_id: CommunicatorId(training_comm_id),
                    rank: 1,
                    num_ranks: 2,
                }))
                .unwrap();
            match daemon2_0_comp_rx.recv() {
                Ok(ProxyCompletion::InitCommunicator) => (),
                _ => panic!(),
            };
            match daemon2_1_comp_rx.recv() {
                Ok(ProxyCompletion::InitCommunicator) => (),
                _ => panic!(),
            };

            // -----------------------------------------------------------

            const BUFFER_SIZE: usize = 1024 * 1024 * 1024 * 2;
            const BUFFER_SIZE_2: usize = 1024 * 1024 * 1024 * 4;

            // Inference
            let (dev_buf_0, dev_buf_1) = Self::initialize_test_region(BUFFER_SIZE, 1883, 2042);
            log::info!("dev_buf_0: {:p} of size {BUFFER_SIZE} bytes", dev_buf_0);
            log::info!("dev_buf_1: {:p} of size {BUFFER_SIZE} bytes", dev_buf_1);
            // training
            let (dev_buf2_0, dev_buf2_1) = Self::initialize_test_region(BUFFER_SIZE_2, 2049, 40999);
            log::info!("dev_buf2_0: {:p} of size {BUFFER_SIZE_2} bytes", dev_buf2_0);
            log::info!("dev_buf2_1: {:p} of size {BUFFER_SIZE_2} bytes", dev_buf2_1);

            log::info!("Initialization: {} ms", initial_timer.elapsed().as_millis());

            //--------------------------------------------------------
            let before_allgather = Instant::now();
            // inference
            daemon_0_cmd_tx
                .send(ProxyCommand::AllGather(AllGather {
                    communicator_id: CommunicatorId(inference_comm_id),
                    send_buf_addr: dev_buf_0 as usize,
                    recv_buf_addr: dev_buf_0 as usize,
                    size: BUFFER_SIZE / 2,
                }))
                .unwrap();

            daemon_1_cmd_tx
                .send(ProxyCommand::AllGather(AllGather {
                    communicator_id: CommunicatorId(inference_comm_id),
                    send_buf_addr: dev_buf_1 as usize + BUFFER_SIZE / 2,
                    recv_buf_addr: dev_buf_1 as usize,
                    size: BUFFER_SIZE / 2,
                }))
                .unwrap();

            // training
            daemon2_0_cmd_tx
                .send(ProxyCommand::AllGather(AllGather {
                    communicator_id: CommunicatorId(training_comm_id),
                    send_buf_addr: dev_buf2_0 as usize,
                    recv_buf_addr: dev_buf2_0 as usize,
                    size: BUFFER_SIZE_2 / 2,
                }))
                .unwrap();

            daemon2_1_cmd_tx
                .send(ProxyCommand::AllGather(AllGather {
                    communicator_id: CommunicatorId(training_comm_id),
                    send_buf_addr: dev_buf2_1 as usize + BUFFER_SIZE_2 / 2,
                    recv_buf_addr: dev_buf2_1 as usize,
                    size: BUFFER_SIZE_2 / 2,
                }))
                .unwrap();

            match daemon_0_comp_rx.recv() {
                Ok(ProxyCompletion::AllGather(_)) => (),
                _ => panic!(),
            }
            match daemon_1_comp_rx.recv() {
                Ok(ProxyCompletion::AllGather(_)) => (),
                _ => panic!(),
            }
            log::info!(
                "Inference All Gather: {} ms",
                before_allgather.elapsed().as_millis()
            );

            match daemon2_0_comp_rx.recv() {
                Ok(ProxyCompletion::AllGather(_)) => (),
                _ => panic!(),
            }
            match daemon2_1_comp_rx.recv() {
                Ok(ProxyCompletion::AllGather(_)) => (),
                _ => panic!(),
            }

            log::info!("All Gather: {} ms", before_allgather.elapsed().as_millis());

            //---------------------------------------------------

            // check inference
            let mut buf = vec![0; BUFFER_SIZE];
            unsafe {
                let err = cudaMemcpy(
                    buf.as_mut_ptr() as *mut _,
                    dev_buf_1,
                    BUFFER_SIZE,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                );
                if err != cudaError::cudaSuccess {
                    panic!("cudaMemcpy failed");
                }
            };
            assert_eq!(buf[0], 1883);
            assert_eq!(buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()], 2042);

            // check training
            let mut buf = vec![0; BUFFER_SIZE_2];
            unsafe {
                let err = cudaMemcpy(
                    buf.as_mut_ptr() as *mut _,
                    dev_buf2_1,
                    BUFFER_SIZE_2,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                );
                if err != cudaError::cudaSuccess {
                    panic!("cudaMemcpy failed");
                }
            };
            assert_eq!(buf[0], 2049);
            assert_eq!(buf[BUFFER_SIZE_2 / 2 / std::mem::size_of::<i32>()], 40999);
            log::info!("Pass data check");
        }
        Ok(())
    }
    */
}
