use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::anyhow;

use crossbeam::channel::{Receiver, Sender};
use cuda_runtime_sys::cudaGetDeviceCount;
use ipc::customer::ShmCustomer;
use ipc::unix::DomainSocket;
use qos_service::QosSchedule;

use crate::comm::CommunicatorId;
use crate::config::CommPatternConfig;
use crate::config::Config;
use crate::cuda_warning;
use crate::daemon::engine::DaemonEngine;
use crate::daemon::DaemonId;
use crate::exchange::command::{ExchangeCommand, ExchangeNotification};
use crate::exchange::engine::ExchangeEngine;
use crate::message::{ControlNotification, ControlRequest};
use crate::proxy::engine::ProxyEngine;
use crate::proxy::DeviceInfo;
use crate::registry::GlobalRegistry;
use crate::runtime::affinity::init_nvml;
use crate::runtime::CoreMask;
use crate::runtime::RuntimeManager;
use crate::transport::catalog::TransportCatalog;
use crate::transport::delegator::TransportDelegator;
use crate::transport::engine::TransportEngine;
use crate::transport::engine::TransportEngineId;
use crate::transport::net::RDMA_TRANSPORT;
use crate::utils::duplex_chan::DuplexChannel;

// Imports for tests

// use cuda_runtime_sys::cudaError;
// use cuda_runtime_sys::cudaEventCreateWithFlags;
// use cuda_runtime_sys::cudaEventDisableTiming;
// use cuda_runtime_sys::cudaEventInterprocess;
// use cuda_runtime_sys::cudaEventRecord;
// use cuda_runtime_sys::cudaIpcEventHandle_t;
// use cuda_runtime_sys::cudaIpcGetEventHandle;
// use cuda_runtime_sys::cudaMalloc;
// use cuda_runtime_sys::cudaMemcpy;
// use cuda_runtime_sys::cudaMemcpyKind;
// use cuda_runtime_sys::cudaSetDevice;
// use cuda_runtime_sys::cudaStreamSynchronize;
// use qos_service::QosSchedule;

// use crate::engine::Engine;
// use crate::proxy::command::{AllGatherRequest, InitCommunicator};
// use crate::proxy::command::{ProxyCommand, ProxyCompletion};
// use crate::transport::engine::TransportEngineResources;
// use crate::transport::message::{TransportEngineReply, TransportEngineRequest};
// use crate::transport::net::config::NetTransportConfig;
// use crate::transport::queue::TransrportOpQueue;
// use crate::utils::pool::WorkPool;

pub struct Control {
    sock: DomainSocket,
    config: Config,
    comm_patterns: HashMap<CommunicatorId, CommPatternConfig>,
    daemon_count: usize,
    transport_catalog: Arc<TransportCatalog>,
    transport_delegator: Arc<TransportDelegator>,
    runtime_manager: RuntimeManager,
    proxy_tx: Vec<Sender<ControlNotification>>,
    proxy_rx: Vec<Receiver<ControlRequest>>,
}

fn process_env(path: &Path) -> PathBuf {
    Path::new(&path.to_string_lossy().to_string().replace(
        "${USER}",
        std::env::var("USER").unwrap_or_default().as_str(),
    ))
    .to_path_buf()
}

impl Control {
    pub fn new(config: Config, host: usize) -> Self {
        use crate::transport::net::provider::NetProvierWrap;

        let mccs_prefix = process_env(&config.control.prefix);
        fs::create_dir_all(&mccs_prefix)
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

        let transport_delegator = Arc::new(TransportDelegator::new());
        let transport_catalog = Arc::new(TransportCatalog::new());

        let rdma_config = config.comm_global_config.rdma_config.clone();
        transport_catalog.register_config(String::from("NetProviderRdma"), rdma_config);
        let net_config = config.comm_global_config.net_config.clone();
        transport_catalog.register_config(String::from("NetTransport"), net_config);
        let shm_config = config.comm_global_config.shm_config.clone();
        transport_catalog.register_config(String::from("ShmTransport"), shm_config);
        RDMA_TRANSPORT.init(&transport_catalog).unwrap();

        init_nvml();

        let mut comm_patterns = HashMap::new();
        if let Some(comm_patterns_override) = &config.comm_patterns_override {
            for pattern_config in comm_patterns_override.iter() {
                let comm_id = CommunicatorId(pattern_config.communicator_id);
                let pattern_config = pattern_config.clone();
                comm_patterns.insert(comm_id, pattern_config);
            }
        }

        let runtime_manager = RuntimeManager::new();
        let listen_addr = std::net::SocketAddr::new(config.addrs[host].clone(), config.listen_port);
        let exchange_chans = Self::create_exchange_engine(&listen_addr, &runtime_manager);
        let proxy_chans = Self::create_proxy_engines(
            &config.addrs[host],
            exchange_chans,
            &runtime_manager,
            &config,
            &comm_patterns,
            &transport_delegator,
            &transport_catalog,
        );
        let mut proxy_tx = Vec::with_capacity(proxy_chans.len());
        let mut proxy_rx = Vec::with_capacity(proxy_chans.len());
        for chan in proxy_chans.into_iter() {
            proxy_tx.push(chan.tx);
            proxy_rx.push(chan.rx);
        }

        Control {
            sock,
            config,
            comm_patterns,
            daemon_count: 0,
            transport_catalog,
            transport_delegator,
            runtime_manager,
            proxy_tx,
            proxy_rx,
        }
    }

    pub fn mainloop(&mut self, exit_flag: &AtomicBool) -> anyhow::Result<()> {
        let mut buf = vec![0u8; 65536];
        while !exit_flag.load(Ordering::Relaxed) {
            match self.sock.recv_with_credential_from(buf.as_mut_slice()) {
                Ok((size, sender, cred)) => {
                    if let Some(cred) = cred {
                        if let Err(_e) = self.dispatch(&mut buf[..size], &sender, &cred) {
                            // log
                        }
                    } else {
                        // log
                    }
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(_e) => {
                    if exit_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    // log
                }
            }
            self.check_proxy_requests();
        }
        Ok(())
    }

    fn check_proxy_requests(&mut self) {
        for proxy_rx in self.proxy_rx.iter() {
            match proxy_rx.try_recv() {
                Ok(req) => match req {
                    ControlRequest::NewTransportEngine(engine_id) => {
                        let mut proxy_endpoints = Vec::new();
                        let mut transport_endpoints = Vec::new();
                        for _ in 0..self.proxy_tx.len() {
                            let (proxy_endpoint, transport_endpoint) =
                                DuplexChannel::new_unbound_pair();
                            proxy_endpoints.push(proxy_endpoint);
                            transport_endpoints.push(transport_endpoint);
                        }
                        let global_registry = GlobalRegistry {
                            default_comm_config: self.config.comm_default_config.clone(),
                            comm_pattern_override: self.comm_patterns.clone(),
                            transport_delegator: Arc::clone(&self.transport_delegator),
                            transport_catalog: Arc::clone(&self.transport_catalog),
                        };
                        let qos_schedule = match self.config.qos_schedule {
                            Some(ref schedule) => schedule.clone().into(),
                            None => QosSchedule {
                                schedule: HashMap::new(),
                                epoch_microsecs: 0,
                            },
                        };
                        let engine = TransportEngine::new(
                            engine_id,
                            transport_endpoints,
                            global_registry,
                            qos_schedule,
                        );
                        let cores = CoreMask::from_device_affinity(engine_id.cuda_device_idx);
                        let container = Box::new(engine);
                        self.runtime_manager.submit_engine(
                            container,
                            Some(engine_id.cuda_device_idx),
                            Some(cores),
                        );
                        for (proxy_tx, proxy_endpoint) in
                            self.proxy_tx.iter_mut().zip(proxy_endpoints.into_iter())
                        {
                            proxy_tx
                                .send(ControlNotification::NewTransportEngine {
                                    id: engine_id,
                                    chan: proxy_endpoint,
                                })
                                .unwrap();
                        }
                    }
                },
                Err(crossbeam::channel::TryRecvError::Empty) => (),
                Err(crossbeam::channel::TryRecvError::Disconnected) => {
                    panic!("Proxy engines shall never shutdown")
                }
            }
        }
    }

    fn create_proxy_engines(
        addr: &std::net::IpAddr,
        exchange_chans: Vec<DuplexChannel<ExchangeCommand, ExchangeNotification>>,
        runtime_manager: &RuntimeManager,
        config: &Config,
        comm_patterns: &HashMap<CommunicatorId, CommPatternConfig>,
        transport_delegator: &Arc<TransportDelegator>,
        transport_catalog: &Arc<TransportCatalog>,
    ) -> Vec<DuplexChannel<ControlNotification, ControlRequest>> {
        // we don't allow device hot-plug, so we create connected proxies in advance
        let device_cnt = {
            let mut i = 0;
            cuda_warning!(unsafe { cudaGetDeviceCount(&mut i) });
            i as i32 as usize
        };
        assert_eq!(device_cnt, exchange_chans.len());
        let mut control_endpoints = Vec::new();
        for (dev_idx, exchange_chan) in exchange_chans.into_iter().enumerate() {
            let (control_endpoint, proxy_endpoint) = DuplexChannel::new_unbound_pair();
            control_endpoints.push(control_endpoint);
            let device_info = DeviceInfo {
                host: addr.clone(),
                listen_port: config.listen_port,
                cuda_device_idx: dev_idx as i32,
            };
            let global_registry = GlobalRegistry {
                default_comm_config: config.comm_default_config.clone(),
                comm_pattern_override: comm_patterns.clone(),
                transport_delegator: Arc::clone(transport_delegator),
                transport_catalog: Arc::clone(transport_catalog),
            };
            let cores = CoreMask::from_device_affinity(dev_idx as i32);
            let engine =
                ProxyEngine::new(device_info, global_registry, proxy_endpoint, exchange_chan);
            let container = Box::new(engine);
            runtime_manager.submit_engine(container, Some(dev_idx as i32), Some(cores));
        }
        control_endpoints
    }

    fn create_exchange_engine(
        addr: &std::net::SocketAddr,
        runtime_manager: &RuntimeManager,
    ) -> Vec<DuplexChannel<ExchangeCommand, ExchangeNotification>> {
        let mut exchange_txs = Vec::new();
        let mut exchange_rxs = Vec::new();
        let mut proxy_endpoints = Vec::new();
        let device_cnt = {
            let mut i = 0;
            cuda_warning!(unsafe { cudaGetDeviceCount(&mut i) });
            i as i32 as usize
        };
        for _ in 0..device_cnt {
            let (exchange_endpoint, proxy_endpoint) = DuplexChannel::new_unbound_pair();
            proxy_endpoints.push(proxy_endpoint);
            exchange_txs.push(exchange_endpoint.tx);
            exchange_rxs.push(exchange_endpoint.rx);
        }
        let engine = ExchangeEngine::new(addr.clone(), exchange_txs, exchange_rxs);
        let container = Box::new(engine);
        runtime_manager.submit_engine(container, None, None);
        proxy_endpoints
    }

    fn dispatch(
        &mut self,
        buf: &mut [u8],
        sender: &std::os::unix::net::SocketAddr,
        _cred: &std::os::unix::net::UCred,
    ) -> anyhow::Result<()> {
        use ipc::control;
        let msg: control::Request = bincode::deserialize(buf).unwrap();
        match msg {
            control::Request::NewClient(device_affnity) => {
                let client_path = sender
                    .as_pathname()
                    .ok_or_else(|| anyhow!("peer is unnamed, something is wrong"))?;

                let uuid = uuid::Uuid::new_v4();
                let instance_name = format!("{}-{}.sock", self.config.mccs_daemon_basename, uuid);
                let engine_path = process_env(&self.config.mccs_daemon_prefix).join(instance_name);

                // create customer stub
                let customer = ShmCustomer::accept(&self.sock, client_path, engine_path)?;

                let daemon_id = DaemonId(self.daemon_count as u32);
                let num_devices = self.proxy_tx.len();
                let mut daemon_channels = Vec::with_capacity(num_devices);

                for device_idx in 0..num_devices {
                    let endpoint_tx = &mut self.proxy_tx[device_idx];
                    let (daemon_side, proxy_side) = DuplexChannel::new_unbound_pair();
                    let proxy_endpoint = ControlNotification::NewDaemon {
                        id: daemon_id,
                        chan: proxy_side,
                    };
                    endpoint_tx.send(proxy_endpoint).unwrap();
                    daemon_channels.push(daemon_side);
                }

                let engine = DaemonEngine::new(daemon_id, customer, daemon_channels);
                let container = Box::new(engine);
                // TODO: check cuda dev
                let cores =
                    device_affnity.map(|dev_idx| CoreMask::from_device_affinity(dev_idx as i32));
                self.runtime_manager.submit_engine(container, None, cores);
                self.daemon_count += 1;

                Ok(())
            }
        }
    }
}

// impl Control {
//     fn start_test_proxy(
//         cuda_device: i32,
//         ip_addr: std::net::IpAddr,
//         listen_port: u16,
//         daemon_tx: HashMap<DaemonId, crossbeam::channel::Sender<ProxyCompletion>>,
//         daemon_rx: Vec<(DaemonId, crossbeam::channel::Receiver<ProxyCommand>)>,
//         exchange_chan: DuplexChannel<ExchangeCommand, ExchangeNotification>,
//         transport_engines_tx: HashMap<
//             TransportEngineId,
//             crossbeam::channel::Sender<TransportEngineRequest>,
//         >,
//         transport_engines_rx: Vec<(
//             TransportEngineId,
//             crossbeam::channel::Receiver<TransportEngineReply>,
//         )>,
//         global_registry: GlobalRegistry,
//     ) -> anyhow::Result<()> {
//         let dev_info = DeviceInfo {
//             host: ip_addr,
//             listen_port,
//             cuda_device_idx: cuda_device,
//         };
//         let (control_req_tx, _control_req_rx) = crossbeam::channel::unbounded();
//         let (_control_notify_tx, control_notify_rx) = crossbeam::channel::unbounded();

//         let proxy_resources = crate::proxy::engine::ProxyResources {
//             device_info: dev_info,
//             control_chan: DuplexChannel {
//                 tx: control_req_tx,
//                 rx: control_notify_rx,
//             },
//             daemon_tx,
//             daemon_rx,
//             exchange_chan,
//             comms_init: HashMap::new(),
//             comms_suspended: HashMap::new(),
//             user_events: HashMap::new(),
//             communicators: HashMap::new(),
//             global_registry,
//             transport_engines_tx,
//             transport_engines_rx,
//             transport_submission_cache: HashMap::new(),
//             task_submit_pool: Vec::new(),
//             daemon_shutdown: Vec::new(),
//             transport_shutdown: Vec::new(),
//         };
//         let mut proxy = ProxyEngine {
//             resources: proxy_resources,
//             ops: WorkPool::new(),
//             async_tasks: WorkPool::new(),
//         };
//         std::thread::spawn(move || {
//             unsafe {
//                 let error = cudaSetDevice(cuda_device);
//                 if error != cudaError::cudaSuccess {
//                     panic!("cudaSetDevice");
//                 }
//             }
//             loop {
//                 proxy.progress();
//             }
//         });
//         Ok(())
//     }

//     pub fn dist_test(host: usize) {
//         let comm_id = 1042;
//         use crate::transport::net::provider::NetProvierWrap;
//         let transport_delegator = Arc::new(TransportDelegator::new());
//         transport_delegator
//             .active_connections
//             .insert(0, vec![(0, 0)]);
//         let transport_engine_id = TransportEngineId {
//             cuda_device_idx: 0,
//             index: 0,
//         };
//         let (transport_cmd_tx, transport_cmd_rx) = crossbeam::channel::unbounded();
//         let (transport_comp_tx, transport_comp_rx) = crossbeam::channel::unbounded();
//         let mut transport_engines_tx = HashMap::new();
//         transport_engines_tx.insert(transport_engine_id, transport_cmd_tx);
//         let transport_engines_rx = vec![(transport_engine_id, transport_comp_rx)];

//         let transport_catalog = TransportCatalog::new();
//         let net_config = NetTransportConfig {
//             gdr_enable: false,
//             gdr_copy_sync_enable: false,
//             gdr_copy_flush_enable: false,
//         };
//         let rdma_config = crate::transport::net::provider::RdmaTransportConfig::default();
//         transport_catalog.register_config(String::from("NetTransport"), net_config);
//         transport_catalog.register_config(String::from("NetProviderRdma"), rdma_config);
//         crate::transport::net::provider::RDMA_TRANSPORT
//             .init(&transport_catalog)
//             .unwrap();
//         let registry = GlobalRegistry {
//             default_comm_config: crate::config::DefaultCommConfig::default(),
//             transport_delegator,
//             transport_catalog: Arc::new(transport_catalog),
//         };

//         let qos_schedule = QosSchedule {
//             schedule: HashMap::new(),
//             epoch_microsecs: 0,
//         };
//         let resources = TransportEngineResources {
//             agent_setup: HashMap::new(),
//             agent_connected: HashMap::new(),
//             proxy_chan: vec![DuplexChannel {
//                 tx: transport_comp_tx,
//                 rx: transport_cmd_rx,
//             }],
//             global_registry: registry.clone(),
//             qos_schedule,
//         };
//         let mut transport_engine = TransportEngine {
//             id: transport_engine_id,
//             resources,
//             async_tasks: WorkPool::new(),
//             op_queue: TransrportOpQueue::new(),
//         };
//         std::thread::spawn(move || {
//             unsafe {
//                 let error = cudaSetDevice(0);
//                 if error != cudaError::cudaSuccess {
//                     panic!("cudaSetDevice");
//                 }
//             }
//             loop {
//                 transport_engine.progress();
//             }
//         });

//         let (daemon_cmd_tx, daemon_cmd_rx) = crossbeam::channel::unbounded();
//         let (daemon_comp_tx, daemon_comp_rx) = crossbeam::channel::unbounded();
//         let (exchange_chans_proxy_endpoint, exchange_chans_exchange_endpoint) =
//             DuplexChannel::new_unbound_pair();
//         let sock_addr = if host == 0 {
//             std::net::SocketAddr::new(
//                 std::net::IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 211, 66)),
//                 5000,
//             )
//         } else {
//             std::net::SocketAddr::new(
//                 std::net::IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 211, 194)),
//                 5000,
//             )
//         };
//         let mut daemon_tx = HashMap::new();
//         daemon_tx.insert(DaemonId(0), daemon_comp_tx);
//         let daemon_rx = vec![(DaemonId(0), daemon_cmd_rx)];
//         Self::start_test_proxy(
//             0,
//             sock_addr.ip(),
//             sock_addr.port(),
//             daemon_tx,
//             daemon_rx,
//             exchange_chans_proxy_endpoint,
//             transport_engines_tx,
//             transport_engines_rx,
//             registry.clone(),
//         )
//         .unwrap();

//         let proxy_tx = vec![exchange_chans_exchange_endpoint.tx];
//         let proxy_rx = vec![exchange_chans_exchange_endpoint.rx];
//         let mut exchange_engine = ExchangeEngine::new(sock_addr, proxy_tx, proxy_rx);
//         std::thread::spawn(move || loop {
//             exchange_engine.progress();
//         });

//         let root_sock_addr = std::net::IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 211, 66));
//         let rank = host;
//         let init_comm = InitCommunicator {
//             communicator_id: CommunicatorId(comm_id),
//             root_mccs_addr: root_sock_addr,
//             rank,
//             num_ranks: 2,
//         };
//         let cmd = ProxyCommand::InitCommunicator(init_comm);
//         daemon_cmd_tx.send(cmd).unwrap();
//         let comp = daemon_comp_rx.recv().unwrap();
//         match comp {
//             ProxyCompletion::InitCommunicator(_) => (),
//             _ => panic!("unexpected"),
//         }
//         log::info!("Init communicator done");

//         /// ----------------------------------------------------------
//         const BUFFER_SIZE: usize = 8 * 1024 * 1024;

//         let first = if host == 0 { 42 } else { 99 };
//         let second = if host == 0 { 88 } else { 37 };
//         log::info!("Buffer content: [{};{}]", first, second);

//         let dev_buf_0 = Self::initialize_test_region(BUFFER_SIZE, first, second);

//         let handle = unsafe {
//             let mut event = std::ptr::null_mut();
//             cuda_warning!(cudaEventCreateWithFlags(
//                 &mut event,
//                 cudaEventInterprocess | cudaEventDisableTiming
//             ));
//             cuda_warning!(cudaEventRecord(event, std::ptr::null_mut()));
//             let mut handle = cudaIpcEventHandle_t::default();
//             cuda_warning!(cudaIpcGetEventHandle(&mut handle, event));
//             handle
//         };
//         log::info!("Start proxying");

//         daemon_cmd_tx
//             .send(ProxyCommand::AllGather(AllGatherRequest {
//                 communicator_id: CommunicatorId(comm_id),
//                 send_buf_addr: dev_buf_0 as usize + if host == 0 { 0 } else { BUFFER_SIZE / 2 },
//                 recv_buf_addr: dev_buf_0 as usize,
//                 size: BUFFER_SIZE / 2,
//                 user_stream: 0,
//             }))
//             .unwrap();
//         log::info!("Sent request");
//         match daemon_comp_rx.recv().unwrap() {
//             ProxyCompletion::AllGather => (),
//             _ => panic!("Unexpected"),
//         };
//         log::info!("Got handle");

//         // wait
//         unsafe {
//             // let mut event = std::ptr::null_mut();
//             // cuda_warning!(cudaIpcOpenEventHandle(&mut event, handle.into()));
//             // cuda_warning!(cudaStreamWaitEvent(std::ptr::null_mut(), event, 0));
//             std::thread::sleep(Duration::from_secs(4));
//             log::info!("wake up");
//             cuda_warning!(cudaStreamSynchronize(std::ptr::null_mut()));
//         }
//         log::info!("synchronized");

//         // check
//         let mut buf = vec![0u8; BUFFER_SIZE];
//         unsafe {
//             let err = cudaMemcpy(
//                 buf.as_mut_ptr() as *mut _,
//                 dev_buf_0,
//                 BUFFER_SIZE,
//                 cudaMemcpyKind::cudaMemcpyDeviceToHost,
//             );
//             if err != cudaError::cudaSuccess {
//                 panic!("cudaMemcpy failed");
//             }
//         };
//         log::info!("memcpy done");
//         println!(
//             "[0]={} [100]={} [4MB-1]={}",
//             buf[0],
//             buf[100],
//             buf[BUFFER_SIZE / 2 - 1]
//         );
//         println!(
//             "[4MB]={} [4MB+100]={} [Last]={}",
//             buf[BUFFER_SIZE / 2],
//             buf[BUFFER_SIZE / 2 + 100],
//             buf[BUFFER_SIZE - 1]
//         );
//         assert_eq!(buf[0], 42);
//         assert_eq!(buf[BUFFER_SIZE / 2], 37);
//         log::info!("Success");
//     }

//     fn initialize_test_region(
//         buf_size: usize,
//         first_content: u8,
//         second_content: u8,
//     ) -> *mut nix::libc::c_void {
//         unsafe {
//             let error = cudaSetDevice(0);
//             if error != cudaError::cudaSuccess {
//                 panic!("cudaSetDevice");
//             }
//         }
//         // Inference
//         let dev_buf_0 = unsafe {
//             let mut dev_ptr = std::ptr::null_mut();
//             cuda_warning!(cudaMalloc(&mut dev_ptr, buf_size));
//             dev_ptr
//         };
//         let mut buf = vec![first_content; buf_size / 2];
//         buf.extend(vec![second_content; buf_size / 2]);

//         unsafe {
//             cudaMemcpy(
//                 dev_buf_0,
//                 buf.as_ptr() as *const _,
//                 buf_size,
//                 cudaMemcpyKind::cudaMemcpyHostToDevice,
//             )
//         };

//         dev_buf_0
//     }
// }

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
