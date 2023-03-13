use std::collections::HashMap;
use std::fs;
use std::hash::Hash;
use std::io;
use std::net::IpAddr;
use std::net::Ipv4Addr;
use std::net::SocketAddrV4;
use std::os::unix::net::{SocketAddr, UCred};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use std::time::Duration;

use anyhow::anyhow;
use crossbeam::channel::Sender;
use cuda_runtime_sys::cudaMalloc;
use cuda_runtime_sys::cudaMemcpy;
use cuda_runtime_sys::cudaMemcpyKind;
use dashmap::DashMap;

use cuda_runtime_sys::cudaError;
use cuda_runtime_sys::cudaGetDeviceCount;
use cuda_runtime_sys::cudaSetDevice;
use ipc::customer::ShmCustomer;
use ipc::unix::DomainSocket;

use crate::comm::CommunicatorId;
use crate::comm::HostIdent;
use crate::config::Config;
use crate::daemon::DaemonId;
use crate::proxy::DeviceInfo;
use crate::proxy::command::AllGather;
use crate::proxy::command::InitCommunicator;
use crate::proxy::command::ProxyCommand;
use crate::proxy::command::ProxyCompletion;
use crate::proxy::engine::ProxyEngine;
use crate::proxy::engine::ProxyResources;
use crate::registry::GlobalRegistry;
use crate::transport::catalog::TransportCatalog;
use crate::transport::delegator::TransportDelegator;
use crate::transport::shm::config::ShmConfig;
use crate::utils::pool::WorkPool;

pub struct Control {
    sock: DomainSocket,
    config: Config,
    daemon_cnt: DaemonId,
}

impl Control {
    pub fn new(config: Config) -> Self {
        let mccs_prefix = &config.control.prefix;
        fs::create_dir_all(mccs_prefix).unwrap_or_else(|e| {
            panic!("Failed to create directory for {:?}: {}", mccs_prefix, e)
        });

        let mccs_path = mccs_prefix.join(&config.control.path);
        if mccs_path.exists() {
            fs::remove_file(&mccs_path).expect("remove_file");
        }
        let sock = DomainSocket::bind(&mccs_path)
            .unwrap_or_else(|e| panic!("Cannot bind domain socket at {:?}: {}", mccs_path, e));

        sock.set_read_timeout(Some(Duration::from_millis(1)))
            .expect("set_read_timeout");
        sock.set_write_timeout(Some(Duration::from_millis(1)))
            .expect("set_write_timeout");

        // let transport_setup = TransportSetupRegistry::new();

        // let global_resources = GlobalResources {
        //     communicators: DashMap::new(),
        //     transport_setup: transport_setup,
        // };

        let mut control = Control {
            sock,
            config,
            daemon_cnt: 0,
        };
        control.test().unwrap();
        control
        // control.create_proxies().unwrap();
        // control

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
        }
        Ok(())
    }

    fn test(&mut self) -> anyhow::Result<()> {
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

        let sock_addr = std::net::SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            8000
        );
        let dev_info = DeviceInfo {
            host: HostIdent(sock_addr),
            cuda_device_idx: 0,
        };
        let (control_req_tx, _control_req_rx) = crossbeam::channel::unbounded();
        let (_control_notify_tx, control_notify_rx) = crossbeam::channel::unbounded();
        let mut daemon_tx = HashMap::new();
        daemon_tx.insert(0, daemon_0_comp_tx);
        let daemon_rx = vec![(0, daemon_0_cmd_rx)];
        let proxy_peer_tx = vec![proxy_0_tx.clone(), proxy_1_tx.clone()];
        let proxy_0_resources = ProxyResources {
            device_info: dev_info,
            control_tx: control_req_tx,
            control_rx: control_notify_rx,
            daemon_tx,
            daemon_rx,
            proxy_peer_tx,
            proxy_peer_rx: proxy_0_rx,
            comms_init: HashMap::new(),
            communicators: HashMap::new(),
            global_registry: Arc::clone(&registry),
            transport_engines_tx: HashMap::new(),
            transport_engines_rx: Vec::new(),
            transport_submission_pool: HashMap::new(),
        };
        let mut proxy_0 = ProxyEngine {
            resources: proxy_0_resources,
            ops: WorkPool::new(),
        };

        let (control_req_tx, _control_req_rx) = crossbeam::channel::unbounded();
        let (_control_notify_tx, control_notify_rx) = crossbeam::channel::unbounded();
        let mut daemon_tx = HashMap::new();
        daemon_tx.insert(0, daemon_1_comp_tx);
        let daemon_rx = vec![(0, daemon_1_cmd_rx)];
        let proxy_peer_tx = vec![proxy_0_tx.clone(), proxy_1_tx.clone()];
        let dev_info = DeviceInfo {
            host: HostIdent(sock_addr),
            cuda_device_idx: 1,
        };
        let proxy_1_resources = ProxyResources {
            device_info: dev_info,
            control_tx: control_req_tx,
            control_rx: control_notify_rx,
            daemon_tx,
            daemon_rx,
            proxy_peer_tx,
            proxy_peer_rx: proxy_1_rx,
            comms_init: HashMap::new(),
            communicators: HashMap::new(),
            global_registry: Arc::clone(&registry),
            transport_engines_tx: HashMap::new(),
            transport_engines_rx: Vec::new(),
            transport_submission_pool: HashMap::new(),
        };
        let mut proxy_1 = ProxyEngine {
            resources: proxy_1_resources,
            ops: WorkPool::new(),
        };
        std::thread::spawn(move || {
            unsafe {
                let error = cudaSetDevice(0);
                if error != cudaError::cudaSuccess {
                    panic!("cudaSetDevice");
                }
            }
            proxy_0.mainloop();
        });
        std::thread::spawn(move || {
            unsafe {
                let error = cudaSetDevice(1);
                if error != cudaError::cudaSuccess {
                    panic!("cudaSetDevice");
                }
            }
            proxy_1.mainloop();
        });
        let cmd = InitCommunicator {
            communicator_id: CommunicatorId(0),
            rank: 0,
            num_ranks: 2,
        };
        let cmd = ProxyCommand::InitCommunicator(cmd);
        daemon_0_cmd_tx.send(cmd).unwrap();
        let cmd = InitCommunicator {
            communicator_id: CommunicatorId(0),
            rank: 1,
            num_ranks: 2,
        };
        let cmd = ProxyCommand::InitCommunicator(cmd);
        daemon_1_cmd_tx.send(cmd).unwrap();
        let comp = daemon_0_comp_rx.recv().unwrap();
        match comp {
            ProxyCompletion::InitCommunicator => (),
            ProxyCompletion::AllGather => panic!(),
        }
        let comp = daemon_1_comp_rx.recv().unwrap();
        match comp {
            ProxyCompletion::InitCommunicator => (),
            ProxyCompletion::AllGather => panic!(),
        }

        unsafe {
            let error = cudaSetDevice(0);
            if error != cudaError::cudaSuccess {
                panic!("cudaSetDevice");
            }
        }
        const BUFFER_SIZE: usize = 8192;
        let dev_buf_0 = unsafe {
            let mut dev_ptr = std::ptr::null_mut();
            cudaMalloc(&mut dev_ptr, BUFFER_SIZE);
            dev_ptr
        };
        let mut buf = vec![1883i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()];
        buf.extend(vec![0i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]);
        unsafe { 
            cudaMemcpy(
                dev_buf_0, 
                buf.as_ptr() as *const _, 
                BUFFER_SIZE,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        }; 

        unsafe {
            let error = cudaSetDevice(1);
            if error != cudaError::cudaSuccess {
                panic!("cudaSetDevice");
            }
        }
        let dev_buf_1 = unsafe {
            let mut dev_ptr = std::ptr::null_mut();
            cudaMalloc(&mut dev_ptr, BUFFER_SIZE);
            dev_ptr
        };
        let mut buf = vec![0i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()];
        buf.extend(vec![2042i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]);
        unsafe { 
            cudaMemcpy(
                dev_buf_1, 
                buf.as_ptr() as *const _, 
                BUFFER_SIZE,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        }; 
        let cmd = AllGather {
            communicator_id: CommunicatorId(0),
            send_buf_addr: dev_buf_0 as usize,
            recv_buf_addr: dev_buf_0 as usize,
            size: BUFFER_SIZE / 2,
        };
        let cmd = ProxyCommand::AllGather(cmd);
        daemon_0_cmd_tx.send(cmd).unwrap();
        let cmd = AllGather {
            communicator_id: CommunicatorId(0),
            send_buf_addr: dev_buf_1 as usize + BUFFER_SIZE / 2,
            recv_buf_addr: dev_buf_1 as usize,
            size: BUFFER_SIZE / 2,
        };
        let cmd = ProxyCommand::AllGather(cmd);
        daemon_1_cmd_tx.send(cmd).unwrap();

        let comp = daemon_0_comp_rx.recv().unwrap();
        match comp {
            ProxyCompletion::InitCommunicator => panic!(),
            ProxyCompletion::AllGather => (),
        }
        let comp = daemon_1_comp_rx.recv().unwrap();
        match comp {
            ProxyCompletion::InitCommunicator => panic!(),
            ProxyCompletion::AllGather => (),
        }

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
        println!("OK");
            // let (endpoint_tx, endpoint_rx) = crossbeam::channel::unbounded();
            // let device_info = DeviceInfo {
            //     cuda_device_idx: idx,
            //     cuda_comp_cap: 0,
            // };
            // let mut proxy_engine = ProxyEngine {
            //     device_info,
            //     outstanding_ops: std::collections::LinkedList::new(),
            //     enqueue_ops: std::collections::LinkedList::new(),
            //     daemon_endpoint_rx: endpoint_rx,
            //     daemon_command_rx: Vec::new(),
            //     daemon_completion_tx: Vec::new(),
            //     communicators: HashMap::new(),
            //     global_resources: self.global_resources.clone(),
            //     hmem_senders: HashMap::new(),
            //     hmem_receivers: HashMap::new(),
            // };
            // self.proxy_cmd_endpoints_tx.push(endpoint_tx);
            // std::thread::spawn(move || {
            //     unsafe {
            //         let error = cudaSetDevice(idx as _);
            //         if error != cudaError::cudaSuccess {
            //             panic!("cudaSetDevice");
            //         }
            //     }
            //     proxy_engine.mainloop();
            // });
        Ok(())
    }

    fn dispatch(
        &mut self,
        buf: &mut [u8],
        sender: &SocketAddr,
        _cred: &UCred,
    ) -> anyhow::Result<()> {
        use ipc::control;
        let msg: control::Request = bincode::deserialize(buf).unwrap();
        match msg {
            control::Request::NewClient => {
                let client_path = sender
                    .as_pathname()
                    .ok_or_else(|| anyhow!("peer is unnamed, something is wrong"))?;

                let uuid = uuid::Uuid::new_v4();
                let instance_name = format!("{}-{}.sock", self.config.mccs_daemon_basename, uuid);
                let engine_path = self.config.mccs_daemon_prefix.join(instance_name);

                // create customer stub
                // let customer = ShmCustomer::accept(&self.sock, client_path, engine_path)?;

                // let daemon_id = self.daemon_cnt;
                // let num_devices = self.proxy_cmd_endpoints_tx.len();
                // let mut command_txs = Vec::with_capacity(num_devices);
                // let mut completion_rxs = Vec::with_capacity(num_devices);

                // for device_idx in 0..num_devices {
                //     let endpoint_tx = &mut self.proxy_cmd_endpoints_tx[device_idx];
                //     let (cmd_tx, cmd_rx) = crossbeam::channel::unbounded();
                //     let (cmp_tx, cmp_rx) = crossbeam::channel::unbounded();
                //     let proxy_endpoint = CommandEndpointProxy {
                //         daemon_id,
                //         command_rx: cmd_rx,
                //         completion_tx: cmp_tx,
                //     };
                //     endpoint_tx.send(proxy_endpoint).unwrap();
                //     command_txs.push(cmd_tx);
                //     completion_rxs.push(cmp_rx);
                // }

                // let mut engine = crate::daemon::engine::DaemonEngine {
                //     id: daemon_id,
                //     proxy_command_tx: command_txs,
                //     proxy_completion_rx: completion_rxs,
                //     device_mem: HashMap::new(),
                //     comm_delegation: HashMap::new(),
                //     customer,
                // };
                // std::thread::spawn(move || {
                //     engine.mainloop();
                // });
                self.daemon_cnt += 1;
                
                Ok(())
            }
        }
    }

}
