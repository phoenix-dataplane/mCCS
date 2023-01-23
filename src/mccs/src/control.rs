use std::collections::HashMap;
use std::fs;
use std::io;
use std::os::unix::net::{SocketAddr, UCred};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use std::time::Duration;

use anyhow::anyhow;
use crossbeam::channel::Sender;
use dashmap::DashMap;

use cuda_runtime_sys::cudaError;
use cuda_runtime_sys::cudaGetDeviceCount;
use cuda_runtime_sys::cudaSetDevice;
use ipc::customer::ShmCustomer;
use ipc::unix::DomainSocket;

use crate::config::Config;
use crate::daemon::DaemonId;
use crate::proxy::DeviceInfo;
use crate::proxy::command::CommandEndpointProxy;
use crate::proxy::engine::ProxyEngine;
use crate::resources::GlobalResources;
use crate::transport::registry::TransportSetupRegistry;

pub struct Control {
    sock: DomainSocket,
    config: Config,
    proxy_cmd_endpoints_tx: Vec<Sender<CommandEndpointProxy>>,
    global_resources: Arc<GlobalResources>,
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

        let transport_setup = TransportSetupRegistry::new();

        let global_resources = GlobalResources {
            communicators: DashMap::new(),
            transport_setup: transport_setup,
        };

        let mut control = Control {
            sock,
            config,
            proxy_cmd_endpoints_tx: Vec::new(),
            global_resources: Arc::new(global_resources),
            daemon_cnt: 0,
        };
        
        control.create_proxies().unwrap();
        control
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

    fn create_proxies(&mut self) -> anyhow::Result<()> {
        let mut num_devices = 0;
        unsafe {
            let error = cudaGetDeviceCount(&mut num_devices as *mut _);
            if error != cudaError::cudaSuccess {
                panic!("cudaGetDeviceCount");
            }
        }
        for idx in 0..(num_devices as usize) {
            let (endpoint_tx, endpoint_rx) = crossbeam::channel::unbounded();
            let device_info = DeviceInfo {
                cuda_device_idx: idx,
                cuda_comp_cap: 0,
            };
            let mut proxy_engine = ProxyEngine {
                device_info,
                outstanding_ops: std::collections::LinkedList::new(),
                daemon_endpoint_rx: endpoint_rx,
                daemon_command_rx: Vec::new(),
                daemon_completion_tx: Vec::new(),
                communicators: HashMap::new(),
                global_resources: self.global_resources.clone(),
                hmem_senders: HashMap::new(),
                hmem_receivers: HashMap::new(),
            };
            self.proxy_cmd_endpoints_tx.push(endpoint_tx);
            std::thread::spawn(move || {
                unsafe {
                    let error = cudaSetDevice(idx as _);
                    if error != cudaError::cudaSuccess {
                        panic!("cudaSetDevice");
                    }
                }
                proxy_engine.mainloop();
            });
        }
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
                let customer = ShmCustomer::accept(&self.sock, client_path, engine_path)?;

                let daemon_id = self.daemon_cnt;
                let num_devices = self.proxy_cmd_endpoints_tx.len();
                let mut command_txs = Vec::with_capacity(num_devices);
                let mut completion_rxs = Vec::with_capacity(num_devices);

                for device_idx in 0..num_devices {
                    let endpoint_tx = &mut self.proxy_cmd_endpoints_tx[device_idx];
                    let (cmd_tx, cmd_rx) = crossbeam::channel::unbounded();
                    let (cmp_tx, cmp_rx) = crossbeam::channel::unbounded();
                    let proxy_endpoint = CommandEndpointProxy {
                        daemon_id,
                        command_rx: cmd_rx,
                        completion_tx: cmp_tx,
                    };
                    endpoint_tx.send(proxy_endpoint).unwrap();
                    command_txs.push(cmd_tx);
                    completion_rxs.push(cmp_rx);
                }

                let mut engine = crate::daemon::engine::DaemonEngine {
                    id: daemon_id,
                    proxy_command_tx: command_txs,
                    proxy_completion_rx: completion_rxs,
                    device_mem: HashMap::new(),
                    comm_delegation: HashMap::new(),
                    customer,
                };
                std::thread::spawn(move || {
                    engine.mainloop();
                });
                self.daemon_cnt += 1;
                
                Ok(())
            }
        }
    }

}
