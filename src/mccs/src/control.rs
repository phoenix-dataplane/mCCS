use std::fs;
use std::io;
use std::os::unix::net::{SocketAddr, UCred};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::anyhow;

use ipc::customer::ShmCustomer;
use ipc::unix::DomainSocket;

use crate::config::Config;

pub struct Control {
    sock: DomainSocket,
    config: Config,
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

        Control {
            sock,
            config,
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

                let mut engine = crate::daemon::engine::DaemonEngine {
                    customer
                };
                std::thread::spawn(move || {
                    engine.mainloop();
                });
                    
                Ok(())
            }
        }
    }

}
