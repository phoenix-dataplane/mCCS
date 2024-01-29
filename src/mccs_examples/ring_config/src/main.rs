use std::io::Write;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian};
use serde::{Deserialize, Serialize};
use structopt::StructOpt;

use ipc::mccs::reconfig::{CommPatternReconfig, ExchangeReconfigCommand};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    mccs_addrs: Vec<IpAddr>,
    mccs_port: u16,
    comm_patterns_reconfig: Vec<CommPatternReconfig>,
}

impl Config {
    fn from_path<P: AsRef<Path>>(path: P) -> Config {
        let content = std::fs::read_to_string(path).unwrap();
        let config = toml::from_str(&content).unwrap();
        config
    }
}

#[derive(Debug, Clone, StructOpt)]
#[structopt(name = "Comm pattern configurator")]
struct Opts {
    // Path to toml traffic trace
    #[structopt(long, short = "c")]
    config: PathBuf,
}

const EXCHANGE_MAGIC: u64 = 0x424ab9f2fc4b9d6e;

fn main() {
    let opts = Opts::from_args();
    let config = Config::from_path(opts.config);

    let pattern_config = config.comm_patterns_reconfig.clone();
    let command = ExchangeReconfigCommand::CommPatternReconfig(pattern_config);
    let encoded = bincode::serialize(&command).unwrap();

    for addr in config.mccs_addrs.iter() {
        let addr = SocketAddr::new(*addr, config.mccs_port);
        let mut buf = [0u8; 5];
        buf[0] = 1;
        LittleEndian::write_u32(&mut buf[1..], encoded.len() as u32);
        let mut magic_buf = [0u8; std::mem::size_of::<u64>()];
        LittleEndian::write_u64(&mut magic_buf, EXCHANGE_MAGIC);
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.set_nodelay(true).unwrap();
        stream.write_all(&magic_buf).unwrap();
        stream.write_all(&buf).unwrap();
        stream.write_all(encoded.as_slice()).unwrap();

        println!("Sent command to {}", addr);
    }
}
