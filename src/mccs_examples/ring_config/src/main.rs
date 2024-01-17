use std::io::Write;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream};

use byteorder::{ByteOrder, LittleEndian};

use ipc::mccs::reconfig::{
    ChannelPattern, CommPatternReconfig, CommunicatorId, ExchangeReconfigCommand,
};

fn main() {
    let ring = vec![2, 1, 0];
    let config = CommPatternReconfig {
        communicator_id: CommunicatorId(42),
        channels: vec![ChannelPattern {
            channel_id: 0,
            ring,
        }],
    };
    let cmd = ExchangeReconfigCommand::CommPatternReconfig(config);

    let encoded = bincode::serialize(&cmd).unwrap();
    let mut buf = [0u8; 5];
    buf[0] = 1;
    LittleEndian::write_u32(&mut buf[1..], encoded.len() as u32);

    let mut stream = TcpStream::connect(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::new(192, 168, 211, 130)),
        5000,
    ))
    .unwrap();
    stream.set_nodelay(true).unwrap();
    stream.write_all(&buf).unwrap();
    stream.write_all(encoded.as_slice()).unwrap();

    let mut stream = TcpStream::connect(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::new(192, 168, 211, 195)),
        5000,
    ))
    .unwrap();
    stream.set_nodelay(true).unwrap();
    stream.write_all(&buf).unwrap();
    stream.write_all(encoded.as_slice()).unwrap();
}
