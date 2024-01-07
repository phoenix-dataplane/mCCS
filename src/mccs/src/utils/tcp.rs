use std::net::SocketAddr;
use std::net::{SocketAddrV4, SocketAddrV6, Ipv4Addr, Ipv6Addr};

use byteorder::{ByteOrder, LittleEndian};
use bytes::{BufMut, Buf};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpSocket, TcpStream};


pub fn async_listen(addr: &SocketAddr) -> std::io::Result<TcpListener> {
    let socket = if addr.is_ipv4() {
        TcpSocket::new_v4()?
    } else {
        TcpSocket::new_v6()?
    };
    socket.bind(addr.to_owned())?;
    socket.set_reuseaddr(true)?;
    socket.set_reuseport(true)?;

    socket.listen(16384)
}

pub async fn async_accept(listener: &TcpListener, magic: u64) -> std::io::Result<TcpStream> {
    let mut buf = [0u8; std::mem::size_of::<u64>()];
    let stream = loop {
        let (mut stream, _) = listener.accept().await?;
        stream.read_exact(buf.as_mut_slice()).await?;
        let recv_magic = LittleEndian::read_u64(&buf);
        if recv_magic == magic {
            break stream;
        } else {
            log::warn!(
                "TCP listener accept: invalid magic {} != {}",
                recv_magic,
                magic
            );
        }
    };
    Ok(stream)
}

pub async fn async_connect(addr: &SocketAddr, magic: u64) -> std::io::Result<TcpStream> {
    let mut stream = TcpStream::connect(addr).await?;
    stream.set_nodelay(true)?;
    let mut buf = [0u8; std::mem::size_of::<u64>()];
    LittleEndian::write_u64(&mut buf, magic);
    stream.write_all(&buf).await?;
    Ok(stream)
}

pub fn encode_socket_addr<B: BufMut>(sock_addr: &SocketAddr, buf: &mut B) {
    match sock_addr {
        SocketAddr::V4(addr) => {
            buf.put_u8(4);
            buf.put_slice(&addr.ip().octets());
            buf.put_u16(addr.port());
        }
        SocketAddr::V6(addr) => {
            buf.put_u8(6);
            buf.put_slice(&addr.ip().octets());
            buf.put_u16(addr.port());
            buf.put_u32(addr.flowinfo());
            buf.put_u32(addr.scope_id());
        }
    }
}

pub fn decode_socket_addr<B: Buf>(buf: &mut B) -> SocketAddr {
    let addr_type = buf.get_u8();
    match addr_type {
        4 => {
            let port = buf.get_u16();
            let mut octets = [0u8; 4];
            buf.copy_to_slice(&mut octets);
            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::from(octets), port))
        }
        6 => {
            let mut octets = [0u8; 16];
            buf.copy_to_slice(&mut octets);
            let port = buf.get_u16();
            let flowinfo = buf.get_u32();
            let scope_id = buf.get_u32();
            SocketAddr::V6(SocketAddrV6::new(
                Ipv6Addr::from(octets),
                port,
                flowinfo,
                scope_id,
            ))
        }
        _ => panic!("unexpected address type"),
    }
}