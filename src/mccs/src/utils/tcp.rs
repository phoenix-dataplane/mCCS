use byteorder::{ByteOrder, LittleEndian};
use std::net::SocketAddr;
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
    log::debug!("async_connect to {addr}");
    let mut stream = TcpStream::connect(addr).await?;
    let mut buf = [0u8; std::mem::size_of::<u64>()];
    LittleEndian::write_u64(&mut buf, magic);
    stream.write_all(&buf).await?;
    Ok(stream)
}
