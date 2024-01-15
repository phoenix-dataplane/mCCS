use std::net::SocketAddr;
use std::sync::Arc;

use byteorder::{ByteOrder, LittleEndian};
use bytes::buf::{Buf, BufMut};
use smol::io::{AsyncReadExt, AsyncWriteExt};
use smol::lock::Mutex;
use smol::net::{TcpListener, TcpStream};
use socket2::Socket;

use super::{BootstrapError, BootstrapHandle, BootstrapState, UnexpectedConn};
use crate::bootstrap::BootstrapRing;
use crate::utils::tcp;

const EXCHANGE_INFO_SEND_SIZE: usize = 80;
const SOCK_ADDR_SEND_SIZE: usize = 32;

pub struct BootstrapExchangeInfo {
    pub rank: usize,
    pub num_ranks: usize,
    pub listen_addr_root: SocketAddr,
    pub listen_addr: SocketAddr,
}

impl BootstrapExchangeInfo {
    fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u64(self.rank as u64);
        buf.put_u64(self.num_ranks as u64);
        tcp::encode_socket_addr(&self.listen_addr_root, buf);
        tcp::encode_socket_addr(&self.listen_addr, buf);
    }

    fn decode<B: Buf>(buf: &mut B) -> Self {
        let rank = buf.get_u64() as usize;
        let num_ranks = buf.get_u64() as usize;
        let listen_addr_root = tcp::decode_socket_addr(buf);
        let listen_addr = tcp::decode_socket_addr(buf);
        Self {
            rank,
            num_ranks,
            listen_addr_root,
            listen_addr,
        }
    }
}

pub async fn bootstrap_net_send(stream: &mut TcpStream, data: &[u8]) -> Result<(), BootstrapError> {
    let mut buf = [0u8; 4];
    LittleEndian::write_u32(&mut buf, data.len() as u32);
    stream.write_all(&buf).await?;
    stream.write_all(data).await?;
    Ok(())
}

pub async fn bootstrap_net_recv(
    stream: &mut TcpStream,
    data: &mut [u8],
) -> Result<(), BootstrapError> {
    let mut buf = [0u8; 4];
    stream.read_exact(&mut buf).await?;
    let recv_size = LittleEndian::read_u32(&buf);
    if recv_size != data.len() as u32 {
        Err(BootstrapError::RecvSizeMismatch(
            recv_size,
            data.len() as u32,
        ))?;
    }
    stream.read_exact(data).await?;
    Ok(())
}

pub async fn bootstrap_root(listen_sock: Socket, magic: u64) -> Result<(), BootstrapError> {
    listen_sock.listen(16384)?;
    let listener: std::net::TcpListener = listen_sock.into();
    let listener = TcpListener::try_from(listener)?;

    let mut recv_buf = [0u8; EXCHANGE_INFO_SEND_SIZE];
    let mut stream = tcp::async_accept(&listener, magic).await?;
    log::trace!(
        "before net receive: Bootstrap root listen {:?}",
        listener.local_addr(),
    );
    bootstrap_net_recv(&mut stream, recv_buf.as_mut_slice()).await?;
    let mut buf = recv_buf.as_slice();
    let exchange_info = BootstrapExchangeInfo::decode(&mut buf);

    let mut rank_addrs = vec![None; exchange_info.num_ranks];
    rank_addrs[exchange_info.rank] = Some(exchange_info.listen_addr);
    let mut rank_addrs_root = vec![None; exchange_info.num_ranks];
    rank_addrs_root[exchange_info.rank] = Some(exchange_info.listen_addr_root);
    let num_ranks = exchange_info.num_ranks;
    let mut received = 1;
    log::trace!(
        "Bootstrap root received check-in from rank {}",
        exchange_info.rank
    );

    while received < num_ranks {
        let mut stream = tcp::async_accept(&listener, magic).await?;
        bootstrap_net_recv(&mut stream, recv_buf.as_mut_slice()).await?;
        let mut buf = recv_buf.as_slice();
        let exchange_info = BootstrapExchangeInfo::decode(&mut buf);
        if exchange_info.num_ranks != num_ranks {
            Err(BootstrapError::NumRanksMismatch(
                exchange_info.num_ranks,
                num_ranks,
            ))?;
        }
        if exchange_info.rank >= num_ranks {
            Err(BootstrapError::RankOverflow(exchange_info.rank))?;
        }
        if rank_addrs[exchange_info.rank].is_some() {
            Err(BootstrapError::DuplicatedCheckIn(exchange_info.rank))?;
        }
        rank_addrs[exchange_info.rank] = Some(exchange_info.listen_addr);
        rank_addrs_root[exchange_info.rank] = Some(exchange_info.listen_addr_root);
        received += 1;
        log::trace!(
            "Bootstrap root received check-in from rank {}",
            exchange_info.rank
        );
    }
    let mut send_buf = [0u8; SOCK_ADDR_SEND_SIZE];
    for r in 0..num_ranks {
        let next = (r + 1) % num_ranks;
        let connect_addr = rank_addrs_root[r].as_ref().unwrap();
        log::trace!("Bootstrap root connecting to {:?}", connect_addr);
        let mut stream = tcp::async_connect(connect_addr, magic).await?;
        let mut buf = send_buf.as_mut();
        let send_addr = rank_addrs[next].as_ref().unwrap();
        tcp::encode_socket_addr(send_addr, &mut buf);
        let data = send_buf.as_slice();
        bootstrap_net_send(&mut stream, data).await?;
    }
    log::trace!("Bootstrap root has sent out all peer addresses");
    Ok(())
}

pub fn bootstrap_create_root(
    listen_addr: &SocketAddr,
) -> Result<(Socket, BootstrapHandle), BootstrapError> {
    let socket = if listen_addr.is_ipv4() {
        Socket::new(socket2::Domain::IPV4, socket2::Type::STREAM, None)?
    } else {
        Socket::new(socket2::Domain::IPV6, socket2::Type::STREAM, None)?
    };
    let addr = listen_addr.to_owned().into();
    socket.bind(&addr)?;
    socket.set_reuse_address(true)?;
    socket.set_reuse_port(true)?;
    socket.set_nonblocking(true)?;

    let addr = socket.local_addr()?.as_socket().unwrap();
    let magic = rand::random();
    let handle = BootstrapHandle { addr, magic };
    Ok((socket, handle))
}

pub async fn bootstrap_all_gather_internal(
    ring_send: &mut TcpStream,
    ring_recv: &mut TcpStream,
    rank: usize,
    num_ranks: usize,
    data: &mut [u8],
) -> Result<(), BootstrapError> {
    assert_eq!(data.len() % num_ranks, 0);
    let size = data.len() / num_ranks;
    for i in 0..(num_ranks - 1) {
        let recv_slice_idx = (rank + num_ranks - i - 1) % num_ranks;
        let send_slice_idx = (rank + num_ranks - i) % num_ranks;

        let send_data = &data[send_slice_idx * size..(send_slice_idx + 1) * size];
        // send slice to the right
        bootstrap_net_send(ring_send, send_data).await?;
        let recv_data = &mut data[recv_slice_idx * size..(recv_slice_idx + 1) * size];
        // recv slice from the left
        bootstrap_net_recv(ring_recv, recv_data).await?;
    }
    Ok(())
}

impl BootstrapState {
    pub async fn init(
        handle: BootstrapHandle,
        listen_addr: SocketAddr,
        rank: usize,
        num_ranks: usize,
    ) -> Result<BootstrapState, BootstrapError> {
        let mut listen_addr = listen_addr.to_owned();
        listen_addr.set_port(0);

        let peer_listener = tcp::async_listen(&listen_addr)?;
        let peer_listen_addr = peer_listener.local_addr()?;
        let root_listener = tcp::async_listen(&listen_addr)?;
        let root_listen_addr = root_listener.local_addr()?;
        log::trace!(
            "Rank {} of {} root listening on {:?}",
            rank,
            num_ranks,
            root_listen_addr
        );

        if num_ranks > 128 {
            let dura = std::time::Duration::from_millis(rank as u64);
            log::trace!("Rank {} delaying connection to root by {} ms", rank, rank);
            smol::Timer::after(dura).await;
        }

        log::trace!(
            "Rank {} of {} connecting to root {:?}",
            rank,
            num_ranks,
            handle.addr
        );
        let mut stream = tcp::async_connect(&handle.addr, handle.magic).await?;
        let info = BootstrapExchangeInfo {
            rank,
            num_ranks,
            listen_addr_root: root_listen_addr,
            listen_addr: peer_listen_addr,
        };

        // send info on my listening socket to root
        let mut send_buf = [0u8; EXCHANGE_INFO_SEND_SIZE];
        let mut buf = send_buf.as_mut();
        info.encode(&mut buf);
        bootstrap_net_send(&mut stream, send_buf.as_slice()).await?;

        // get info on my next rank in the bootstrap ring from root
        let mut stream = tcp::async_accept(&root_listener, handle.magic).await?;
        let mut recv_buf = [0u8; SOCK_ADDR_SEND_SIZE];
        bootstrap_net_recv(&mut stream, recv_buf.as_mut_slice()).await?;
        let mut buf = recv_buf.as_slice();
        let next_addr = tcp::decode_socket_addr(&mut buf);

        let mut ring_send = tcp::async_connect(&next_addr, handle.magic).await?;
        let mut ring_recv = tcp::async_accept(&peer_listener, handle.magic).await?;

        let mut all_addrs_buf = vec![0u8; SOCK_ADDR_SEND_SIZE * num_ranks];
        let mut my_buf =
            &mut all_addrs_buf[rank * SOCK_ADDR_SEND_SIZE..(rank + 1) * SOCK_ADDR_SEND_SIZE];
        tcp::encode_socket_addr(&peer_listen_addr, &mut my_buf);
        bootstrap_all_gather_internal(
            &mut ring_send,
            &mut ring_recv,
            rank,
            num_ranks,
            all_addrs_buf.as_mut_slice(),
        )
        .await?;
        let mut peer_listen_addrs = Vec::with_capacity(num_ranks);
        for i in 0..num_ranks {
            let mut buf = &all_addrs_buf[i * SOCK_ADDR_SEND_SIZE..(i + 1) * SOCK_ADDR_SEND_SIZE];
            let addr = tcp::decode_socket_addr(&mut buf);
            peer_listen_addrs.push(addr);
        }
        let state = BootstrapState {
            rank,
            num_ranks,
            peer_addrs: peer_listen_addrs,
            listener: peer_listener,
            magic: handle.magic,
            ring: Mutex::new(BootstrapRing {
                ring_send,
                ring_recv,
            }),
            unexpected_connections: Mutex::new(Vec::new()),
        };
        Ok(state)
    }
}

impl BootstrapState {
    fn unexpected_enqueue(&self, peer: usize, tag: u32, stream: TcpStream) {
        let mut connections = self
            .unexpected_connections
            .try_lock()
            .ok_or_else(|| BootstrapError::MutexAcquire)
            .unwrap();
        let conn = UnexpectedConn { peer, tag, stream };
        connections.push(conn);
    }

    fn unexpected_dequeue(&self, peer: usize, tag: u32) -> Option<TcpStream> {
        let mut connections = self
            .unexpected_connections
            .try_lock()
            .ok_or_else(|| BootstrapError::MutexAcquire)
            .unwrap();
        let idx = connections
            .iter()
            .position(|c| c.peer == peer && c.tag == tag);
        if let Some(idx) = idx {
            let conn = connections.swap_remove(idx);
            Some(conn.stream)
        } else {
            None
        }
    }

    pub async fn bootstrap_send_internal(
        &self,
        peer: usize,
        tag: u32,
        data: &[u8],
    ) -> Result<(), BootstrapError> {
        log::trace!(
            "Bootstrap rank {:?} send to peer {} ({:?}) tag {}",
            self.rank,
            peer,
            self.peer_addrs[peer],
            tag
        );
        let mut stream = tcp::async_connect(&self.peer_addrs[peer], self.magic).await?;
        let mut buf = [0u8; 8];
        LittleEndian::write_u64(&mut buf, self.rank as u64);
        stream.write_all(&buf).await?;
        let mut buf = [0u8; 4];
        LittleEndian::write_u32(&mut buf, tag);
        stream.write_all(&buf).await?;

        bootstrap_net_send(&mut stream, data).await?;
        Ok(())
    }

    pub async fn bootstrap_recv_internal(
        &self,
        peer: usize,
        tag: u32,
        recv_buf: &mut [u8],
    ) -> Result<(), BootstrapError> {
        if let Some(mut stream) = self.unexpected_dequeue(peer, tag) {
            bootstrap_net_recv(&mut stream, recv_buf).await?;
            return Ok(());
        }
        loop {
            let mut stream = tcp::async_accept(&self.listener, self.magic).await?;
            let peer_addr = stream.peer_addr()?;
            log::trace!(
                "Bootstrap rank {:?} recv from peer {} ({:?}) tag {}",
                self.rank,
                peer,
                peer_addr,
                tag
            );

            let mut buf = [0u8; 8];
            stream.read_exact(&mut buf).await?;
            let recv_peer = LittleEndian::read_u64(&buf) as usize;
            let mut buf = [0u8; 4];
            stream.read_exact(&mut buf).await?;
            let recv_tag = LittleEndian::read_u32(&buf);

            if recv_peer == peer && recv_tag == tag {
                bootstrap_net_recv(&mut stream, recv_buf).await?;
                return Ok(());
            } else {
                self.unexpected_enqueue(recv_peer, recv_tag, stream);
            }
        }
    }
    pub async fn bootstrap_send(
        self: Arc<Self>,
        peer: usize,
        tag: u32,
        data: Vec<u8>,
    ) -> Result<(), BootstrapError> {
        let mut stream = tcp::async_connect(&self.peer_addrs[peer], self.magic).await?;

        let mut buf = [0u8; 8];
        LittleEndian::write_u64(&mut buf, self.rank as u64);
        stream.write_all(&buf).await?;
        let mut buf = [0u8; 4];
        LittleEndian::write_u32(&mut buf, tag);
        stream.write_all(&buf).await?;
        bootstrap_net_send(&mut stream, data.as_slice()).await?;
        Ok(())
    }

    // When wrapping the future in a Box, the async block must have 'static lifetime.
    // It must own all input parameters, as well as the return values
    pub async fn bootstrap_recv(
        self: Arc<Self>,
        peer: usize,
        tag: u32,
        size: u32,
    ) -> Result<Vec<u8>, BootstrapError> {
        if let Some(mut stream) = self.unexpected_dequeue(peer, tag) {
            let mut buf = vec![0u8; size as usize];
            bootstrap_net_recv(&mut stream, buf.as_mut_slice()).await?;
            return Ok(buf);
        }
        loop {
            let mut stream = tcp::async_accept(&self.listener, self.magic).await?;

            let mut buf = [0u8; 8];
            stream.read_exact(&mut buf).await?;
            let recv_peer = LittleEndian::read_u64(&buf) as usize;
            let mut buf = [0u8; 4];
            stream.read_exact(&mut buf).await?;
            let recv_tag = LittleEndian::read_u32(&buf);

            if recv_peer == peer && recv_tag == tag {
                let mut buf = vec![0u8; size as usize];
                bootstrap_net_recv(&mut stream, buf.as_mut_slice()).await?;
                return Ok(buf);
            } else {
                self.unexpected_enqueue(recv_peer, recv_tag, stream);
            }
        }
    }

    // slice only contains my rank's data, the returned vector contains all ranks' data,
    // which has a size of slice.len() * num_ranks
    pub async fn bootstrap_all_gather(
        self: Arc<Self>,
        slice: Vec<u8>,
    ) -> Result<Vec<u8>, BootstrapError> {
        let mut data = vec![0u8; slice.len() * self.num_ranks];
        let slice_size = slice.len();
        let rank = self.rank;
        let num_ranks = self.num_ranks;
        let my_rank_data = &mut data.as_mut_slice()[rank * slice_size..(rank + 1) * slice_size];
        my_rank_data.copy_from_slice(slice.as_slice());

        let mut ring = self
            .ring
            .try_lock()
            .ok_or_else(|| BootstrapError::MutexAcquire)?;
        for i in 0..(num_ranks - 1) {
            let recv_slice_idx = (rank + num_ranks - i - 1) % num_ranks;
            let send_slice_idx = (rank + num_ranks - i) % num_ranks;

            let send_data =
                &data.as_slice()[send_slice_idx * slice_size..(send_slice_idx + 1) * slice_size];
            // send slice to the right
            bootstrap_net_send(&mut ring.ring_send, send_data).await?;
            let recv_data = &mut data.as_mut_slice()
                [recv_slice_idx * slice_size..(recv_slice_idx + 1) * slice_size];
            // recv slice from the left
            bootstrap_net_recv(&mut ring.ring_recv, recv_data).await?;
        }
        log::trace!(
            "Bootstrap AllGather done: rank {} of {}, size: {}",
            rank,
            num_ranks,
            slice_size
        );
        Ok(data)
    }

    // ranks: a list that maps ranks in a subgroup to global ranks
    // rank: my rank in the subgroup
    pub async fn bootstrap_barrier(
        self: Arc<Self>,
        ranks: Vec<usize>,
        rank: usize,
        tag: u32,
    ) -> Result<(), BootstrapError> {
        let num_ranks = ranks.len();
        if num_ranks == 1 {
            return Ok(());
        }
        let mut data = [0u8; 1];
        let mut mask = 1;
        while mask < num_ranks {
            let src_idx = (rank + num_ranks - mask) % num_ranks;
            let dst_idx = (rank + mask) % num_ranks;
            self.bootstrap_send_internal(ranks[dst_idx], tag, data.as_slice())
                .await?;
            self.bootstrap_recv_internal(ranks[src_idx], tag, data.as_mut_slice())
                .await?;
            mask <<= 1;
        }
        log::trace!("Bootstrap barrier done: rank {} of {}", rank, num_ranks);
        Ok(())
    }

    pub async fn intra_node_all_gather(
        self: Arc<Self>,
        ranks: Vec<usize>,
        rank: usize,
        slice: Vec<u8>,
    ) -> Result<Vec<u8>, BootstrapError> {
        let num_ranks = ranks.len();
        if num_ranks == 1 {
            return Ok(slice);
        }
        let mut data = vec![0u8; slice.len() * num_ranks];
        let slice_size = slice.len();
        let my_rank_data = &mut data.as_mut_slice()[rank * slice_size..(rank + 1) * slice_size];
        my_rank_data.copy_from_slice(slice.as_slice());
        for i in 1..num_ranks {
            let src_idx = (rank + num_ranks - i) % num_ranks;
            let dst_idx = (rank + i) % num_ranks;
            let send_data = &data.as_slice()[rank * slice_size..(rank + 1) * slice_size];
            self.bootstrap_send_internal(ranks[dst_idx], i as u32, send_data)
                .await?;
            let recv_data =
                &mut data.as_mut_slice()[src_idx * slice_size..(src_idx + 1) * slice_size];
            self.bootstrap_recv_internal(ranks[src_idx], i as u32, recv_data)
                .await?;
        }
        log::trace!(
            "Bootstrap intra node AllGather done: rank {} of {}, size: {}",
            rank,
            num_ranks,
            slice_size
        );
        Ok(data)
    }
}
