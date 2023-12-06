use std::os::fd::RawFd;
use std::pin::Pin;
use std::marker::PhantomPinned;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::ffi::c_void;

use async_std::net::{TcpStream, TcpListener};
use once_cell::sync::OnceCell;
use thiserror::Error;
use socket2::SockAddr;
use volatile::{VolatilePtr, map_field};

use ibverbs::{Context, ibv_access_flags};
use ibverbs::{ProtectionDomain, CompletionQueue, MemoryRegionAlloc, QueuePair, MemoryRegionRegister};
use ibverbs::ffi::{ibv_device_attr, ibv_port_attr};
use ibverbs::ffi::{ibv_sge, ibv_send_wr};
use ibverbs::ffi::{ibv_send_flags, ibv_wr_opcode};

use super::NET_MAX_REQUESTS;

const IB_MAX_RECVS: usize = 8;
const IB_MAX_QPS: usize = 128;
const IB_MAX_REQUESTS: usize = NET_MAX_REQUESTS * IB_MAX_RECVS;


const IB_AR_THRESHOLD: usize = 8192;

macro_rules! ibv_check {
    ($ibv_op:expr) => {{
        let errno = $ibv_op;
        if errno != 0 {
            Err(std::io::Error::from_raw_os_error(errno))?;
        }
    }};
}

#[derive(Debug, Error)]
pub enum IbError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Insufficient recv size {recv} to match send size {send}")]
    SendOverflow { send: usize, recv: usize },
    #[error("Maximum number of outstanding requests of {} reached", IB_MAX_REQUESTS)]
    RequestBufferFull,
}

const IBV_WIDTHS: [u32; 5] = [1, 4, 8, 12, 2];
const IBV_SPEEDS: [u32; 8] = [
    2500, // SDR
    5000, // DDR
    10000, // QDR
    10000, // FDR10
    14000, // FDR
    25000, // EDR
    50000, // HDR
    100000, // NDR
];

#[inline]
fn get_ib_width(active_width: u8) -> u32 {
    IBV_WIDTHS[active_width.trailing_zeros() as usize]
}

#[inline]
fn get_ib_speed(active_speed: u8) -> u32 {
    IBV_SPEEDS[active_speed.trailing_zeros() as usize]
}

pub struct IbDeviceResources {
    pd: Option<Arc<ProtectionDomain<'static>>>,
    mr_cache: Vec<Arc<MemoryRegionRegister>>,
}

pub struct IbDevice {
    device: usize,
    guid: u64,
    port: u8,
    link: u8,
    speed: u32,
    context: Arc<Context>,
    device_name: String,
    pci_path: String,
    real_port: u8,
    max_qp: u32,
    adaptive_routing: bool,
    resources: Mutex<IbDeviceResources>,
    _pinned: PhantomPinned,
}

pub struct RdmaTransportConfig {
    gid_index: u32,
    timeout: u32,
    retry_count: u32,
    pkey: u32,
    use_inline: bool,
}

struct RdmaTransportContext {
    devices: Vec<IbDevice>,
    listen_addr: SockAddr,
    page_size: usize,
}

pub struct RdmaTransportProvider(OnceCell<RdmaTransportContext>);

pub static RDMA_TRANSPORT: RdmaTransportProvider = RdmaTransportProvider(OnceCell::new());

fn get_pci_path(device_name: &str, current_devices: &Vec<IbDevice>) -> Result<(String, u8), IbError> {
    let device_path = format!("/sys/class/infiniband/{}/device", device_name);
    //  char* p = realpath(devicePath, NULL);
    let real_path = std::fs::canonicalize(device_path)?;
    let mut p = real_path.to_str().unwrap().to_string();
    let len = p.len();
    // Merge multi-port NICs into the same PCI device
    p.replace_range(len-1..len, "0");
    // Also merge virtual functions (VF) into the same device
    p.replace_range(len-3..len-2, "0");
    let mut real_port = 0;
    for device in current_devices.iter() {
        if device.pci_path == p {
            real_port += 1;
        }
    }
    Ok((p, real_port))
}   

fn init_transport_context() -> Result<RdmaTransportContext, IbError> {
    let devices = ibverbs::devices()?;
    let mut devices_ctx = Vec::with_capacity(devices.len());
    for (idx, dev) in devices.iter().enumerate() {
        let context = Arc::new(dev.open()?);
        let mut dev_attr = ibv_device_attr::default();
        unsafe {
            ibv_check!(ibverbs::ffi::ibv_query_device(context.ctx, &mut dev_attr));
        }
        for port in 1..=dev_attr.phys_port_cnt {
            let mut port_attr = ibv_port_attr::default();
            unsafe { 
                let ptr = &mut port_attr as *mut ibv_port_attr as *mut _;
                ibv_check!(ibverbs::ffi::ibv_query_port(context.ctx, port, ptr));
            }
            if port_attr.state != ibverbs::ffi::ibv_port_state::IBV_PORT_ACTIVE {
                continue;
            }
            if port_attr.link_layer != ibverbs::ffi::IBV_LINK_LAYER_ETHERNET as u8
                && port_attr.link_layer != ibverbs::ffi::IBV_LINK_LAYER_INFINIBAND as u8 {
                continue;
            }

            // TODO: check against user specified HCAs/ports
            let device_name = if let Some(name) = dev.name() {
                name.to_str().unwrap().to_string()
            } else {
                String::new()
            };
            log::info!(
                "Initialize RDMA device [{idx}] {device_name}:{port}, {}", 
                if port_attr.link_layer == ibverbs::ffi::IBV_LINK_LAYER_INFINIBAND as u8 { "IB" } else { "RoCE" }
            );

            let speed = get_ib_speed(port_attr.active_speed) * get_ib_width(port_attr.active_width);
            let (pci_path, real_port) = get_pci_path(&device_name, &devices_ctx)?;
            let ar = port_attr.link_layer == ibverbs::ffi::IBV_LINK_LAYER_INFINIBAND as u8;
            let resources = IbDeviceResources {
                pd: None,
                mr_cache: Vec::new(),
            };
            let dev_context = IbDevice {
                device: idx,
                guid: dev_attr.sys_image_guid,
                port,
                link: port_attr.link_layer,
                speed,
                context: Arc::clone(&context),
                device_name,
                pci_path,
                real_port,
                max_qp: dev_attr.max_qp as _,
                adaptive_routing: ar,
                resources: Mutex::new(resources),
                _pinned: PhantomPinned,
            };
            devices_ctx.push(dev_context);
        }
    }
    use nix::unistd::{sysconf, SysconfVar};
    let page_size = sysconf(SysconfVar::PAGE_SIZE)
        .map_err(|e| std::io::Error::from_raw_os_error(e as i32))?
        .unwrap() as usize;
    todo!()
}

#[derive(Clone, Copy)]
pub struct SendRequest {
    size: usize,
    data: *mut c_void,
    lkey: u32,
    offset: usize,
}

#[derive(Clone, Copy)]
pub struct RecvRequest {
    size: [usize; IB_MAX_RECVS],
}

pub union SendRecvRequest {
    send: SendRequest,
    recv: RecvRequest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestType {
    Unused,
    Send,
    Recv,
    Flush,
}

pub struct IbRequest {
    events: u32,
    num_requests: u32,
    ty: RequestType,
    send_recv: SendRecvRequest,
}

pub struct IbVerbs<'ctx> {
    device: usize,
    pd: Arc<ProtectionDomain<'ctx>>,
    cq: Arc<CompletionQueue<'ctx>>,
    requests: [IbRequest; IB_MAX_REQUESTS],
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(32))]
pub struct IbSendFifo {
    addr: u64,
    // we use u32 instead of usize to make sure IbSendInfo is 32-byte
    size: u32,
    rkey: u32,
    num_requests: u32,
    tag: u32,
    idx: u64,
}

impl Default for IbSendFifo {
    fn default() -> Self {
        Self {
            addr: 0,
            size: 0,
            rkey: 0,
            num_requests: 0,
            tag: 0,
            idx: 0,
        }
    }
}

pub struct IbSendComm<'ctx> {
    verbs: IbVerbs<'ctx>,
    fifo: MemoryRegionAlloc<IbSendFifo>,
    fifo_head: u64,
    fifo_requests_idx: [[usize; IB_MAX_RECVS]; IB_MAX_REQUESTS],
    wrs: [ibv_send_wr; IB_MAX_RECVS + 1],
    sges: [ibv_sge; IB_MAX_RECVS],
    qps: Vec<QueuePair<'ctx>>,
    adaptive_routing: bool,
    _pin: PhantomPinned,
}

// IbSendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
// By default, Vec<T> guarantees that memory is properly aligned for type T
static_assertions::const_assert_eq!(std::mem::size_of::<IbSendFifo>() % 32, 0);

pub struct IbRemoteFifo {
    mr: MemoryRegionAlloc<IbSendFifo>,
    fifo_tail: u64,
    addr: u64,
    rkey: u64,
    flags: ibv_send_flags,
    sge: ibv_sge,
}

pub struct IbGpuFlush<'ctx> {
    enabled: bool,
    host_mr: Option<MemoryRegionAlloc<i32>>,
    sge: ibv_sge,
    qp: Option<QueuePair<'ctx>>,
}

pub struct IbRecvComm<'ctx> {
    verbs: IbVerbs<'ctx>,
    remote_fifo: IbRemoteFifo,
    qps: Vec<QueuePair<'ctx>>,
    flush: IbGpuFlush<'ctx>,
    _pinned: PhantomPinned,
}

pub struct IbConnectHandle {
    connect_addr: SockAddr,
    magic: u64,
}

pub async fn ib_connect(
    device: u32,
    handle: &IbConnectHandle,
) -> Result<IbSendComm<'static>, IbError> {
    let connect_addr =  handle.connect_addr.as_socket().unwrap();
    let stream = TcpStream::connect(&connect_addr);
    todo!()
}

pub struct IbMrHandle(Arc<MemoryRegionRegister>);
pub struct IbRequestId(usize);

pub fn ib_register_mr_dma_buf(
    verbs: &mut IbVerbs<'_>,
    data: *mut c_void,
    size: usize,
    offset: u64,
    fd: RawFd,
) -> Result<IbMrHandle, IbError> {
    let transport_ctx = RDMA_TRANSPORT.0.get().unwrap();
    let device_ctx = &transport_ctx.devices[verbs.device];

    let page_size = transport_ctx.page_size;
    let addr = data.addr() & (-(page_size as isize)) as usize;
    let pages = (data.addr() + size - addr + page_size - 1) / page_size;
    let size_aligned = pages * page_size;

    let mut guard = device_ctx.resources.lock().unwrap();
    let cached_mr = guard.mr_cache.iter().find(|mr| {
        if mr.addr() == addr && mr.size() == size_aligned {
            true
        } else {
            false
        }
    });
    if let Some(mr) = cached_mr {
        let handle = IbMrHandle(Arc::clone(mr));
        Ok(handle)
    } else {
        let flags = ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
            | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
            | ibv_access_flags::IBV_ACCESS_REMOTE_READ;
        let addr = addr as *mut c_void;
        let mr = if fd != -1 {
            verbs.pd.register_dmabuf_mr(addr, size_aligned, offset, fd, flags)?
        } else {
            // TODO: uses ibv_reg_mr_iova2 to support IBV_ACCESS_RELAXED_ORDERING
            verbs.pd.register_mr(addr, size_aligned, flags)?
        };
        let mr = Arc::new(mr);
        guard.mr_cache.push(Arc::clone(&mr));
        let handle = IbMrHandle(mr);
        Ok(handle)
    }
}

#[inline]
pub fn ib_register_mr(
    verbs: &mut IbVerbs<'_>,
    data: *mut c_void,
    size: usize,
) -> Result<IbMrHandle, IbError> {
    ib_register_mr_dma_buf(verbs, data, size, 0, -1)
}

pub fn ib_deregister_mr(
    verbs: &mut IbVerbs<'_>,
    handle: IbMrHandle,
) {
    std::mem::drop(handle);
}

fn ib_get_request(verbs: &mut IbVerbs<'_>) -> Option<usize> {
    for i in 0..IB_MAX_REQUESTS {
        let r = &mut verbs.requests[i];
        if r.ty == RequestType::Unused {
            r.events = 1;
            return Some(i);
        }
    }
    return None;
}

pub fn ib_multi_send(
    comm: Pin<&mut IbSendComm<'_>>,
    slot: usize,
) -> Result<(), IbError> {
    let comm = unsafe { comm.get_unchecked_mut() };
    let requests_idx = &comm.fifo_requests_idx[slot];
    let offset = slot * IB_MAX_RECVS;
    let slots_ptr = unsafe { comm.fifo.as_mut_ptr().add(offset) };

    let slot_non_null = unsafe { NonNull::new_unchecked(slots_ptr) };
    let slot_ptr = unsafe { VolatilePtr::new(slot_non_null) };
    let num_requests = map_field!(slot_ptr.num_requests).read() as usize;
    assert!(num_requests <= IB_MAX_RECVS);

    let mut wr_id = 0;
    for r in 0..num_requests {
        let wr = &mut comm.wrs[r];
        unsafe { 
            let wr_ptr = wr as *mut ibv_send_wr;
            wr_ptr.write_bytes(0, 1);
        }
        let sge = &mut comm.sges[r];

        let request = &comm.verbs.requests[requests_idx[r]];
        unsafe { 
            sge.addr = request.send_recv.send.data.addr() as u64;
            sge.lkey = request.send_recv.send.lkey;
        }

        let slot_non_null = unsafe { NonNull::new_unchecked(slots_ptr.add(r)) };
        let slot_ptr = unsafe { VolatilePtr::new(slot_non_null) } ;
        wr.opcode = ibv_wr_opcode::IBV_WR_RDMA_WRITE;
        wr.send_flags = 0;
        wr.wr.rdma.remote_addr = map_field!(slot_ptr.addr).read();
        wr.wr.rdma.rkey = map_field!(slot_ptr.rkey).read();
        wr.next = unsafe { (wr as *mut ibv_send_wr).add(1) };
        // 8: there are IB_MAX_REQUESTS (256) request slots
        // can be represented by 8 bytes
        wr_id += (requests_idx[r] as u64) << (r * 8);
    }

    let mut imm_data = 0;
    if num_requests == 1{
        imm_data = unsafe { comm.verbs.requests[requests_idx[0]].send_recv.send.size as u32 };
    } else {
        assert!(num_requests <= 32, "Cannot store sizes of {} requests in a 32-bits field", num_requests);
        for r in 0..num_requests {
            let size = unsafe { comm.verbs.requests[requests_idx[r]].send_recv.send.size };
            imm_data |= (if size > 0 { 1 } else { 0 }) << r;
        }
    }

    let mut last_wr = &mut comm.wrs[num_requests - 1];

    let first_size = unsafe { comm.verbs.requests[requests_idx[0]].send_recv.send.size };
    if num_requests > 1 || (comm.adaptive_routing && first_size > IB_AR_THRESHOLD) {
        last_wr = &mut comm.wrs[num_requests];
        unsafe { 
            let wr_ptr = last_wr as *mut ibv_send_wr;
            wr_ptr.write_bytes(0, 1);
        }
    }
    last_wr.wr_id = wr_id;
    last_wr.opcode = ibv_wr_opcode::IBV_WR_RDMA_WRITE_WITH_IMM;
    last_wr.__bindgen_anon_1.imm_data = imm_data;
    last_wr.next = std::ptr::null_mut();
    last_wr.send_flags = ibv_send_flags::IBV_SEND_SIGNALED.0;

    const ALIGN: usize = 128;
    let num_qps = comm.qps.len();
    for qp in comm.qps.iter_mut() {
        for r in 0..num_requests {
            let send_size = unsafe { comm.verbs.requests[requests_idx[r]].send_recv.send.size };
            let chunk_size = send_size.div_ceil(num_qps).div_ceil(ALIGN) * ALIGN;
            let offset = unsafe { comm.verbs.requests[requests_idx[r]].send_recv.send.offset };
            if send_size >= offset {
                comm.wrs[r].sg_list = std::ptr::null_mut();
                comm.wrs[r].num_sge = 0;
            } else {
                let length = std::cmp::min(send_size - offset, chunk_size);
                comm.sges[r].length = length as u32;
                comm.wrs[r].sg_list = &mut comm.sges[r];
                comm.wrs[r].num_sge = 1;
            };
        }
        unsafe { 
            let qp_ptr = qp.get_qp();
            let qp_ref = &mut *qp_ptr;
            let ctx = &mut *qp_ref.context;
            let send_fn = ctx.ops.post_send.unwrap();

            let mut bad_wr = std::ptr::null_mut();
            ibv_check!(send_fn(qp_ptr, comm.wrs.as_mut_ptr(), &mut bad_wr));
        }

        for r in 0..num_requests {
            let send_size = unsafe { comm.verbs.requests[requests_idx[r]].send_recv.send.size };
            let chunk_size = send_size.div_ceil(num_qps).div_ceil(ALIGN) * ALIGN;
            unsafe { 
                comm.verbs.requests[requests_idx[r]].send_recv.send.offset += chunk_size;
                comm.sges[r].addr += chunk_size as u64;
                comm.wrs[r].wr.rdma.remote_addr += chunk_size as u64;
            }
        }
    }
    Ok(())
}

pub fn ib_initiate_send(
    comm: Pin<&mut IbSendComm<'_>>,
    data: *mut c_void,
    size: usize,
    tag: u32,
    mr: &IbMrHandle,
) -> Result<Option<IbRequestId>, IbError> {
    let comm = unsafe { comm.get_unchecked_mut() };
    let mr = unsafe { &*mr.0.get_mr() };

    let slot = (comm.fifo_head % IB_MAX_REQUESTS as u64) as usize;
    let offset = slot * IB_MAX_RECVS;
    let slots_ptr = unsafe { comm.fifo.as_mut_ptr().add(offset) };
    let slot_non_null = unsafe { NonNull::new_unchecked(slots_ptr) };
    let slot_ptr = unsafe { VolatilePtr::new(slot_non_null) };

    let idx_expected = comm.fifo_head + 1;
    let idx_slot = map_field!(slot_ptr.idx).read();
    if idx_expected != idx_slot {
        // Wait for the receiver to post corresponding receive
        return Ok(None);
    }
    let num_requests = map_field!(slot_ptr.num_requests).read() as usize;
    for r in 1..num_requests {
        let slot_non_null = unsafe { NonNull::new_unchecked(slots_ptr.add(r)) };
        let slot_ptr = unsafe { VolatilePtr::new(slot_non_null) };
        let idx_slot = map_field!(slot_ptr.idx);
        while idx_slot.read() != idx_expected {}
    }
    // order the nreqsPtr load against tag/rkey/addr loads below
    // __sync_synchronize();
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    for r in 0..num_requests {
        let slot_non_null = unsafe { NonNull::new_unchecked(slots_ptr.add(r)) };
        let slot_ptr = unsafe { VolatilePtr::new(slot_non_null) };
        // IB_MAX_REQUESTS stands for NULL
        let tag_slot = map_field!(slot_ptr.tag).read();
        if comm.fifo_requests_idx[slot][r] < IB_MAX_REQUESTS || tag_slot != tag {
            continue;
        }
        
        let recv_size = map_field!(slot_ptr.size).read() as usize;
        if size > recv_size {
            Err(IbError::SendOverflow { send: size, recv: recv_size })?;
        }
        let remote_addr = map_field!(slot_ptr.addr).read();
        let rkey = map_field!(slot_ptr.rkey).read();
        if remote_addr == 0 || rkey == 0 {
            panic!("Peer posted incorrect receive info");
        }
        let request_id = ib_get_request(&mut comm.verbs)
            .ok_or_else(|| IbError::RequestBufferFull)?;
        let request = &mut comm.verbs.requests[request_id];
        request.ty = RequestType::Send;
        request.num_requests = num_requests as u32;
        request.send_recv.send.size = size;
        request.send_recv.send.data = data;
        request.send_recv.send.lkey = mr.lkey;
        request.send_recv.send.offset = 0;
        request.events = comm.qps.len() as u32;
        comm.fifo_requests_idx[slot][r] = request_id;

        let handle = IbRequestId(request_id);
        if comm.fifo_requests_idx[slot].iter().any(|id| *id == IB_MAX_REQUESTS) {
            return Ok(Some(handle));
        }

        ib_multi_send(unsafe { Pin::new_unchecked(comm) }, slot)?;
        
        let slot_non_null = unsafe { NonNull::new_unchecked(slots_ptr) };
        let slot_ptr = unsafe { VolatilePtr::new(slot_non_null) };
        // clear slots[0]'s num_requests and idx field for sanity checks
        slot_ptr.write(IbSendFifo::default());
        let requests_idx = &mut comm.fifo_requests_idx[slot];
        requests_idx.fill(IB_MAX_REQUESTS);
        comm.fifo_head += 1;
        return Ok(Some(handle));
    }
    return Ok(None);
}

pub fn ib_post_fifo(
    comm: Pin<&mut IbRecvComm<'_>>,
    data: &[*mut c_void],
    sizes: &[usize],
    tags: &[u32], 
    mr_handles: &[&IbMrHandle],
) -> Result<(), IbError> {
    let comm = unsafe { comm.get_unchecked_mut() };
    let mut wr = ibv_send_wr::default();
    let slot = (comm.remote_fifo.fifo_tail % IB_MAX_REQUESTS as u64) as usize;
    let local_elems = &mut comm.remote_fifo.mr[slot*IB_MAX_RECVS..(slot+1)*IB_MAX_RECVS];
    
    let num_requests = data.len();
    for i in 0..num_requests {
        local_elems[i].addr = data[i] as u64;
        let mr = unsafe { &*mr_handles[i].0.get_mr() };
        local_elems[i].rkey = mr.rkey;
        local_elems[i].num_requests = num_requests as u32;
        local_elems[i].size = sizes[i] as u32;
        local_elems[i].tag = tags[i];
        local_elems[i].idx = comm.remote_fifo.fifo_tail + 1;
    }
    wr.wr.rdma.remote_addr = comm.remote_fifo.addr + (slot * IB_MAX_RECVS * std::mem::size_of::<IbSendFifo>()) as u64;

    Ok(())
}