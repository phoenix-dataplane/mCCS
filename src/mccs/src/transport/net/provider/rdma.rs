use std::ffi::c_void;
use std::io::ErrorKind;
use std::marker::PhantomPinned;
use std::net::{IpAddr, SocketAddr};
use std::os::fd::RawFd;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, Weak};

use async_trait::async_trait;
use byteorder::{ByteOrder, LittleEndian};
use nix::unistd::{access, AccessFlags};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use smol::io::{AsyncReadExt, AsyncWriteExt};
use smol::net::TcpListener;
use thiserror::Error;
use volatile::{map_field, VolatilePtr};

use ibverbs::ffi::ibv_async_event;
use ibverbs::ffi::mlx5dv_modify_qp_udp_sport;
use ibverbs::ffi::{ibv_access_flags, ibv_qp_state, ibv_qp_type, ibv_send_flags, ibv_wr_opcode};
use ibverbs::ffi::{
    ibv_ack_async_event, ibv_event_type_str, ibv_get_async_event, IBV_EVENT_COMM_EST,
};
use ibverbs::ffi::{ibv_create_qp, ibv_modify_qp};
use ibverbs::ffi::{
    ibv_device_attr, ibv_port_attr, ibv_qp_attr, ibv_qp_attr_mask, ibv_qp_init_attr,
};
use ibverbs::ffi::{ibv_recv_wr, ibv_send_wr, ibv_sge, ibv_wc};
use ibverbs::ffi::{ibv_wc_opcode, ibv_wc_status};
use ibverbs::Context;
use ibverbs::{
    CompletionQueue, MemoryRegionAlloc, MemoryRegionRegister, ProtectionDomain, QueuePair,
};

use super::NET_MAX_REQUESTS;
use super::{AnyMrHandle, AnyNetComm, CommType, NetProvider, NetRequestId};
use super::{MemoryRegion, NetListener, NetProperties};
use crate::transport::catalog::TransportCatalog;
use crate::utils::interfaces;
use crate::utils::tcp;

const IB_MAX_RECVS: usize = 8;
const IB_MAX_QPS: usize = 16;
const IB_MAX_REQUESTS: usize = NET_MAX_REQUESTS * IB_MAX_RECVS;

// request id are encoded in wr_id and we need up to 8 requests ids per completion
static_assertions::const_assert!(IB_MAX_REQUESTS <= 256);

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
    #[error("RDMA transport config not found")]
    ConfigNotFound,
    #[error("RDMA transport context is not initialized")]
    ContextUninitialized,
    #[error("RDMA transport context  initialized")]
    ContextAlreadyInitialized,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Insufficient recv size {recv} to match send size {send}")]
    SendOverflow { send: usize, recv: usize },
    #[error(
        "Maximum number of outstanding requests of {} reached",
        IB_MAX_REQUESTS
    )]
    RequestBufferFull,
    #[error("RDMA transport context is not initialized")]
    ContextUninitalized,
    #[error("Failed to allocate protection domain")]
    AllocPd,
    #[error("Failed to poll completion queue")]
    PollCq,
    #[error("Work completion error, opcode={0}, byte_len={1}, status={2}, vendor_err={3}")]
    WcError(ibv_wc_opcode::Type, usize, ibv_wc_status::Type, u32),
    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("Number of MR handles, data sizes or tags mismatch number of data elements ({0})")]
    NumElemsMismatch(usize),
    #[error("Number of receives ({0}) exceeeds maximum {}", IB_MAX_RECVS)]
    ExceedMaxRecv(usize),
    #[error("Failed to downcast MR handle")]
    DowncastMrHandle,
    #[error("Failed to downcast send/recv communicator")]
    DowncastNetComm,
    #[error("Network interface error: {0}")]
    Interface(#[from] interfaces::NetInterfaceError),
    #[error("No IP interface found")]
    NoIpInterface,
}

const IBV_WIDTHS: [u32; 5] = [1, 4, 8, 12, 2];
const IBV_SPEEDS: [u32; 8] = [
    2500,   // SDR
    5000,   // DDR
    10000,  // QDR
    10000,  // FDR10
    14000,  // FDR
    25000,  // EDR
    50000,  // HDR
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
    pd: Weak<ProtectionDomain<'static>>,
    mr_cache: Vec<Weak<MemoryRegionRegister>>,
}

pub struct IbDevice {
    #[allow(unused)]
    device: usize,
    guid: u64,
    port: u8,
    #[allow(unused)]
    link: u8,
    speed: u32,
    context: Arc<Context>,
    device_name: String,
    pci_path: String,
    real_port: u8,
    max_qp: u32,
    adaptive_routing: bool,
    resources: Mutex<IbDeviceResources>,
    dma_buf_support: bool,
    _pinned: PhantomPinned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdmaTransportConfig {
    pub gid_index: u8,
    pub qps_per_conn: usize,
    pub timeout: u8,
    pub retry_count: u8,
    pub pkey: u16,
    pub use_inline: bool,
    pub service_level: u8,
    pub traffic_class: u8,
    pub adaptive_routing: Option<bool>,
    pub ar_threshold: usize,
    pub pci_relaxed_ordering: bool,
    pub gdr_flush_disable: bool,
    pub socket_if_prefix: Option<String>,
    pub ib_if_prefix: Option<String>,
}

impl Default for RdmaTransportConfig {
    fn default() -> Self {
        RdmaTransportConfig {
            gid_index: 3,
            qps_per_conn: 1,
            timeout: 18,
            retry_count: 7,
            pkey: 0,
            use_inline: false,
            service_level: 0,
            traffic_class: 0,
            adaptive_routing: None,
            ar_threshold: 8192,
            // TODO: set to true
            pci_relaxed_ordering: false,
            // TODO: set to false
            gdr_flush_disable: true,
            socket_if_prefix: Some("rdma0".to_string()),
            ib_if_prefix: None,
        }
    }
}

pub struct RdmaTransportContext {
    devices: Vec<IbDevice>,
    listen_addr: IpAddr,
    page_size: usize,
    gdr_support: bool,
    config: RdmaTransportConfig,
}

pub struct RdmaTransportProvider(pub OnceCell<RdmaTransportContext>);

pub static RDMA_TRANSPORT: RdmaTransportProvider = RdmaTransportProvider(OnceCell::new());

fn get_pci_path(
    device_name: &str,
    current_devices: &Vec<IbDevice>,
) -> Result<(String, u8), IbError> {
    let device_path = format!("/sys/class/infiniband/{}/device", device_name);
    //  char* p = realpath(devicePath, NULL);
    let real_path = std::fs::canonicalize(device_path)?;
    let mut p = real_path.to_str().unwrap().to_string();
    let len = p.len();
    // Merge multi-port NICs into the same PCI device
    p.replace_range(len - 1..len, "0");
    // Also merge virtual functions (VF) into the same device
    p.replace_range(len - 3..len - 2, "0");
    let mut real_port = 0;
    for device in current_devices.iter() {
        if device.pci_path == p {
            real_port += 1;
        }
    }
    Ok((p, real_port))
}

fn ib_gdr_support() -> bool {
    const PATH_NV_PEER_MEM: &'static str = "/sys/kernel/mm/memory_peers/nv_mem/version";
    const PATH_NVIDIA_PEERMEM: &'static str = "/sys/kernel/mm/memory_peers/nvidia-peermem/version";
    if access(PATH_NV_PEER_MEM, AccessFlags::F_OK).is_ok()
        || access(PATH_NVIDIA_PEERMEM, AccessFlags::F_OK).is_ok()
    {
        true
    } else {
        false
    }
}

fn ib_async_therad(context: Arc<Context>) {
    let ctx = context.ctx;
    loop {
        unsafe {
            let mut event = ibv_async_event::default();
            let ret = ibv_get_async_event(ctx, &mut event);
            if ret == -1 {
                log::warn!("ibv_get_async_event: failed to get async event");
            }
            let type_str = ibv_event_type_str(event.event_type);
            let c_str = std::ffi::CStr::from_ptr(type_str);
            if event.event_type != IBV_EVENT_COMM_EST {
                log::warn!(
                    "RDMA transport: Got async event {}",
                    c_str.to_str().unwrap()
                );
            }
            ibv_ack_async_event(&mut event);
        }
    }
}

fn ib_dma_buf_support(context: &Context) -> Result<bool, IbError> {
    let pd = context.alloc_pd().map_err(|_| IbError::AllocPd)?;
    unsafe {
        let pd_ptr = pd.get_pd();
        let _ = ibverbs::ffi::ibv_reg_dmabuf_mr(pd_ptr, 0u64, 0usize, 0u64, -1, 0);
        let errno = nix::errno::Errno::last();
        if errno != nix::errno::Errno::EOPNOTSUPP || errno != nix::errno::Errno::EPROTONOSUPPORT {
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

pub fn ib_init_transport_context(
    config: RdmaTransportConfig,
) -> Result<RdmaTransportContext, IbError> {
    let socket_if_prefix = config.socket_if_prefix.as_ref().map(|x| x.as_str());
    let mut interface = interfaces::find_interfaces(socket_if_prefix, None, 1)?;
    if interface.is_empty() {
        Err(IbError::NoIpInterface)?;
    }
    let (interface_name, interface_addr) = interface.pop().unwrap();
    log::info!(
        "RDMA transport: using interface {}: {:?} for socket bootstrap",
        interface_name,
        interface_addr
    );
    let devices = ibverbs::devices()?;
    let (ib_if_specs, search_exact, search_not) =
        if let Some(if_prefix) = config.ib_if_prefix.as_ref() {
            let mut prefix_list = if_prefix.as_str();
            let search_not = prefix_list.chars().nth(0) == Some('^');
            if search_not {
                prefix_list = &prefix_list[1..];
            }
            let search_exact = prefix_list.chars().nth(0) == Some('=');
            if search_exact {
                prefix_list = &prefix_list[1..];
            }
            let specs = interfaces::parse_prefix_list(prefix_list)?;
            (Some(specs), search_exact, search_not)
        } else {
            (None, false, false)
        };
    let mut devices_ctx = Vec::with_capacity(devices.len());
    let mut dev_enabled = false;
    for (idx, dev) in devices.iter().enumerate() {
        let context = Arc::new(match dev.open() {
            Ok(v) => v,
            Err(e) => {
                if e.kind() == ErrorKind::Other {
                    continue;
                } else {
                    Err(e)?
                }
            }
        });
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
                && port_attr.link_layer != ibverbs::ffi::IBV_LINK_LAYER_INFINIBAND as u8
            {
                continue;
            }

            let device_name = if let Some(name) = dev.name() {
                name.to_str().unwrap().to_string()
            } else {
                String::new()
            };
            if let Some(if_specs) = ib_if_specs.as_ref() {
                let hit = interfaces::match_interface_list(
                    device_name.as_str(),
                    Some(port as u16),
                    if_specs,
                    search_exact,
                ) ^ search_not;
                if !hit {
                    continue;
                }
            }

            log::info!(
                "Initialize RDMA device [{idx}] {device_name}:{port}, {}",
                if port_attr.link_layer == ibverbs::ffi::IBV_LINK_LAYER_INFINIBAND as u8 {
                    "IB"
                } else {
                    "RoCE"
                }
            );

            let speed = get_ib_speed(port_attr.active_speed) * get_ib_width(port_attr.active_width);
            let (pci_path, real_port) = get_pci_path(&device_name, &devices_ctx)?;
            let mut ar = port_attr.link_layer == ibverbs::ffi::IBV_LINK_LAYER_INFINIBAND as u8;
            if let Some(ar_override) = config.adaptive_routing {
                ar = ar_override;
            }
            let dma_buf_support = ib_dma_buf_support(&context)?;
            let resources = IbDeviceResources {
                pd: Weak::new(),
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
                dma_buf_support,
                _pinned: PhantomPinned,
            };
            dev_enabled = true;
            devices_ctx.push(dev_context);
        }
        if dev_enabled {
            let join_handle = std::thread::spawn(move || {
                ib_async_therad(context);
            });
            std::mem::drop(join_handle);
        }
    }
    use nix::unistd::{sysconf, SysconfVar};
    let page_size = sysconf(SysconfVar::PAGE_SIZE)
        .map_err(|e| std::io::Error::from_raw_os_error(e as i32))?
        .unwrap() as usize;
    let gdr_support = ib_gdr_support();
    let interface_addr = interface_addr.as_socket().unwrap().ip();
    let transport_context = RdmaTransportContext {
        devices: devices_ctx,
        listen_addr: interface_addr,
        page_size,
        gdr_support,
        config,
    };
    Ok(transport_context)
}

pub fn ib_get_num_devices() -> Result<usize, IbError> {
    let transport_ctx = RDMA_TRANSPORT.0.get().ok_or(IbError::ContextUninitalized)?;
    let num_devices = transport_ctx.devices.len();
    Ok(num_devices)
}

pub fn ib_get_properties(device: usize) -> Result<NetProperties, IbError> {
    let transport_ctx = RDMA_TRANSPORT.0.get().ok_or(IbError::ContextUninitalized)?;
    let device_ctx = &transport_ctx.devices[device];

    let mut ptr_support = super::PtrSupport::PTR_HOST;
    if transport_ctx.gdr_support {
        ptr_support |= super::PtrSupport::PTR_CUDA;
    }
    if device_ctx.dma_buf_support {
        ptr_support |= super::PtrSupport::PTR_DMA_BUF;
    }

    let props = NetProperties {
        name: device_ctx.device_name.clone(),
        pci_path: device_ctx.pci_path.clone(),
        guid: device_ctx.guid,
        ptr_support,
        speed: device_ctx.speed,
        port: device_ctx.port as u16 + device_ctx.real_port as u16,
        latency: 0.0,
        max_comms: device_ctx.max_qp as usize,
        max_recvs: IB_MAX_RECVS,
    };
    Ok(props)
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
    sizes: [usize; IB_MAX_RECVS],
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

unsafe impl Send for IbRequest {}

#[derive(Clone, Debug)]
#[repr(C)]
struct IbQpInfo {
    lid: u32,
    ib_port: u8,
    link_layer: u8,
    qpn: [u32; IB_MAX_QPS],

    spn: u64,
    iid: u64,
    mtu: u32,

    fifo_rkey: u32,
    fifo_addr: u64,
}

const IB_QP_INFO_SEND_SIZE: usize = 6 + IB_MAX_QPS * 4 + 8 + 8 + 4 + 4 + 8;

impl IbQpInfo {
    fn write(&self, buf: &mut [u8; IB_QP_INFO_SEND_SIZE]) {
        LittleEndian::write_u32(&mut buf[0..4], self.lid);
        buf[4] = self.ib_port;
        buf[5] = self.link_layer;
        for q in 0..IB_MAX_QPS {
            let offset = 6 + q * 4;
            LittleEndian::write_u32(&mut buf[offset..offset + 4], self.qpn[q]);
        }
        let offset = 6 + IB_MAX_QPS * 4;
        LittleEndian::write_u64(&mut buf[offset..offset + 8], self.spn);
        LittleEndian::write_u64(&mut buf[offset + 8..offset + 16], self.iid);
        LittleEndian::write_u32(&mut buf[offset + 16..offset + 20], self.mtu);
        LittleEndian::write_u32(&mut buf[offset + 20..offset + 24], self.fifo_rkey);
        LittleEndian::write_u64(&mut buf[offset + 24..offset + 32], self.fifo_addr);
    }

    fn read(buf: &[u8; IB_QP_INFO_SEND_SIZE]) -> IbQpInfo {
        let lid = LittleEndian::read_u32(&buf[0..4]);
        let ib_port = buf[4];
        let link_layer = buf[5];
        let mut qpn = [0; IB_MAX_QPS];
        for q in 0..IB_MAX_QPS {
            let offset = 6 + q * 4;
            qpn[q] = LittleEndian::read_u32(&buf[offset..offset + 4]);
        }
        let offset = 6 + IB_MAX_QPS * 4;
        let spn = LittleEndian::read_u64(&buf[offset..offset + 8]);
        let iid = LittleEndian::read_u64(&buf[offset + 8..offset + 16]);
        let mtu = LittleEndian::read_u32(&buf[offset + 16..offset + 20]);
        let fifo_rkey = LittleEndian::read_u32(&buf[offset + 20..offset + 24]);
        let fifo_addr = LittleEndian::read_u64(&buf[offset + 24..offset + 32]);
        IbQpInfo {
            lid,
            ib_port,
            link_layer,
            qpn,
            spn,
            iid,
            mtu,
            fifo_rkey,
            fifo_addr,
        }
    }
}

pub struct IbVerbs<'ctx> {
    device: usize,
    cq: CompletionQueue<'ctx>,
    pd: Arc<ProtectionDomain<'ctx>>,
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
    qps: Vec<QueuePair<'ctx>>,
    fifo: MemoryRegionAlloc<IbSendFifo>,
    verbs: IbVerbs<'ctx>,
    fifo_head: u64,
    fifo_requests_idx: [[usize; IB_MAX_RECVS]; IB_MAX_REQUESTS],
    wrs: [ibv_send_wr; IB_MAX_RECVS + 1],
    sges: [ibv_sge; IB_MAX_RECVS],
    adaptive_routing: bool,
    ar_threshold: usize,
    _pin: PhantomPinned,
}

unsafe impl<'ctx> Send for IbSendComm<'ctx> {}

// IbSendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
// By default, Vec<T> guarantees that memory is properly aligned for type T
static_assertions::const_assert_eq!(std::mem::size_of::<IbSendFifo>() % 32, 0);

pub struct IbRemoteFifo {
    mr: MemoryRegionAlloc<IbSendFifo>,
    fifo_tail: u64,
    addr: u64,
    rkey: u32,
    flags: ibv_send_flags,
    sge: ibv_sge,
}

pub struct IbGpuFlush<'ctx> {
    enabled: bool,
    #[allow(unused)]
    host_mr: Option<MemoryRegionAlloc<i32>>,
    sge: ibv_sge,
    qp: Option<QueuePair<'ctx>>,
}

pub struct IbRecvComm<'ctx> {
    qps: Vec<QueuePair<'ctx>>,
    remote_fifo: IbRemoteFifo,
    flush: IbGpuFlush<'ctx>,
    verbs: IbVerbs<'ctx>,
    _pinned: PhantomPinned,
}

unsafe impl<'ctx> Send for IbRecvComm<'ctx> {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IbConnectHandle {
    connect_addr: std::net::SocketAddr,
    magic: u64,
}

pub struct IbListenComm {
    device: usize,
    listener: TcpListener,
    magic: u64,
}

fn ib_init_verbs(device: usize) -> Result<IbVerbs<'static>, IbError> {
    let transport_ctx = RDMA_TRANSPORT.0.get().ok_or(IbError::ContextUninitalized)?;
    let device_ctx = &transport_ctx.devices[device];

    let mut guard = device_ctx.resources.lock().unwrap();

    // 2*MAX_REQUESTS*ncclParamIbQpsPerConn()
    let num_cqe = 2 * IB_MAX_REQUESTS * transport_ctx.config.qps_per_conn;
    let cq = device_ctx.context.create_cq(num_cqe as i32, 0)?;
    let requests = std::array::from_fn(|_| IbRequest {
        events: 0,
        num_requests: 0,
        ty: RequestType::Unused,
        send_recv: SendRecvRequest {
            send: SendRequest {
                size: 0,
                data: std::ptr::null_mut(),
                lkey: 0,
                offset: 0,
            },
        },
    });
    let pd = if let Some(pd) = guard.pd.upgrade() {
        pd
    } else {
        let ctx = &*device_ctx.context;
        let pd = ctx.alloc_pd().map_err(|_| IbError::AllocPd)?;
        let pd = Arc::new(pd);
        guard.pd = Arc::downgrade(&pd);
        pd
    };
    let verbs = IbVerbs {
        device,
        pd,
        cq,
        requests,
    };
    Ok(verbs)
}

pub fn create_qp(
    port: u8,
    verbs: &mut IbVerbs<'_>,
    access_flags: ibv_access_flags,
) -> Result<QueuePair<'static>, IbError> {
    let transport_ctx = RDMA_TRANSPORT.0.get().ok_or(IbError::ContextUninitalized)?;

    let mut qp_init_attr = ibv_qp_init_attr::default();
    let cq = verbs.cq.get_cq();
    qp_init_attr.send_cq = cq;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.qp_type = ibv_qp_type::IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = 2 * IB_MAX_REQUESTS as u32;
    qp_init_attr.cap.max_recv_wr = IB_MAX_REQUESTS as u32;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    // ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;
    qp_init_attr.cap.max_inline_data = if transport_ctx.config.use_inline {
        std::mem::size_of::<IbSendFifo>() as u32
    } else {
        0
    };
    let pd = verbs.pd.get_pd();
    let qp = unsafe {
        let qp = ibv_create_qp(pd, &mut qp_init_attr);
        if qp.is_null() {
            Err(std::io::Error::last_os_error())?;
        }
        qp
    };
    let mut qp_attr = ibv_qp_attr::default();
    qp_attr.qp_state = ibv_qp_state::IBV_QPS_INIT;
    qp_attr.pkey_index = transport_ctx.config.pkey;
    qp_attr.port_num = port;
    qp_attr.qp_access_flags = access_flags.0;
    unsafe {
        let attr_mask = ibv_qp_attr_mask::IBV_QP_STATE
            | ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
            | ibv_qp_attr_mask::IBV_QP_PORT
            | ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;
        ibv_check!(ibv_modify_qp(qp, &mut qp_attr, attr_mask.0 as i32));
    }
    let qp = QueuePair::new(qp);
    Ok(qp)
}

fn ib_rtr_qp(
    qp: *mut ibverbs::ffi::ibv_qp,
    remote_qpn: u32,
    remote_qp_info: &IbQpInfo,
    traffic_class: Option<u8>,
) -> Result<(), IbError> {
    let transport_ctx = RDMA_TRANSPORT.0.get().ok_or(IbError::ContextUninitalized)?;

    let mut qp_attr = ibv_qp_attr::default();
    qp_attr.qp_state = ibv_qp_state::IBV_QPS_RTR;
    qp_attr.path_mtu = remote_qp_info.mtu;
    qp_attr.dest_qp_num = remote_qpn;
    qp_attr.rq_psn = 0;
    qp_attr.max_dest_rd_atomic = 1;
    qp_attr.min_rnr_timer = 12;
    if remote_qp_info.link_layer == ibverbs::ffi::IBV_LINK_LAYER_ETHERNET as u8 {
        qp_attr.ah_attr.is_global = 1;
        qp_attr.ah_attr.grh.dgid.global.subnet_prefix = remote_qp_info.spn;
        qp_attr.ah_attr.grh.dgid.global.interface_id = remote_qp_info.iid;
        qp_attr.ah_attr.grh.flow_label = 0;
        qp_attr.ah_attr.grh.sgid_index = transport_ctx.config.gid_index;
        qp_attr.ah_attr.grh.hop_limit = 255;
        if let Some(tc) = traffic_class {
            qp_attr.ah_attr.grh.traffic_class = tc;
        } else {
            qp_attr.ah_attr.grh.traffic_class = transport_ctx.config.traffic_class;
        }
    } else {
        qp_attr.ah_attr.is_global = 0;
        qp_attr.ah_attr.dlid = remote_qp_info.lid as u16;
    }
    qp_attr.ah_attr.sl = transport_ctx.config.service_level;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = remote_qp_info.ib_port;
    unsafe {
        let attr_mask = ibv_qp_attr_mask::IBV_QP_STATE
            | ibv_qp_attr_mask::IBV_QP_AV
            | ibv_qp_attr_mask::IBV_QP_PATH_MTU
            | ibv_qp_attr_mask::IBV_QP_DEST_QPN
            | ibv_qp_attr_mask::IBV_QP_RQ_PSN
            | ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC
            | ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;
        ibv_check!(ibv_modify_qp(qp, &mut qp_attr, attr_mask.0 as i32));
    }
    Ok(())
}

fn ib_rts_qp(qp: *mut ibverbs::ffi::ibv_qp, udp_sport: Option<u16>) -> Result<(), IbError> {
    let transport_ctx = RDMA_TRANSPORT.0.get().ok_or(IbError::ContextUninitalized)?;

    let mut qp_attr = ibv_qp_attr::default();
    qp_attr.qp_state = ibv_qp_state::IBV_QPS_RTS;
    qp_attr.timeout = transport_ctx.config.timeout;

    qp_attr.retry_cnt = transport_ctx.config.retry_count;
    qp_attr.rnr_retry = 7;
    qp_attr.sq_psn = 0;
    qp_attr.max_rd_atomic = 1;
    unsafe {
        let attr_mask = ibv_qp_attr_mask::IBV_QP_STATE
            | ibv_qp_attr_mask::IBV_QP_TIMEOUT
            | ibv_qp_attr_mask::IBV_QP_RETRY_CNT
            | ibv_qp_attr_mask::IBV_QP_RNR_RETRY
            | ibv_qp_attr_mask::IBV_QP_SQ_PSN
            | ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC;
        ibv_check!(ibv_modify_qp(qp, &mut qp_attr, attr_mask.0 as i32));
        if let Some(udp_sport) = udp_sport {
            let ret = mlx5dv_modify_qp_udp_sport(qp, udp_sport);
            if ret != 0 {
                log::warn!("Failed to set UDP source port {}: {}", udp_sport, ret);
            }
        }
    }
    Ok(())
}

pub async fn ib_listen(device: usize) -> Result<(IbConnectHandle, IbListenComm), IbError> {
    let transport_ctx = RDMA_TRANSPORT.0.get().ok_or(IbError::ContextUninitalized)?;

    let listen_addr = SocketAddr::new(transport_ctx.listen_addr, 0);
    let listener = tcp::async_listen(&listen_addr)?;
    let listen_addr = listener.local_addr()?;
    let magic = rand::random::<u64>();
    let handle = IbConnectHandle {
        connect_addr: listen_addr,
        magic,
    };
    let listen_comm = IbListenComm {
        device,
        listener,
        magic,
    };
    Ok((handle, listen_comm))
}

pub async fn ib_connect(
    device: usize,
    handle: &IbConnectHandle,
    udp_sport: Option<u16>,
    tc: Option<u8>,
) -> Result<IbSendComm<'static>, IbError> {
    // Stage 1: connect to peer and set up QPs
    let connect_addr = handle.connect_addr;
    let mut stream = tcp::async_connect(&connect_addr, handle.magic).await?;

    log::debug!("RDMA transport provider connects to {:?}", connect_addr);
    let mut verbs = ib_init_verbs(device)?;
    let transport_ctx = RDMA_TRANSPORT.0.get().ok_or(IbError::ContextUninitalized)?;
    let device_ctx = &transport_ctx.devices[device];
    let ib_port = device_ctx.port;
    let mut qps = Vec::with_capacity(transport_ctx.config.qps_per_conn);
    for _ in 0..transport_ctx.config.qps_per_conn {
        let qp = create_qp(
            ib_port,
            &mut verbs,
            ibv_access_flags::IBV_ACCESS_REMOTE_WRITE,
        )?;
        qps.push(qp);
    }
    let ar = device_ctx.adaptive_routing;
    let mut port_attr = ibv_port_attr::default();
    unsafe {
        let ptr = &mut port_attr as *mut ibv_port_attr as *mut _;
        ibv_check!(ibverbs::ffi::ibv_query_port(
            device_ctx.context.ctx,
            ib_port,
            ptr
        ));
    }
    let mut qp_info = IbQpInfo {
        lid: port_attr.lid as u32,
        ib_port,
        link_layer: port_attr.link_layer,
        qpn: [0; IB_MAX_QPS],
        spn: 0,
        iid: 0,
        mtu: port_attr.active_mtu,
        fifo_rkey: 0,
        fifo_addr: 0,
    };
    for (idx, qp) in qps.iter().enumerate() {
        let qp = unsafe { &*qp.get_qp() };
        qp_info.qpn[idx] = qp.qp_num;
    }

    let access_flags = ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
        | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
        | ibv_access_flags::IBV_ACCESS_REMOTE_READ;
    let fifo = verbs
        .pd
        .allocate::<IbSendFifo>(IB_MAX_REQUESTS * IB_MAX_RECVS, access_flags)?;
    qp_info.fifo_rkey = fifo.rkey().0;
    qp_info.fifo_addr = fifo.addr() as u64;

    if qp_info.link_layer == ibverbs::ffi::IBV_LINK_LAYER_ETHERNET as u8 {
        let mut gid = ibverbs::ffi::ibv_gid::default();
        let gid_index = transport_ctx.config.gid_index;
        unsafe {
            ibv_check!(ibverbs::ffi::ibv_query_gid(
                device_ctx.context.ctx,
                ib_port,
                gid_index as i32,
                &mut gid,
            ));
            qp_info.spn = gid.global.subnet_prefix;
            qp_info.iid = gid.global.interface_id;
        }
        log::info!(
            "RDMA transport connect: device={}, port={}, mtu={}, GID={:} ({:x}/{:x})",
            device,
            ib_port,
            qp_info.mtu,
            gid_index,
            qp_info.spn,
            qp_info.iid
        );
    } else {
        log::info!(
            "RDMA transport connect: device={}, port={}, mtu={}, LID={}",
            device,
            ib_port,
            qp_info.mtu,
            qp_info.lid
        );
    }
    // Stage 2: send QP info to peer
    let mut buf = [0; IB_QP_INFO_SEND_SIZE];
    qp_info.write(&mut buf);
    stream.write_all(buf.as_slice()).await?;

    // Stage 3: receive peer's QP info
    stream.read_exact(buf.as_mut_slice()).await?;
    let remote_qp_info = IbQpInfo::read(&buf);

    for (q, qp) in qps.iter().enumerate() {
        let qp = qp.get_qp();
        let remote_qpn = remote_qp_info.qpn[q];
        ib_rtr_qp(qp, remote_qpn, &remote_qp_info, tc)?;
        ib_rts_qp(qp, udp_sport)?;
    }

    // Stage 4: signal ready to peer
    let buf = [1u8; 1];
    // no need to preserve the TCP connection
    // it will be dropped when the function exits
    stream.write_all(&buf.as_slice()).await?;

    let fifo_requests_idx = [[IB_MAX_REQUESTS; IB_MAX_RECVS]; IB_MAX_REQUESTS];
    let wrs = [ibv_send_wr::default(); IB_MAX_RECVS + 1];
    let sges = [ibv_sge::default(); IB_MAX_RECVS];

    let send_comm = IbSendComm {
        verbs,
        fifo,
        fifo_head: 0,
        fifo_requests_idx,
        wrs,
        sges,
        qps,
        adaptive_routing: ar,
        ar_threshold: transport_ctx.config.ar_threshold,
        _pin: PhantomPinned,
    };
    Ok(send_comm)
}

pub async fn ib_accept(
    listen_comm: IbListenComm,
    tc: Option<u8>,
) -> Result<IbRecvComm<'static>, IbError> {
    let mut stream = tcp::async_accept(&listen_comm.listener, listen_comm.magic).await?;

    let mut buf = [0u8; IB_QP_INFO_SEND_SIZE];
    stream.read_exact(buf.as_mut_slice()).await?;
    let mut remote_qp_info = IbQpInfo::read(&buf);

    let transport_ctx = RDMA_TRANSPORT.0.get().ok_or(IbError::ContextUninitalized)?;
    let device_ctx = &transport_ctx.devices[listen_comm.device];
    let ib_port = device_ctx.port;
    let mut port_attr = ibv_port_attr::default();
    unsafe {
        let ptr = &mut port_attr as *mut ibv_port_attr as *mut _;
        ibv_check!(ibverbs::ffi::ibv_query_port(
            device_ctx.context.ctx,
            ib_port,
            ptr
        ));
    }
    let mut gid = ibverbs::ffi::ibv_gid::default();
    let gid_index = transport_ctx.config.gid_index;
    unsafe {
        ibv_check!(ibverbs::ffi::ibv_query_gid(
            device_ctx.context.ctx,
            ib_port,
            gid_index as i32,
            &mut gid,
        ));
    }

    let mut verbs = ib_init_verbs(listen_comm.device)?;
    let mut qps = Vec::with_capacity(transport_ctx.config.qps_per_conn);
    for _ in 0..transport_ctx.config.qps_per_conn {
        let qp = create_qp(
            ib_port,
            &mut verbs,
            ibv_access_flags::IBV_ACCESS_REMOTE_WRITE,
        )?;
        qps.push(qp);
    }

    remote_qp_info.mtu = std::cmp::min(remote_qp_info.mtu, port_attr.active_mtu);

    for (q, qp) in qps.iter().enumerate() {
        let qp = qp.get_qp();
        let remote_qpn = remote_qp_info.qpn[q];
        ib_rtr_qp(qp, remote_qpn, &remote_qp_info, tc)?;
        ib_rts_qp(qp, None)?;
    }

    let access_flags = ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
        | ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
        | ibv_access_flags::IBV_ACCESS_REMOTE_READ;
    let fifo_mr = verbs
        .pd
        .allocate::<IbSendFifo>(IB_MAX_REQUESTS * IB_MAX_RECVS, access_flags)?;
    let mut fifo_sge = ibv_sge::default();
    fifo_sge.lkey = unsafe { &*fifo_mr.get_mr() }.lkey;
    let send_flags = if transport_ctx.config.use_inline {
        ibv_send_flags::IBV_SEND_INLINE
    } else {
        ibv_send_flags(0)
    };
    let remote_fifo = IbRemoteFifo {
        mr: fifo_mr,
        fifo_tail: 0,
        addr: remote_qp_info.fifo_addr,
        rkey: remote_qp_info.fifo_rkey,
        flags: send_flags,
        sge: fifo_sge,
    };
    log::debug!(
        "ib_accept received remote qp fifo addr={:0x}, rkey={}",
        remote_fifo.addr,
        remote_fifo.rkey
    );
    // Allocate Flush dummy buffer for GPU Direct RDMA
    // TODO: check GDR support
    let gpu_flush_enable = transport_ctx.gdr_support && !transport_ctx.config.gdr_flush_disable;
    let gpu_flush = if gpu_flush_enable {
        let flush_mr = verbs
            .pd
            .allocate::<i32>(1, ibv_access_flags::IBV_ACCESS_LOCAL_WRITE)?;
        let mut sge = ibv_sge::default();
        sge.addr = flush_mr.addr() as u64;
        sge.length = 1;
        let mr = unsafe { &*flush_mr.get_mr() };
        sge.lkey = mr.lkey;
        let access_flags =
            ibv_access_flags::IBV_ACCESS_LOCAL_WRITE | ibv_access_flags::IBV_ACCESS_REMOTE_READ;
        let flush_qp = create_qp(ib_port, &mut verbs, access_flags)?;
        let local_qp_info = IbQpInfo {
            lid: port_attr.lid as u32,
            ib_port,
            link_layer: port_attr.link_layer,
            qpn: [0; IB_MAX_QPS],
            spn: unsafe { gid.global.subnet_prefix },
            iid: unsafe { gid.global.interface_id },
            mtu: port_attr.active_mtu,
            fifo_rkey: 0,
            fifo_addr: 0,
        };
        let qp_ptr = flush_qp.get_qp();
        let qpn = unsafe { &*qp_ptr }.qp_num;
        ib_rtr_qp(qp_ptr, qpn, &local_qp_info, None)?;
        ib_rts_qp(qp_ptr, None)?;
        let flush = IbGpuFlush {
            enabled: true,
            host_mr: Some(flush_mr),
            sge,
            qp: Some(flush_qp),
        };
        flush
    } else {
        let flush = IbGpuFlush {
            enabled: false,
            host_mr: None,
            sge: ibv_sge::default(),
            qp: None,
        };
        flush
    };

    let mut qp_info = IbQpInfo {
        lid: port_attr.lid as u32,
        ib_port,
        link_layer: port_attr.link_layer,
        qpn: [0; IB_MAX_QPS],
        spn: unsafe { gid.global.subnet_prefix },
        iid: unsafe { gid.global.interface_id },
        mtu: port_attr.active_mtu,
        fifo_rkey: 0,
        fifo_addr: 0,
    };
    for (idx, qp) in qps.iter().enumerate() {
        let qp = unsafe { &*qp.get_qp() };
        qp_info.qpn[idx] = qp.qp_num;
    }
    let mut buf = [0; IB_QP_INFO_SEND_SIZE];
    qp_info.write(&mut buf);
    stream.write_all(buf.as_slice()).await?;

    let mut buf = [0u8; 1];
    stream.read_exact(buf.as_mut_slice()).await?;

    let recv_comm = IbRecvComm {
        verbs,
        remote_fifo,
        qps,
        flush: gpu_flush,
        _pinned: PhantomPinned,
    };
    Ok(recv_comm)
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
    let transport_ctx = RDMA_TRANSPORT.0.get().ok_or(IbError::ContextUninitalized)?;
    let device_ctx = &transport_ctx.devices[verbs.device];

    let page_size = transport_ctx.page_size;
    let addr = data.addr() & (-(page_size as isize)) as usize;
    let pages = (data.addr() + size - addr + page_size - 1) / page_size;
    let size_aligned = pages * page_size;

    let mut guard = device_ctx.resources.lock().unwrap();
    let cached_mr = guard.mr_cache.iter().find_map(|mr| {
        let mr = mr.upgrade();
        if let Some(mr) = mr {
            if mr.addr() == addr && mr.size() == size_aligned {
                return Some(mr);
            }
        }
        None
    });

    // Clear unused MRs in MR cache
    guard.mr_cache.retain(|mr| mr.strong_count() > 0);

    if let Some(mr) = cached_mr {
        let handle = IbMrHandle(mr);
        Ok(handle)
    } else {
        let mut flags = ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
            | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
            | ibv_access_flags::IBV_ACCESS_REMOTE_READ;
        if transport_ctx.config.pci_relaxed_ordering {
            flags |= ibv_access_flags::IBV_ACCESS_RELAXED_ORDERING;
        }
        // TODO: ceheck for relaxed ordering
        let addr = addr as *mut c_void;
        let mr = if fd != -1 {
            verbs
                .pd
                .register_dmabuf_mr(addr, size_aligned, offset, fd, flags)?
        } else {
            if transport_ctx.config.pci_relaxed_ordering {
                verbs.pd.register_mr_iova2(addr, size_aligned, flags)?
            } else {
                verbs.pd.register_mr(addr, size_aligned, flags)?
            }
        };
        let mr = Arc::new(mr);
        let mr_weak = Arc::downgrade(&mr);
        guard.mr_cache.push(mr_weak);
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

pub fn ib_deregister_mr(handle: IbMrHandle) {
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

#[inline]
fn ib_free_request(request: &mut IbRequest) {
    request.ty = RequestType::Unused;
}

fn ib_multi_send(comm: Pin<&mut IbSendComm<'_>>, slot: usize) -> Result<(), IbError> {
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
        let slot_ptr = unsafe { VolatilePtr::new(slot_non_null) };
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
    if num_requests == 1 {
        imm_data = unsafe { comm.verbs.requests[requests_idx[0]].send_recv.send.size as u32 };
    } else {
        assert!(
            num_requests <= 32,
            "Cannot store sizes of {} requests in a 32-bits field",
            num_requests
        );
        for r in 0..num_requests {
            let size = unsafe { comm.verbs.requests[requests_idx[r]].send_recv.send.size };
            imm_data |= (if size > 0 { 1 } else { 0 }) << r;
        }
    }

    let mut last_wr = &mut comm.wrs[num_requests - 1];

    let first_size = unsafe { comm.verbs.requests[requests_idx[0]].send_recv.send.size };
    if num_requests > 1 || (comm.adaptive_routing && first_size > comm.ar_threshold) {
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
            if send_size <= offset {
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
            let qp_ref = &*qp_ptr;
            let ctx = &mut *qp_ref.context;
            let send_fn = ctx.ops.post_send.as_mut().unwrap();

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
            Err(IbError::SendOverflow {
                send: size,
                recv: recv_size,
            })?;
        }
        let remote_addr = map_field!(slot_ptr.addr).read();
        let rkey = map_field!(slot_ptr.rkey).read();
        if remote_addr == 0 || rkey == 0 {
            panic!("Peer posted incorrect receive info");
        }
        let request_id =
            ib_get_request(&mut comm.verbs).ok_or_else(|| IbError::RequestBufferFull)?;
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
        if comm.fifo_requests_idx[slot][0..num_requests]
            .iter()
            .any(|id| *id == IB_MAX_REQUESTS)
        {
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
    Ok(None)
}

fn ib_post_fifo(
    comm: Pin<&mut IbRecvComm<'_>>,
    data: &[*mut c_void],
    sizes: &[usize],
    tags: &[u32],
    mr_handles: &[&AnyMrHandle],
    request_id: usize,
) -> Result<(), IbError> {
    let comm = unsafe { comm.get_unchecked_mut() };
    let mut wr = ibv_send_wr::default();
    let slot = (comm.remote_fifo.fifo_tail % IB_MAX_REQUESTS as u64) as usize;
    let local_elems = &mut comm.remote_fifo.mr[slot * IB_MAX_RECVS..(slot + 1) * IB_MAX_RECVS];

    let num_requests = data.len();
    for i in 0..num_requests {
        local_elems[i].addr = data[i].addr() as u64;
        let mr = unsafe {
            let mr_handle = mr_handles[i]
                .downcast_ref::<IbMrHandle>()
                .ok_or_else(|| IbError::DowncastMrHandle)?;
            &*mr_handle.0.get_mr()
        };
        local_elems[i].rkey = mr.rkey;
        local_elems[i].num_requests = num_requests as u32;
        local_elems[i].size = sizes[i] as u32;
        local_elems[i].tag = tags[i];
        local_elems[i].idx = comm.remote_fifo.fifo_tail + 1;
    }
    wr.wr.rdma.remote_addr =
        comm.remote_fifo.addr + (slot * IB_MAX_RECVS * std::mem::size_of::<IbSendFifo>()) as u64;
    wr.wr.rdma.rkey = comm.remote_fifo.rkey;
    comm.remote_fifo.sge.addr = local_elems.as_ptr() as u64;
    comm.remote_fifo.sge.length = (num_requests * std::mem::size_of::<IbSendFifo>()) as u32;
    wr.sg_list = &mut comm.remote_fifo.sge;
    wr.num_sge = 1;
    wr.opcode = ibv_wr_opcode::IBV_WR_RDMA_WRITE;
    wr.send_flags = comm.remote_fifo.flags.0;

    if slot == 0 {
        wr.send_flags |= ibv_send_flags::IBV_SEND_SIGNALED.0;
        wr.wr_id = request_id as u64;
        comm.verbs.requests[request_id].events += 1;
    }
    unsafe {
        let qp_ptr = comm.qps[0].get_qp();
        let qp_ref = &*qp_ptr;
        let ctx = &mut *qp_ref.context;
        let send_fn = ctx.ops.post_send.as_mut().unwrap();

        let mut bad_wr = std::ptr::null_mut();
        ibv_check!(send_fn(qp_ptr, &mut wr, &mut bad_wr));
    }
    comm.remote_fifo.fifo_tail += 1;

    Ok(())
}

pub fn ib_initiate_recv(
    comm: Pin<&mut IbRecvComm<'_>>,
    data: &[*mut c_void],
    sizes: &[usize],
    tags: &[u32],
    mr_handles: &[&AnyMrHandle],
) -> Result<Option<IbRequestId>, IbError> {
    let comm = unsafe { comm.get_unchecked_mut() };
    let num_requests = data.len();
    if num_requests > IB_MAX_RECVS {
        Err(IbError::ExceedMaxRecv(num_requests))?;
    }
    if sizes.len() != num_requests || tags.len() != num_requests || mr_handles.len() != num_requests
    {
        Err(IbError::NumElemsMismatch(num_requests))?;
    }

    let request_id = ib_get_request(&mut comm.verbs).ok_or_else(|| IbError::RequestBufferFull)?;
    let request = &mut comm.verbs.requests[request_id];
    request.ty = RequestType::Recv;
    request.num_requests = num_requests as u32;
    for i in 0..num_requests {
        unsafe {
            request.send_recv.recv.sizes[i] = 0;
        }
    }

    let mut wr = ibv_recv_wr::default();
    wr.wr_id = request_id as u64;
    wr.sg_list = std::ptr::null_mut();
    wr.num_sge = 0;

    for qp in comm.qps.iter() {
        unsafe {
            let qp_ptr = qp.get_qp();
            let qp_ref = &*qp_ptr;
            let ctx = &mut *qp_ref.context;
            let recv_fn = ctx.ops.post_recv.as_mut().unwrap();

            let mut bad_wr = std::ptr::null_mut();
            ibv_check!(recv_fn(qp_ptr, &mut wr, &mut bad_wr));
        }
    }
    request.events = comm.qps.len() as u32;

    ib_post_fifo(
        unsafe { Pin::new_unchecked(comm) },
        data,
        sizes,
        tags,
        mr_handles,
        request_id,
    )?;

    let request_id = IbRequestId(request_id);
    Ok(Some(request_id))
}

pub fn ib_initiate_flush(
    comm: Pin<&mut IbRecvComm<'_>>,
    data: &[*mut c_void],
    sizes: &[usize],
    mr_handles: &[&AnyMrHandle],
) -> Result<Option<IbRequestId>, IbError> {
    let comm = unsafe { comm.get_unchecked_mut() };
    let mut last = None;
    for (i, size) in sizes.iter().enumerate() {
        if *size > 0 {
            last = Some(i);
        }
    }
    if !comm.flush.enabled || last.is_none() {
        return Ok(None);
    }
    let num_requests = data.len();
    if num_requests > IB_MAX_RECVS {
        Err(IbError::ExceedMaxRecv(num_requests))?;
    }
    if sizes.len() != num_requests || mr_handles.len() != num_requests {
        Err(IbError::NumElemsMismatch(num_requests))?;
    }

    let last = last.unwrap();
    let request_id = ib_get_request(&mut comm.verbs).ok_or_else(|| IbError::RequestBufferFull)?;
    let request = &mut comm.verbs.requests[request_id];
    request.ty = RequestType::Flush;

    let mr = mr_handles[0]
        .downcast_ref::<IbMrHandle>()
        .ok_or_else(|| IbError::DowncastMrHandle)?;
    let mr_ptr = mr.0.get_mr();
    let mr_ref = unsafe { &*mr_ptr };

    let mut wr = ibv_send_wr::default();
    wr.wr_id = request_id as u64;

    wr.wr.rdma.remote_addr = data[last].addr() as u64;
    wr.wr.rdma.rkey = mr_ref.rkey;
    wr.sg_list = &mut comm.flush.sge;
    wr.num_sge = 1;
    wr.opcode = ibv_wr_opcode::IBV_WR_RDMA_READ;
    wr.send_flags = ibv_send_flags::IBV_SEND_SIGNALED.0;

    unsafe {
        let qp_ptr = comm.flush.qp.as_mut().unwrap().get_qp();
        let qp_ref = &*qp_ptr;
        let ctx = &mut *qp_ref.context;
        let send_fn = ctx.ops.post_send.as_mut().unwrap();

        let mut bad_wr = std::ptr::null_mut();
        ibv_check!(send_fn(qp_ptr, &mut wr, &mut bad_wr));
    }
    todo!()
}

pub fn ib_test(
    verbs: &mut IbVerbs<'_>,
    request_id: IbRequestId,
    sizes: Option<&mut [usize]>,
) -> Result<bool, IbError> {
    loop {
        let request = &mut verbs.requests[request_id.0];
        if request.events == 0 {
            if sizes.is_some() && request.ty == RequestType::Recv {
                let sizes = sizes.unwrap();
                for i in 0..request.num_requests as usize {
                    sizes[i] = unsafe { request.send_recv.recv.sizes[i] };
                }
            }
            ib_free_request(request);
            return Ok(true);
        }

        let mut wcs = [ibv_wc::default(); 4];
        let cq_ptr = verbs.cq.get_cq();
        let cq_ref = unsafe { &*cq_ptr };
        let ctx = unsafe { &mut *cq_ref.context };
        let poll_fn = ctx.ops.poll_cq.as_mut().unwrap();
        let wr_done = unsafe { poll_fn(cq_ptr, 4, wcs.as_mut_ptr()) };
        if wr_done < 0 {
            Err(IbError::PollCq)?;
        }
        if wr_done == 0 {
            return Ok(false);
        }

        for wc in wcs[0..wr_done as usize].iter_mut() {
            if !wc.is_valid() {
                let (status, vendor_err) = wc.error().unwrap();
                Err(IbError::WcError(wc.opcode(), wc.len(), status, vendor_err))?;
            }
            let wr_id = wc.wr_id();
            let root_request = &mut verbs.requests[wr_id as usize & 0xff];
            if root_request.ty == RequestType::Send {
                for i in 0..root_request.num_requests as usize {
                    let send_request_id = (wr_id as usize >> (i * 8)) & 0xff;
                    let send_request = &mut verbs.requests[send_request_id];
                    assert!(send_request.events > 0, "Request already completed");
                    send_request.events -= 1;
                }
            } else {
                if wc.opcode() == ibv_wc_opcode::IBV_WC_RECV_RDMA_WITH_IMM {
                    assert_eq!(
                        root_request.ty,
                        RequestType::Recv,
                        "Unexpected request type"
                    );
                    let imm = wc.imm_data().unwrap();
                    if root_request.num_requests > 1 {
                        // In the case of a multi recv, we only set sizes to 0 or 1.
                        for i in 0..root_request.num_requests as usize {
                            unsafe {
                                root_request.send_recv.recv.sizes[i] = (imm as usize >> i) & 0x1;
                            }
                        }
                    } else {
                        unsafe {
                            root_request.send_recv.recv.sizes[0] += imm as usize;
                        }
                    }
                }
                root_request.events -= 1;
            }
        }
    }
}

#[async_trait]
impl NetProvider for RdmaTransportProvider {
    type NetError = IbError;
    type NetHandle = IbConnectHandle;

    #[inline]
    fn init(&self, catalog: &TransportCatalog) -> Result<(), Self::NetError> {
        let config = catalog
            .get_config::<RdmaTransportConfig>("NetProviderRdma")
            .map_err(|_| IbError::ConfigNotFound)?
            .clone();
        let context = ib_init_transport_context(config)?;
        RDMA_TRANSPORT
            .0
            .set(context)
            .map_err(|_| IbError::ContextAlreadyInitialized)?;
        Ok(())
    }

    #[inline]
    fn get_num_devices(&self) -> Result<usize, Self::NetError> {
        ib_get_num_devices()
    }

    #[inline]
    fn get_properties(&self, device: usize) -> Result<NetProperties, Self::NetError> {
        ib_get_properties(device)
    }

    #[inline]
    async fn listen(&self, device: usize) -> Result<NetListener<Self::NetHandle>, Self::NetError> {
        let (handle, listen_comm) = ib_listen(device).await?;
        let listener = NetListener {
            handle,
            listen_comm: Box::new(listen_comm),
        };
        Ok(listener)
    }

    async fn connect(
        &self,
        device: usize,
        handle: Self::NetHandle,
        udp_sport: Option<u16>,
        tc: Option<u8>,
    ) -> Result<Box<AnyNetComm>, Self::NetError> {
        let send_comm = ib_connect(device, &handle, udp_sport, tc).await?;
        Ok(Box::new(send_comm))
    }
    // Finalize connection establishment after remote peer has called connect.
    async fn accept(
        &self,
        listen_comm: Box<AnyNetComm>,
        tc: Option<u8>,
    ) -> Result<Box<AnyNetComm>, Self::NetError> {
        let listen_comm = *listen_comm
            .downcast::<IbListenComm>()
            .map_err(|_| IbError::DowncastNetComm)?;
        let recv_comm = ib_accept(listen_comm, tc).await?;
        Ok(Box::new(recv_comm))
    }

    async fn register_mr(
        &self,
        comm: Pin<&mut AnyNetComm>,
        comm_type: CommType,
        mr: MemoryRegion,
    ) -> Result<Box<AnyMrHandle>, Self::NetError> {
        let comm = unsafe { comm.get_unchecked_mut() };
        let verbs = match comm_type {
            CommType::Send => {
                let send_comm = comm
                    .downcast_mut::<IbSendComm<'static>>()
                    .ok_or_else(|| IbError::DowncastNetComm)?;
                &mut send_comm.verbs
            }
            CommType::Recv => {
                let recv_comm = comm
                    .downcast_mut::<IbRecvComm<'static>>()
                    .ok_or_else(|| IbError::DowncastNetComm)?;
                &mut recv_comm.verbs
            }
        };
        let handle = ib_register_mr(verbs, mr.data, mr.size)?;
        Ok(Box::new(handle))
    }

    async fn register_mr_dma_buf(
        &self,
        comm: Pin<&mut AnyNetComm>,
        comm_type: CommType,
        mr: MemoryRegion,
        offset: u64,
        fd: RawFd,
    ) -> Result<Box<AnyMrHandle>, Self::NetError> {
        let comm = unsafe { comm.get_unchecked_mut() };
        let verbs = match comm_type {
            CommType::Send => {
                let send_comm = comm
                    .downcast_mut::<IbSendComm<'static>>()
                    .ok_or_else(|| IbError::DowncastNetComm)?;
                &mut send_comm.verbs
            }
            CommType::Recv => {
                let recv_comm = comm
                    .downcast_mut::<IbRecvComm<'static>>()
                    .ok_or_else(|| IbError::DowncastNetComm)?;
                &mut recv_comm.verbs
            }
        };
        let handle = ib_register_mr_dma_buf(verbs, mr.data, mr.size, offset, fd)?;
        Ok(Box::new(handle))
    }

    async fn deregister_mr(
        &self,
        _comm: Pin<&mut AnyNetComm>,
        _comm_type: CommType,
        handle: Box<AnyMrHandle>,
    ) -> Result<(), Self::NetError> {
        let handle = *handle
            .downcast::<IbMrHandle>()
            .map_err(|_| IbError::DowncastMrHandle)?;
        ib_deregister_mr(handle);
        Ok(())
    }

    fn initiate_send(
        &self,
        send_comm: Pin<&mut AnyNetComm>,
        data: *mut c_void,
        size: usize,
        tag: u32,
        mr_handle: &AnyMrHandle,
    ) -> Result<Option<NetRequestId>, Self::NetError> {
        let send_comm = unsafe {
            let comm = send_comm
                .get_unchecked_mut()
                .downcast_mut::<IbSendComm<'static>>()
                .ok_or(IbError::DowncastNetComm)?;
            Pin::new_unchecked(comm)
        };
        let mr_handle = mr_handle
            .downcast_ref::<IbMrHandle>()
            .ok_or(IbError::DowncastMrHandle)?;
        let request_id = ib_initiate_send(send_comm, data, size, tag, mr_handle)?
            .map(|id| NetRequestId(id.0 as u32));
        Ok(request_id)
    }

    // Initiate an asynchronous recv operation from a eer.
    // If None is returned, then the send request cannot be performed now,
    // or would block, retry later
    fn initiate_recv(
        &self,
        recv_comm: Pin<&mut AnyNetComm>,
        data: &[*mut c_void],
        sizes: &[usize],
        tags: &[u32],
        mr_handles: &[&AnyMrHandle],
    ) -> Result<Option<NetRequestId>, Self::NetError> {
        let recv_comm = unsafe {
            let comm = recv_comm
                .get_unchecked_mut()
                .downcast_mut::<IbRecvComm<'static>>()
                .ok_or(IbError::DowncastNetComm)?;
            Pin::new_unchecked(comm)
        };
        let request_id = ib_initiate_recv(recv_comm, data, sizes, tags, mr_handles)?
            .map(|id| NetRequestId(id.0 as u32));
        Ok(request_id)
    }

    fn initiate_flush(
        &self,
        recv_comm: Pin<&mut AnyNetComm>,
        data: &[*mut c_void],
        sizes: &[usize],
        mr_handles: &[&AnyMrHandle],
    ) -> Result<Option<NetRequestId>, Self::NetError> {
        let recv_comm = unsafe {
            let comm = recv_comm
                .get_unchecked_mut()
                .downcast_mut::<IbRecvComm<'static>>()
                .ok_or(IbError::DowncastNetComm)?;
            Pin::new_unchecked(comm)
        };
        let request_id = ib_initiate_flush(recv_comm, data, sizes, mr_handles)?
            .map(|id| NetRequestId(id.0 as u32));
        Ok(request_id)
    }

    fn test(
        &self,
        comm: Pin<&mut AnyNetComm>,
        request: NetRequestId,
        comm_type: CommType,
        sizes: Option<&mut [usize]>,
    ) -> Result<bool, Self::NetError> {
        let comm = unsafe { comm.get_unchecked_mut() };
        let verbs = match comm_type {
            CommType::Send => {
                let send_comm = comm
                    .downcast_mut::<IbSendComm<'static>>()
                    .ok_or_else(|| IbError::DowncastNetComm)?;
                &mut send_comm.verbs
            }
            CommType::Recv => {
                let recv_comm = comm
                    .downcast_mut::<IbRecvComm<'static>>()
                    .ok_or_else(|| IbError::DowncastNetComm)?;
                &mut recv_comm.verbs
            }
        };
        let request_id = IbRequestId(request.0 as usize);
        ib_test(verbs, request_id, sizes)
    }
}
