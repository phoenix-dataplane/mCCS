use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr::addr_of_mut;
use std::ptr::NonNull;
use std::sync::atomic::AtomicBool;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use gcollections::ops::Contains;
use strum::IntoEnumIterator;
use volatile::VolatilePtr;

use cuda_driver_sys::cuMemGetHandleForAddressRange;
use cuda_driver_sys::CUmemRangeHandleType;
use cuda_runtime_sys::cudaGetDevice;
use qos_service::{QosMode, QosSchedule};

use super::buffer::{BufferMap, BufferType, MemoryBankAlloc, MemoryBankType};
use super::config::NetTransportConfig;
use super::provider::PtrSupport;
use super::provider::{CommType, MemoryRegion, MrType, NetRequestId};
use super::resources::{AgentRecvConnectReply, AgentSendConnectReply};
use super::resources::{
    AgentRecvConnectRequest, AgentRecvResources, AgentSendConnectRequest, AgentSendResources,
};
use super::resources::{AgentRecvSetup, AgentRecvSetupReply, AgentSendSetup, AgentSetupRequest};
use super::NetAgentError;
use crate::cuda::alloc::{DeviceAlloc, DeviceHostMapped};
use crate::cuda_warning;
use crate::transport::catalog::TransportCatalog;
use crate::transport::meta::{RecvBufMeta, SendBufMeta};
use crate::transport::op::TransportOp;
use crate::transport::op::TransportOpState;
use crate::transport::Protocol;
use crate::transport::NUM_BUFFER_SLOTS;
use crate::utils::gdr::{check_dma_buf_support, wc_store_fence, GdrMappedMem, GDR_HANDLE};

type Result<T> = std::result::Result<T, NetAgentError>;

pub async fn net_agent_send_setup(
    request: AgentSetupRequest,
    catalog: &TransportCatalog,
) -> Result<AgentSendSetup> {
    let config = catalog.get_config::<NetTransportConfig>("NetTransport")?;
    let props = request.provider.get_properties(request.net_device)?;
    let device_idx = unsafe {
        let mut dev = 0;
        cuda_warning!(cudaGetDevice(&mut dev));
        dev
    };
    let dma_buf_support = check_dma_buf_support(device_idx);
    let use_dma_buf = if request.use_gdr
        && dma_buf_support
        && props.ptr_support.contains(PtrSupport::PTR_DMA_BUF)
    {
        true
    } else {
        false
    };
    let resources = AgentSendSetup {
        rank: request.rank,
        local_rank: request.local_rank,
        remote_rank: request.remote_rank,
        net_device: request.net_device,
        use_gdr: request.use_gdr,
        use_dma_buf,
        max_recvs: props.max_recvs,
        buffer_sizes: request.buffer_sizes,
        provider: request.provider,
        gdr_copy_sync_enable: config.gdr_copy_sync_enable,
        udp_sport: request.udp_sport,
        tc: request.tc,
    };
    Ok(resources)
}

pub async fn net_agent_recv_setup(
    request: AgentSetupRequest,
    catalog: &TransportCatalog,
) -> Result<(AgentRecvSetupReply, AgentRecvSetup)> {
    let (gdr_copy_sync_enable, gdr_copy_flush_enable) = {
        let config = catalog.get_config::<NetTransportConfig>("NetTransport")?;
        (config.gdr_copy_sync_enable, config.gdr_copy_flush_enable)
    };
    let props = request.provider.get_properties(request.net_device)?;
    let device_idx = unsafe {
        let mut dev = 0;
        cuda_warning!(cudaGetDevice(&mut dev));
        dev
    };
    let dma_buf_support = check_dma_buf_support(device_idx);
    let use_dma_buf = if request.use_gdr
        && dma_buf_support
        && props.ptr_support.contains(PtrSupport::PTR_DMA_BUF)
    {
        true
    } else {
        false
    };
    let listener = request.provider.listen(request.net_device).await?;

    let resources = AgentRecvSetup {
        rank: request.rank,
        local_rank: request.local_rank,
        remote_rank: request.remote_rank,
        net_device: request.net_device,
        use_gdr: request.use_gdr,
        use_dma_buf,
        max_recvs: props.max_recvs,
        buffer_sizes: request.buffer_sizes,
        provider: request.provider,
        listen_comm: listener.listen_comm,
        need_flush: request.need_flush,
        gdr_copy_sync_enable,
        gdr_copy_flush_enable,
        tc: request.tc,
    };
    let reply = AgentRecvSetupReply {
        handle: listener.handle,
    };
    Ok((reply, resources))
}

pub async fn net_agent_send_connect(
    request: AgentSendConnectRequest,
    setup_resources: AgentSendSetup,
) -> Result<(AgentSendConnectReply, AgentSendResources)> {
    let provider = setup_resources.provider;
    let send_comm = provider
        .connect(
            setup_resources.net_device,
            &request.handle,
            setup_resources.udp_sport,
            setup_resources.tc,
        )
        .await?;
    let mut send_comm = Box::into_pin(send_comm);
    let mut map = BufferMap::new();

    let buffer_sizes = setup_resources.buffer_sizes;
    for proto in Protocol::iter() {
        let buffer_type = BufferType::RingBuffer(proto);
        map.assign_buffer_memory(
            buffer_type,
            buffer_sizes[proto as usize],
            setup_resources.use_gdr,
        )
    }

    map.assign_buffer_memory(
        BufferType::SendMem,
        std::mem::size_of::<SendBufMeta>(),
        false,
    );
    map.assign_buffer_memory(
        BufferType::RecvMem,
        std::mem::size_of::<RecvBufMeta>(),
        false,
    );

    let dev_size = map.get_bank_alloc_size(MemoryBankType::DeviceMem);
    if dev_size > 0 {
        let dev_mem = DeviceAlloc::new(dev_size);
        let alloc = MemoryBankAlloc::Device(dev_mem);
        map.register_bank_alloc(alloc);
    }

    let host_size = map.get_bank_alloc_size(MemoryBankType::HostMem);
    if host_size > 0 {
        let host_mem = DeviceHostMapped::alloc(host_size);
        let alloc = MemoryBankAlloc::Host(host_mem);
        map.register_bank_alloc(alloc);
    }

    let mut gdc_sync = 0;
    // In NCCLL: if (ncclGdrCopy && map->sameProcess && ncclParamGdrCopySyncEnable())
    if setup_resources.use_gdr && !GDR_HANDLE.0.is_null() && setup_resources.gdr_copy_sync_enable {
        let gdc_mem = GdrMappedMem::new(1);
        gdc_sync = (gdc_mem.get_cpu_ptr() as *mut u64).addr();
        let alloc = MemoryBankAlloc::GdcMem(gdc_mem);
        map.register_bank_alloc(alloc);
    }

    let mut mr_handles = MaybeUninit::uninit_array();
    let mut buffers = MaybeUninit::uninit_array();
    for proto in Protocol::iter() {
        let ptr_addr = map
            .get_buffer_cpu_ptr(proto)
            .map(|x| (x.as_ptr() as *mut c_void).addr());
        if let Some(ptr_addr) = ptr_addr {
            buffers[proto as usize].write(ptr_addr);
            let buffer_size = buffer_sizes[proto as usize];
            let mr_type = match map.get_buffer_bank(proto) {
                MemoryBankType::HostMem => MrType::Host,
                MemoryBankType::DeviceMem => MrType::Device,
                MemoryBankType::GdcMem => unreachable!("GDC memory is not used for ring buffer"),
            };
            if mr_type == MrType::Device && setup_resources.use_dma_buf {
                unsafe {
                    let mut dmabuf_fd: i32 = 0;
                    cuMemGetHandleForAddressRange(
                        (&mut dmabuf_fd) as *mut i32 as *mut c_void,
                        ptr_addr as _,
                        buffer_size,
                        CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                        0,
                    );
                    let mr = MemoryRegion {
                        data: ptr_addr as *mut c_void,
                        size: buffer_size,
                        mr_type: MrType::Device,
                    };
                    let mhandle = provider
                        .register_mr_dma_buf(send_comm.as_mut(), CommType::Send, mr, 0, dmabuf_fd)
                        .await?;
                    mr_handles[proto as usize].write(Some(mhandle));
                    nix::unistd::close(dmabuf_fd).map_err(|e| {
                        NetAgentError::BufferRegistration(format!(
                            "Failed to close fd for DMA buffer with errno {}",
                            e
                        ))
                    })?;
                }
            } else {
                let mr = MemoryRegion {
                    data: ptr_addr as *mut c_void,
                    size: buffer_size,
                    mr_type: MrType::Host,
                };
                let mhandle = provider
                    .register_mr(send_comm.as_mut(), CommType::Send, mr)
                    .await?;
                mr_handles[proto as usize].write(Some(mhandle));
            }
        } else {
            buffers[proto as usize].write(0);
            mr_handles[proto as usize].write(None);
        }
    }

    let buffers = unsafe { MaybeUninit::array_assume_init(buffers) };
    let buffers = buffers.map(|x| x as *mut c_void);
    let mr_handles = unsafe { MaybeUninit::array_assume_init(mr_handles) };

    let send_mem = map.get_send_mem_meta().unwrap();
    let recv_mem = map.get_recv_mem_meta().unwrap();
    let send_mem_mut = unsafe { &mut *send_mem.as_ptr_host() };
    send_mem_mut.head = 0;
    let recv_mem_mut = unsafe { &mut *recv_mem.as_ptr_host() };
    for i in 0..NUM_BUFFER_SLOTS {
        recv_mem_mut.slots_sizes[i] = -1;
    }

    let recv_resources = AgentSendResources {
        map: map.clone(),
        send_comm,
        send_mem,
        recv_mem,
        rank: setup_resources.rank,
        local_rank: setup_resources.local_rank,
        remote_rank: setup_resources.remote_rank,
        net_device: setup_resources.net_device,
        use_gdr: setup_resources.use_gdr,
        use_dma_buf: setup_resources.use_dma_buf,
        max_recvs: setup_resources.max_recvs,
        gdc_sync: gdc_sync as *mut u64,
        buffers,
        buffer_sizes,
        mr_handles,
        step: 0,
        provider,
        qos_round: 0,
    };

    let device_idx = unsafe {
        let mut dev = 0;
        cuda_warning!(cudaGetDevice(&mut dev));
        dev
    };
    let reply = AgentSendConnectReply {
        map,
        agent_cuda_dev: device_idx,
    };
    Ok((reply, recv_resources))
}

pub static QOS_DISABLE: AtomicBool = AtomicBool::new(false);

pub async fn net_agent_recv_connect(
    request: AgentRecvConnectRequest,
    setup_resources: AgentRecvSetup,
) -> Result<(AgentRecvConnectReply, AgentRecvResources)> {
    let provider = setup_resources.provider;
    let recv_comm = provider
        .accept(setup_resources.listen_comm, setup_resources.tc)
        .await?;
    let mut recv_comm = Box::into_pin(recv_comm);
    let mut map = BufferMap::new();

    let buffer_sizes = setup_resources.buffer_sizes;
    for proto in Protocol::iter() {
        let buffer_type = BufferType::RingBuffer(proto);
        map.assign_buffer_memory(
            buffer_type,
            buffer_sizes[proto as usize],
            setup_resources.use_gdr,
        )
    }

    map.assign_buffer_memory(
        BufferType::SendMem,
        std::mem::size_of::<SendBufMeta>(),
        false,
    );
    map.assign_buffer_memory(
        BufferType::RecvMem,
        std::mem::size_of::<RecvBufMeta>(),
        false,
    );

    let dev_size = map.get_bank_alloc_size(MemoryBankType::DeviceMem);
    if dev_size > 0 {
        let dev_mem = DeviceAlloc::new(dev_size);
        let alloc = MemoryBankAlloc::Device(dev_mem);
        map.register_bank_alloc(alloc);
    }

    let host_size = map.get_bank_alloc_size(MemoryBankType::HostMem);
    if host_size > 0 {
        let host_mem = DeviceHostMapped::alloc(host_size);
        let alloc = MemoryBankAlloc::Host(host_mem);
        map.register_bank_alloc(alloc);
    }

    let mut gdc_sync = 0;
    let mut gdc_flush = 0;
    // In NCCL: if (ncclGdrCopy && map->sameProcess)
    if setup_resources.use_gdr && !GDR_HANDLE.0.is_null() && setup_resources.gdr_copy_sync_enable {
        let gdc_mem = GdrMappedMem::new(2);
        gdc_sync = (gdc_mem.get_cpu_ptr() as *mut u64).addr();
        if setup_resources.gdr_copy_flush_enable {
            gdc_flush = unsafe { (gdc_mem.get_cpu_ptr() as *mut u64).add(1).addr() };
        }
        let alloc = MemoryBankAlloc::GdcMem(gdc_mem);
        map.register_bank_alloc(alloc);
    }

    let mut mr_handles = MaybeUninit::uninit_array();
    let mut buffers = MaybeUninit::uninit_array();
    for proto in Protocol::iter() {
        let ptr_addr = map
            .get_buffer_cpu_ptr(proto)
            .map(|x| (x.as_ptr() as *mut c_void).addr());
        if let Some(ptr_addr) = ptr_addr {
            buffers[proto as usize].write(ptr_addr);
            let buffer_size = buffer_sizes[proto as usize];
            let mr_type = match map.get_buffer_bank(proto) {
                MemoryBankType::HostMem => MrType::Host,
                MemoryBankType::DeviceMem => MrType::Device,
                MemoryBankType::GdcMem => unreachable!("GDC memory is not used for ring buffer"),
            };
            if mr_type == MrType::Device && setup_resources.use_dma_buf {
                unsafe {
                    let mut dmabuf_fd: i32 = 0;
                    cuMemGetHandleForAddressRange(
                        (&mut dmabuf_fd) as *mut i32 as *mut c_void,
                        ptr_addr as _,
                        buffer_size,
                        CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                        0,
                    );
                    let mr = MemoryRegion {
                        data: ptr_addr as *mut c_void,
                        size: buffer_size,
                        mr_type: MrType::Device,
                    };
                    let mhandle = provider
                        .register_mr_dma_buf(recv_comm.as_mut(), CommType::Recv, mr, 0, dmabuf_fd)
                        .await?;
                    mr_handles[proto as usize].write(Some(mhandle));
                    nix::unistd::close(dmabuf_fd).map_err(|e| {
                        NetAgentError::BufferRegistration(format!(
                            "Failed to close fd for DMA buffer with errno {}",
                            e
                        ))
                    })?;
                }
            } else {
                let mr = MemoryRegion {
                    data: ptr_addr as *mut c_void,
                    size: buffer_size,
                    mr_type: MrType::Host,
                };
                let mhandle = provider
                    .register_mr(recv_comm.as_mut(), CommType::Recv, mr)
                    .await?;
                mr_handles[proto as usize].write(Some(mhandle));
            }
        } else {
            buffers[proto as usize].write(0);
            mr_handles[proto as usize].write(None);
        }
    }

    let buffers = unsafe { MaybeUninit::array_assume_init(buffers) };
    let buffers = buffers.map(|x| x as *mut c_void);
    let mr_handles = unsafe { MaybeUninit::array_assume_init(mr_handles) };

    let send_mem = map.get_send_mem_meta().unwrap();
    let recv_mem = map.get_recv_mem_meta().unwrap();

    let recv_resources = AgentRecvResources {
        map: map.clone(),
        recv_comm,
        send_mem,
        recv_mem,
        rank: setup_resources.rank,
        local_rank: setup_resources.local_rank,
        remote_rank: setup_resources.remote_rank,
        agent_rank: request.send_agent_rank,
        net_device: setup_resources.net_device,
        use_gdr: setup_resources.use_gdr,
        use_dma_buf: setup_resources.use_dma_buf,
        need_flush: setup_resources.need_flush,
        max_recvs: setup_resources.max_recvs,
        gdc_sync: gdc_sync as *mut u64,
        gdc_flush: gdc_flush as *mut u64,
        buffers,
        buffer_sizes,
        mr_handles,
        step: 0,
        provider: setup_resources.provider,
    };
    let reply = AgentRecvConnectReply { map };
    Ok((reply, recv_resources))
}

pub fn net_agent_send_progress(
    resources: &mut AgentSendResources,
    op: &mut TransportOp,
    schedule: &QosSchedule,
) -> Result<()> {
    if op.state == TransportOpState::Init {
        op.base = resources.step.div_ceil(op.chunk_steps as u64) * (op.chunk_steps as u64);
        op.posted = 0;
        op.transmitted = 0;
        op.done = 0;
        op.state = TransportOpState::InProgress;
    }
    op.idle = true;
    if op.state != TransportOpState::InProgress {
        return Ok(());
    }

    // log::trace!("Send op: {:?}", op);

    let provider = resources.provider;
    let proto = op.protocol;
    let num_steps = op.num_steps as u64;
    let max_depth = NUM_BUFFER_SLOTS as u64;

    let mhandle = resources.mr_handles[proto as usize]
        .as_ref()
        .unwrap()
        .as_ref();
    let step_size = resources.buffer_sizes[proto as usize] / NUM_BUFFER_SLOTS;
    let local_buffer = resources.map.get_buffer_cpu_ptr(proto).unwrap();
    if op.posted < num_steps && op.posted < op.done + max_depth {
        op.posted += op.slice_steps as u64;
        op.idle = false;
        return Ok(());
    }

    if op.transmitted < op.posted && op.transmitted < op.done + NUM_BUFFER_SLOTS as u64 {
        let buffer_slot = (op.base + op.transmitted) as usize % NUM_BUFFER_SLOTS;
        let sizes_fifo = unsafe {
            let recv_mem_ptr = resources.recv_mem.as_ptr_host();
            let sizes_fifo_ptr = addr_of_mut!((*recv_mem_ptr).slots_sizes);
            let non_null = NonNull::new_unchecked(sizes_fifo_ptr as *mut [i32]);
            VolatilePtr::new(non_null)
        };
        let size_ptr = unsafe { sizes_fifo.map(|x| x.get_unchecked_mut(buffer_slot)) };
        let recv_tail = unsafe {
            let recv_mem_ptr = resources.recv_mem.as_ptr_host();
            let recv_tail_ptr = addr_of_mut!((*recv_mem_ptr).tail);
            let non_null = NonNull::new_unchecked(recv_tail_ptr);
            VolatilePtr::new(non_null)
        };
        log::trace!(
            "#{} net_agent_send_progress()[1/2]: *size_ptr={}, *recv_tail={}, base={}, transmitted={}, done={}, num_steps={}",
            op.debug_id,
            size_ptr.read(),
            recv_tail.read(),
            op.base,
            op.transmitted,
            op.done,
            op.num_steps
        );
        if size_ptr.read() != -1 && recv_tail.read() > op.base + op.transmitted {
            log::trace!(
                "#{} net_agent_send_progress()[2/2]: *size_ptr={}, *recv_tail={}, base={}, transmitted={}, done={}",
                op.debug_id,
                size_ptr.read(),
                recv_tail.read(),
                op.base,
                op.transmitted,
                op.done
            );
            let size = size_ptr.read() as usize;
            let offset = buffer_slot * step_size;
            let buffer_ptr = unsafe { local_buffer.as_ptr().byte_add(offset) };
            let ready = true;
            if ready {
                let comm_id = qos_service::CommunicatorId(op.communicator_id.0);
                let interval = schedule.schedule.get(&comm_id);
                let delay_send = if let Some(interval) = interval {
                    let enforce = if let Some(step) = interval.enforce_step {
                        resources.qos_round % step == 0 
                    } else {
                        false
                    };
                    if enforce {
                        let time = SystemTime::now();
                        let elapsed = time.duration_since(UNIX_EPOCH).unwrap();
                        let epoch_ts = (elapsed.as_micros() % schedule.epoch_microsecs as u128) as u64;
                        match interval.mode {
                            QosMode::Allow => !interval.intervals.contains(&epoch_ts),
                            QosMode::Deny => interval.intervals.contains(&epoch_ts),
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };
                log::trace!(
                    "net_agent_send_progress()[delay]: interval={:?} delay_send={}",
                    interval,
                    delay_send
                );
                if !delay_send {
                    let request_id = provider.initiate_send(
                        resources.send_comm.as_mut(),
                        buffer_ptr,
                        size as usize,
                        resources.rank as u32,
                        mhandle,
                    )?;
                    if let Some(request_id) = request_id {
                        op.requests_id[buffer_slot] = Some(request_id.0);
                        size_ptr.write(-1);
                        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                        op.transmitted += op.slice_steps as u64;
                        op.idle = false;
                        return Ok(());
                    }
                    // }
                    else {
                        log::trace!("net_agent_send_progress: WTF it's None")
                    }
                }
            }
        }
    }
    if op.done < op.transmitted {
        let buffer_slot = (op.base + op.done) as usize % NUM_BUFFER_SLOTS;
        let request_id = NetRequestId(op.requests_id[buffer_slot].unwrap());
        let done = provider.test(
            resources.send_comm.as_mut(),
            request_id,
            CommType::Send,
            None,
        )?;
        if done {
            op.done += op.slice_steps as u64;
            let send_head = if !resources.gdc_sync.is_null() {
                unsafe {
                    let non_null = NonNull::new_unchecked(resources.gdc_sync);
                    VolatilePtr::new(non_null)
                }
            } else {
                unsafe {
                    let send_mem_ptr = resources.send_mem.as_ptr_host();
                    let send_head_ptr = addr_of_mut!((*send_mem_ptr).head);
                    let non_null = NonNull::new_unchecked(send_head_ptr);
                    VolatilePtr::new(non_null)
                }
            };
            send_head.write(op.base + op.done);
            if !resources.gdc_sync.is_null() {
                wc_store_fence();
            }
            log::trace!(
                "[DONE] net_agent_send_progress(): *send_head={}, base={}, transmitted={}, done={}, num_steps={}",
                send_head.read(),
                op.base,
                op.transmitted,
                op.done,
                num_steps,
            );
            op.idle = false;
            if op.done == num_steps {
                log::debug!("Send Completed");
                resources.step = op.base + num_steps;
                resources.qos_round += 1;
                op.state = TransportOpState::Completed;
                return Ok(());
            }
        }
    }
    Ok(())
}

pub fn net_agent_recv_progress(
    resources: &mut AgentRecvResources,
    op: &mut TransportOp,
) -> Result<()> {
    if op.state == TransportOpState::Init {
        op.base = resources.step.div_ceil(op.chunk_steps as u64) * (op.chunk_steps as u64);
        op.posted = 0;
        op.received = 0;
        op.transmitted = 0;
        op.done = 0;
        op.state = TransportOpState::InProgress;
    }
    op.idle = true;
    if op.state != TransportOpState::InProgress {
        return Ok(());
    }
    // log::trace!("Recv op: {:?}", op);
    log::trace!(
        "#{} net_agent_recv_progress():  base={}, posted={}, transmitted={}, done={} num_step={}",
        op.debug_id,
        op.base,
        op.posted,
        op.transmitted,
        op.done,
        op.num_steps
    );

    let provider = resources.provider;
    let recv_comm = resources.recv_comm.as_mut();
    let proto = op.protocol;
    let num_steps = op.num_steps as u64;
    let max_depth = NUM_BUFFER_SLOTS as u64;
    if op.posted < num_steps && op.posted < op.done + max_depth {
        let step_size = resources.buffer_sizes[proto as usize] / NUM_BUFFER_SLOTS;
        let local_buffer = resources.map.get_buffer_cpu_ptr(proto).unwrap();
        let buffer_slot = (op.base + op.posted) as usize % NUM_BUFFER_SLOTS;
        let offset = buffer_slot * step_size;
        let ptr = unsafe { local_buffer.as_ptr().byte_add(offset) };
        let mhandle = resources.mr_handles[proto as usize]
            .as_ref()
            .unwrap()
            .as_ref();

        let ptrs = [ptr];
        let mhandles = [mhandle];
        let sizes = [step_size * op.slice_steps as usize];
        let tags = [resources.remote_rank as u32];

        let request_id =
            provider.initiate_recv(recv_comm, &ptrs[..], &sizes[..], &tags[..], &mhandles[..])?;
        if let Some(request_id) = request_id {
            op.requests_id[op.posted as usize % NUM_BUFFER_SLOTS] = Some(request_id.0);
            op.posted += op.slice_steps as u64;
            op.idle = false;
        }
    }
    if !op.idle {
        return Ok(());
    }

    if op.posted > op.received {
        let step = op.received;
        let request_id = NetRequestId(op.requests_id[step as usize % NUM_BUFFER_SLOTS].unwrap());
        let mut size = 0;
        let done = provider.test(
            resources.recv_comm.as_mut(),
            request_id,
            CommType::Recv,
            Some(std::slice::from_mut(&mut size)),
        )?;
        if done {
            op.received += op.slice_steps as u64;
            let need_flush =
                step < op.num_steps as u64 && resources.use_gdr && resources.need_flush;
            op.requests_id[step as usize % NUM_BUFFER_SLOTS] = None;
            if size > 0 && proto == Protocol::Simple && need_flush {
                if !resources.gdc_flush.is_null() {
                    unsafe {
                        std::arch::asm!(
                            "mov ({0}), %eax",
                            in(reg) resources.gdc_flush,
                            out("eax") _,
                            options(readonly, nostack, att_syntax)
                        );
                    }
                } else {
                    if step < op.num_steps as u64 {
                        let step_size = resources.buffer_sizes[proto as usize] / NUM_BUFFER_SLOTS;
                        let local_buffer = resources.map.get_buffer_cpu_ptr(proto).unwrap();
                        let buffer_slot = (op.base + op.posted) as usize % NUM_BUFFER_SLOTS;
                        let offset = buffer_slot * step_size;
                        let ptr = unsafe { local_buffer.as_ptr().byte_add(offset) };
                        let mhandle = resources.mr_handles[proto as usize]
                            .as_ref()
                            .unwrap()
                            .as_ref();
                        let request_id = provider.initiate_flush(
                            resources.recv_comm.as_mut(),
                            std::slice::from_ref(&ptr),
                            std::slice::from_ref(&size),
                            std::slice::from_ref(&mhandle),
                        )?;
                        if let Some(request_id) = request_id {
                            op.requests_id[step as usize % NUM_BUFFER_SLOTS] = Some(request_id.0);
                        }
                    }
                }
            }
        }
    }
    if !op.idle {
        return Ok(());
    }

    if op.received > op.transmitted {
        let step = op.transmitted;
        let request_id = op.requests_id[step as usize % NUM_BUFFER_SLOTS];
        let done = if let Some(request_id) = request_id {
            provider.test(
                resources.recv_comm.as_mut(),
                NetRequestId(request_id),
                CommType::Recv,
                None,
            )?
        } else {
            true
        };
        if done {
            op.transmitted += op.slice_steps as u64;
            if step < num_steps {
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                let recv_tail = if !resources.gdc_sync.is_null() {
                    unsafe {
                        let non_null = NonNull::new_unchecked(resources.gdc_sync);
                        VolatilePtr::new(non_null)
                    }
                } else {
                    unsafe {
                        let recv_mem_ptr = resources.recv_mem.as_ptr_host();
                        let recv_tail_ptr = addr_of_mut!((*recv_mem_ptr).tail);
                        let non_null = NonNull::new_unchecked(recv_tail_ptr);
                        VolatilePtr::new(non_null)
                    }
                };
                recv_tail.write(op.base + op.transmitted);
                if !resources.gdc_sync.is_null() {
                    wc_store_fence();
                }
                log::trace!(
                    "[DONE] net_agent_recv_progress(): *recv_tail={}, base={}, transmitted={}, done={}, step={}, num_steps={}",
                    recv_tail.read(),
                    op.base,
                    op.transmitted,
                    op.done,
                    step,
                    num_steps,
                );
            }
            op.idle = false;
        }
    }
    if !op.idle {
        return Ok(());
    }

    if op.transmitted > op.done {
        let send_head = unsafe {
            let send_mem_ptr = resources.send_mem.as_ptr_host();
            let send_head_ptr = addr_of_mut!((*send_mem_ptr).head);
            let non_null = NonNull::new_unchecked(send_head_ptr);
            VolatilePtr::new(non_null)
        };
        let done = send_head.read();
        while done > op.base + op.done &&
            // LL and LL128 can acknowledge 0-bytes send before they even happen. Don't go past what we transmitted.
            op.transmitted > op.done
        {
            op.done += op.slice_steps as u64;
            op.idle = false;
            if op.done == num_steps {
                log::debug!("Recv Completed");
                resources.step = op.base + num_steps;
                op.state = TransportOpState::Completed;
                return Ok(());
            }
        }
    }
    Ok(())
}
