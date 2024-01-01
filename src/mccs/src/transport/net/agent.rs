use std::ffi::c_void;
use std::mem::MaybeUninit;

use strum::IntoEnumIterator;

use cuda_driver_sys::CUmemRangeHandleType;
use cuda_driver_sys::cuMemGetHandleForAddressRange;

use crate::cuda::alloc::{DeviceAlloc, DeviceHostMapped};
use crate::transport::NUM_BUFFER_SLOTS;
use crate::transport::op::TransportOpState;
use crate::utils::gdr::{GDR_HANDLE, GdrMappedMem};
use crate::transport::Protocol;
use crate::transport::meta::{SendBufMeta, RecvBufMeta};
use crate::transport::op::TransportOp;

use super::AgentError;
use super::buffer::{BufferMap, BufferType, MemoryBankType, MemoryBankAlloc};
use super::resources::{AgentRecvConnectRequest, AgentRecvSetup, AgentRecvResources};
use super::provider::{MrType, MemoryRegion};


type Result<T> = std::result::Result<T, AgentError>;

pub async fn net_agent_recv_connect(
    message: AgentRecvConnectRequest, 
    setup_resources: AgentRecvSetup
) -> Result<(BufferMap, AgentRecvResources)> {
    let provider = setup_resources.provider;
    let recv_comm = provider.accept(setup_resources.listen_comm).await?;
    let mut recv_comm = Box::into_pin(recv_comm);
    let mut map = BufferMap::new();

    let buffer_sizes = setup_resources.buffer_sizes;
    for proto in Protocol::iter() {
        let buffer_type = BufferType::RingBuffer(proto);
        map.assign_buffer_memory(buffer_type, buffer_sizes[proto as usize], setup_resources.use_gdr)
    }

    map.assign_buffer_memory(BufferType::SendMem, std::mem::size_of::<SendBufMeta>(), false);
    map.assign_buffer_memory(BufferType::RecvMem, std::mem::size_of::<RecvBufMeta>(), false);

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

    let mut gdc_sync = std::ptr::null_mut::<u64>();
    if !GDR_HANDLE.0.is_null() {
        let gdc_mem = GdrMappedMem::new(1);
        gdc_sync = gdc_mem.get_cpu_ptr();
        let alloc = MemoryBankAlloc::GdcMem(gdc_mem);
        map.register_bank_alloc(alloc);
    }
    
    let mut mr_handles = MaybeUninit::uninit_array();
    for proto in Protocol::iter() {
        if let Some(buffer_ptr) = map.get_buffer_cpu_ptr(proto) {
            let buffer_size = buffer_sizes[proto as usize];
            let mr_type = match map.get_buffer_bank(proto) {
                MemoryBankType::HostMem => MrType::Host,
                MemoryBankType::DeviceMem => MrType::Device,
                MemoryBankType::GdcMem => unreachable!("GDC memory is not used for ring buffer"),
            };
            if mr_type == MrType::Device && setup_resources.use_dma_buf {
                let ptr = buffer_ptr.as_ptr() as *mut c_void;
                unsafe {
                    let mut dmabuf_fd: i32 = 0;
                    cuMemGetHandleForAddressRange(
                        (&mut dmabuf_fd) as *mut i32 as *mut c_void, 
                        ptr.addr() as _,
                        buffer_size,
                        CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                        0
                    );
                    let mr = MemoryRegion {
                        data: ptr,
                        size: buffer_size,
                        mr_type: MrType::Device,
                    };
                    let mhandle = provider.register_mr_dma_buf(
                        recv_comm.as_mut(), 
                        mr,
                        0, 
                        dmabuf_fd
                    ).await?;
                    mr_handles[proto as usize].write(mhandle);
                    nix::unistd::close(dmabuf_fd).map_err(|e|
                        AgentError::BufferRegistration(format!("Failed to close fd for DMA buffer with errno {}", e))
                    )?;
                }
            } else {
            }
        }
    }
    let mr_handles = unsafe { MaybeUninit::array_assume_init(mr_handles) };
    let recv_resources = AgentRecvResources {
        map,
        recv_comm,
        send_mem: todo!(),
        recv_mem: todo!(),
        rank: todo!(),
        local_rank: todo!(),
        remote_rank: todo!(),
        agent_rank: todo!(),
        net_device: todo!(),
        use_gdr: todo!(),
        use_dma_buf: todo!(),
        need_flush: todo!(),
        max_recvs: todo!(),
        gdc_sync,
        gdc_flush: todo!(),
        buffers: todo!(),
        buffer_sizes,
        mr_handles,
        step: todo!(),
        provider,
    };
    todo!()
}

pub fn net_agent_recv_progress(
    resources: &mut AgentRecvResources,
    op: &mut TransportOp
) {
    if op.state == TransportOpState::Init {

    }

    let provider = resources.provider;
    let recv_comm = &mut resources.recv_comm;
    let proto = op.protocol;
    let num_steps = op.num_steps as u64;
    let max_depth = NUM_BUFFER_SLOTS as u64;
    if op.posted < num_steps && op.posted < op.done + max_depth {
        let step_size = resources.buffer_sizes[proto as usize] / NUM_BUFFER_SLOTS;
        let local_bufffer = resources.map.get_buffer_cpu_ptr(proto).unwrap();
        let buffer_slot = (op.base + op.posted) as usize % NUM_BUFFER_SLOTS;
        let offset = buffer_slot * step_size;
        let ptr = unsafe { local_bufffer.as_ptr().byte_add(offset) };
        let mhandle = &resources.mr_handles[proto as usize];

        let ptrs = [ptr];
        let mhandles = [mhandle];
        let sizes = [step_size * op.slice_steps as usize];
        let tags = [resources.remote_rank];

    
    }
    unsafe {
        std::arch::asm!(
            "mov ({0}), %eax",
            in(reg) resources.gdc_flush,
            out("eax") _,
            options(readonly, nostack)
        );
    }
}
