use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use num_enum::TryFromPrimitive;

use crate::cuda::alloc::{DeviceAlloc, DeviceHostMapped};
use crate::cuda::mapped_ptr::DeviceHostPtr;
use crate::cuda::ptr::DeviceNonNull;
use crate::transport::meta::{RecvBufMeta, SendBufMeta};
use crate::transport::{Protocol, NUM_PROTOCOLS};

#[derive(PartialEq, Eq, TryFromPrimitive)]
#[repr(u8)]
pub enum MemoryBankType {
    HostMem = 0,
    DeviceMem = 1,
    GdcMem = 2,
}

pub enum BufferType {
    SendMem,
    RecvMem,
    RingBuffer(Protocol),
}

const NET_MAP_MASK_DEVMEM: u32 = 0x40000000;
const NET_MAP_MASK_USED: u32 = 0x20000000;
const NET_MAP_MASK_OFFSET: u32 = 0x1fffffff;

pub enum BufferAlloc {
    Host(DeviceHostMapped<u8>),
    Device(DeviceAlloc<u8>),
}

#[derive(Clone)]
pub struct BufferBankMem {
    cpu_ptr: *mut c_void,
    gpu_ptr: *mut c_void,
    size: usize,
    alloc: Option<Arc<BufferAlloc>>,
}

pub struct BufferOffset {
    send_mem: u32,
    recv_mem: u32,
    buffers: [u32; NUM_PROTOCOLS],
}

/*
struct connectMap {
  int sameProcess;
  int shared;
  int cudaDev;
  // First 3 bits of offsets determine the mem bank. 001 is host mem, 011 is dev mem, 101 is shared host mem and 111 is shared dev mem.
  struct connectMapMem mems[NCCL_NET_MAP_MEMS];
  // Offsets. 3 MSBs indicate mem bank, 111 indicates NULL.
  struct {
    uint32_t sendMem;
    uint32_t recvMem;
    uint32_t buffs[NCCL_NUM_PROTOCOLS];
  } offsets;
};
*/

pub struct BufferMap {
    pub(crate) mems: [BufferBankMem; std::mem::variant_count::<MemoryBankType>()],
    pub(crate) offsets: BufferOffset,
}

impl BufferMap {
    pub(crate) fn new() -> Self {
        let mems = std::array::from_fn(|_| BufferBankMem {
            cpu_ptr: std::ptr::null_mut(),
            gpu_ptr: std::ptr::null_mut(),
            size: 0,
            alloc: None,
        });
        let offsets = BufferOffset {
            send_mem: 0,
            recv_mem: 0,
            buffers: [0; NUM_PROTOCOLS],
        };
        BufferMap { mems, offsets }
    }
}

impl BufferMap {
    fn assign_memory(&mut self, buffer_type: BufferType, mem_size: usize, device: bool) {
        let bank_mask = NET_MAP_MASK_USED + (device as u32) * NET_MAP_MASK_DEVMEM;
        if device {
            match buffer_type {
                BufferType::SendMem | BufferType::RecvMem => {
                    panic!("SendMem and RecvMem should reside on dedicated host memory");
                }
                BufferType::RingBuffer(proto) => {
                    self.offsets.buffers[proto as usize] =
                        bank_mask + self.mems[MemoryBankType::DeviceMem as usize].size as u32;
                }
            }
            self.mems[MemoryBankType::DeviceMem as usize].size += mem_size;
        } else {
            match buffer_type {
                BufferType::SendMem => {
                    self.offsets.send_mem =
                        bank_mask + self.mems[MemoryBankType::HostMem as usize].size as u32;
                }
                BufferType::RecvMem => {
                    self.offsets.recv_mem =
                        bank_mask + self.mems[MemoryBankType::HostMem as usize].size as u32;
                }
                BufferType::RingBuffer(proto) => {
                    self.offsets.buffers[proto as usize] =
                        bank_mask + self.mems[MemoryBankType::HostMem as usize].size as u32;
                }
            }
            self.mems[MemoryBankType::HostMem as usize].size += mem_size;
        }
    }

    fn is_buffer_device_memory(&self, proto: Protocol) -> bool {
        (self.offsets.buffers[proto as usize] & NET_MAP_MASK_DEVMEM) != 0
    }

    fn get_buffer_bank(&self, proto: Protocol) -> MemoryBankType {
        ((self.offsets.buffers[proto as usize] >> 30) as u8)
            .try_into()
            .unwrap()
    }

    fn is_buffer_null(&self, proto: Protocol) -> bool {
        (self.offsets.buffers[proto as usize] >> 29) == 0
    }

    fn get_buffer_cpu_ptr(&self, proto: Protocol) -> Option<NonNull<c_void>> {
        if self.is_buffer_null(proto) {
            None
        } else {
            let offset = (self.offsets.buffers[proto as usize] & NET_MAP_MASK_OFFSET) as usize;
            let cpu_ptr = unsafe {
                let base_ptr = self.mems[self.get_buffer_bank(proto) as usize].cpu_ptr;
                if base_ptr.is_null() {
                    return None;
                }
                base_ptr.add(offset)
            };
            Some(NonNull::new(cpu_ptr).unwrap())
        }
    }

    fn get_buffer_gpu_ptr(&self, proto: Protocol) -> Option<DeviceNonNull<c_void>> {
        if self.is_buffer_null(proto) {
            None
        } else {
            let offset = (self.offsets.buffers[proto as usize] & NET_MAP_MASK_OFFSET) as usize;
            let gpu_ptr = unsafe {
                let base_ptr = self.mems[self.get_buffer_bank(proto) as usize].gpu_ptr;
                if base_ptr.is_null() {
                    return None;
                }
                base_ptr.add(offset)
            };
            Some(DeviceNonNull::new(gpu_ptr).unwrap())
        }
    }

    fn get_send_mem_meta(&self) -> Option<DeviceHostPtr<SendBufMeta>> {
        if (self.mems[MemoryBankType::HostMem as usize]
            .cpu_ptr
            .is_null())
            || (self.offsets.send_mem >> 29 == 0)
        {
            None
        } else {
            let offset = (self.offsets.send_mem & NET_MAP_MASK_OFFSET) as usize;
            let cpu_ptr = unsafe {
                self.mems[MemoryBankType::HostMem as usize]
                    .cpu_ptr
                    .add(offset)
            } as *mut SendBufMeta;
            let gpu_ptr = unsafe {
                self.mems[MemoryBankType::HostMem as usize]
                    .gpu_ptr
                    .add(offset)
            } as *mut SendBufMeta;
            DeviceHostPtr::new(cpu_ptr, gpu_ptr)
        }
    }

    fn get_recv_mem_meta(&self) -> Option<DeviceHostPtr<RecvBufMeta>> {
        if (self.mems[MemoryBankType::HostMem as usize]
            .cpu_ptr
            .is_null())
            || (self.offsets.recv_mem >> 29 == 0)
        {
            None
        } else {
            let offset = (self.offsets.recv_mem & NET_MAP_MASK_OFFSET) as usize;
            let cpu_ptr = unsafe {
                self.mems[MemoryBankType::HostMem as usize]
                    .cpu_ptr
                    .add(offset)
            } as *mut RecvBufMeta;
            let gpu_ptr = unsafe {
                self.mems[MemoryBankType::HostMem as usize]
                    .gpu_ptr
                    .add(offset)
            } as *mut RecvBufMeta;
            DeviceHostPtr::new(cpu_ptr, gpu_ptr)
        }
    }
}