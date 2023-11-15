use std::ffi::c_void;
use std::ptr::NonNull;

use crate::cuda::mapped_ptr::DeviceHostPtr;
use crate::transport::{NUM_PROTOCOLS, Protocol};
use crate::transport::meta::{SendBufMeta, RecvBufMeta};

#[derive(PartialEq, Eq)]
#[repr(usize)]
enum MemoryBankType {
    HostMem = 0,
    DeviceMem = 1,
    GdcMem = 2,
}

enum BufferType {
    SendMem,
    RecvMem,
    RingBuffer(Protocol)
}

const NET_MAP_MASK_DEVMEM: u32 = 0x40000000;
const NET_MAP_MASK_USED: u32 = 0x20000000;
const NET_MAP_MASK_OFFSET: u32 = 0x1fffffff;

pub struct BufferBankMem {
    cpu_ptr: *mut c_void,
    gpu_ptr: *mut c_void,
    size: usize,
}

pub struct BufferOffset {
    send_mem: u32,
    recv_mem: u32,
    buffs: [u32; NUM_PROTOCOLS]
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
    mems: [BufferBankMem; std::mem::variant_count::<MemoryBankType>()],
    offsets: BufferOffset,
}

impl BufferMap {
    fn assign_memory(&mut self, buffer_type: BufferType, mem_size: usize, device: bool) {
        let bank_mask = NET_MAP_MASK_USED + (device as u32) * NET_MAP_MASK_DEVMEM;
        if device {
            match buffer_type {
                BufferType::SendMem | BufferType::RecvMem => {
                    panic!("SendMem and RecvMem should reside on dedicated host memory");
                },
                BufferType::RingBuffer(proto) => {
                    self.offsets.buffs[proto as usize] = bank_mask + self.mems[MemoryBankType::DeviceMem as usize].size as u32;
                },
            }
            self.mems[MemoryBankType::DeviceMem as usize].size += mem_size;
        } else {
            match buffer_type {
                BufferType::SendMem => {
                    self.offsets.send_mem = bank_mask + self.mems[MemoryBankType::HostMem as usize].size as u32;
                },
                BufferType::RecvMem => {
                    self.offsets.recv_mem = bank_mask + self.mems[MemoryBankType::HostMem as usize].size as u32;
                }
                BufferType::RingBuffer(proto) => {
                    self.offsets.buffs[proto as usize] = bank_mask + self.mems[MemoryBankType::HostMem as usize].size as u32;
                }
            }
            self.mems[MemoryBankType::HostMem as usize].size += mem_size;
        }
    }

    fn is_buffer_device_memory(&self, proto: Protocol) -> bool {
        (self.offsets.buffs[proto as usize] & NET_MAP_MASK_DEVMEM) != 0
    }

    fn get_buffer_cpu_ptr(&self) -> Option<NonNull<c_void>> {
        
    }

    fn get_send_mem_meta(&self) -> Option<DeviceHostPtr<SendBufMeta>> {
        if self.mems[MemoryBankType::HostMem as usize].cpu_ptr.is_null() {
            None
        } else {
            let offset = (self.offsets.send_mem & NET_MAP_MASK_OFFSET) as usize;
            let cpu_ptr = unsafe {
                self.mems[MemoryBankType::HostMem as usize].cpu_ptr.add(offset)
            } as *mut SendBufMeta;
            let gpu_ptr = unsafe {
                self.mems[MemoryBankType::HostMem as usize].gpu_ptr.add(offset)
            } as *mut SendBufMeta;
            DeviceHostPtr::new(cpu_ptr, gpu_ptr)
        }
    }

    fn get_recv_mem_meta(&self) -> Option<DeviceHostPtr<RecvBufMeta>> {
        if self.mems[MemoryBankType::HostMem as usize].cpu_ptr.is_null() {
            None
        } else {
            let offset = (self.offsets.recv_mem & NET_MAP_MASK_OFFSET) as usize;
            let cpu_ptr = unsafe {
                self.mems[MemoryBankType::HostMem as usize].cpu_ptr.add(offset)
            } as *mut RecvBufMeta;
            let gpu_ptr = unsafe {
                self.mems[MemoryBankType::HostMem as usize].gpu_ptr.add(offset)
            } as *mut RecvBufMeta;
            DeviceHostPtr::new(cpu_ptr, gpu_ptr)
        }
    }
}