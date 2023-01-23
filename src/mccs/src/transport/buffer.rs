use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicBool, AtomicPtr};
use std::alloc::{Layout, System, GlobalAlloc};
use std::ops::{Deref, DerefMut};

use super::MAX_BUFFER_SLOTS;
const CACHE_LINE_SIZE: usize = 128;
const ATOMIC_U32_INIT: AtomicU32 = AtomicU32::new(0);
const ATOMIC_U64_INIT: AtomicU64 = AtomicU64::new(0);


#[repr(C, align(4096))]
pub struct SendBufMeta {
    pub head: AtomicU64,
    _pad1: [u8; CACHE_LINE_SIZE - std::mem::size_of::<u64>()],
    _ptr_exchange: AtomicPtr<c_void>,
    _reduce_op_arg_exchange: [AtomicU64; 2],
    _pad2: [u8; CACHE_LINE_SIZE - std::mem::size_of::<*mut c_void>() - 2 * std::mem::size_of::<u64>()],
    _slots_offsets: [AtomicU32; MAX_BUFFER_SLOTS],
}

impl SendBufMeta {
    pub fn new() -> Self {
        SendBufMeta {
            head: ATOMIC_U64_INIT,
            _pad1: [0; CACHE_LINE_SIZE - std::mem::size_of::<u64>()],
            _ptr_exchange: AtomicPtr::new(std::ptr::null_mut()),
            _reduce_op_arg_exchange: [ATOMIC_U64_INIT; 2],
            _pad2: [0; CACHE_LINE_SIZE - std::mem::size_of::<*mut c_void>() - 2 * std::mem::size_of::<u64>()],
            _slots_offsets: [ATOMIC_U32_INIT; MAX_BUFFER_SLOTS],
        }
    }
}

#[repr(C, align(4096))]
pub struct RecvBufMeta {
    pub tail: AtomicU64,
    _pad1: [u8; CACHE_LINE_SIZE - std::mem::size_of::<u64>()],
    _slots_sizes: [AtomicU32; MAX_BUFFER_SLOTS],
    _slots_offsets: [AtomicU32; MAX_BUFFER_SLOTS],
    _flush: AtomicBool,
}

impl RecvBufMeta {
    pub fn new() -> Self {
        RecvBufMeta {
            tail: ATOMIC_U64_INIT,
            _pad1: [0; CACHE_LINE_SIZE - std::mem::size_of::<u64>()],
            _slots_sizes: [ATOMIC_U32_INIT; MAX_BUFFER_SLOTS],
            _slots_offsets: [ATOMIC_U32_INIT; MAX_BUFFER_SLOTS],
            _flush: AtomicBool::new(false),
        }
    }
}

pub struct TransportBuffer<T> {
    ptr: *mut T,
    size: usize,
    align: usize,
}

impl<T> TransportBuffer<T> {
    pub fn new(meta: T, size: usize, align: usize) -> TransportBuffer<T> {
        let layout = Layout::from_size_align(size, align).unwrap();
        let ptr = unsafe { System.alloc(layout) } as *mut T;
        assert_eq!(ptr.align_offset(std::mem::align_of::<T>()), 0);
        unsafe { *ptr = meta };
        TransportBuffer {
            ptr, 
            size,
            align,
        }
    }

    #[inline]
    pub unsafe fn get_meta(&self) -> &T {
        &*self.ptr
    }

    #[inline]
    pub unsafe fn get_meta_mut(&self) -> &mut T {
        &mut *self.ptr
    }
    
    #[inline]
    pub fn buf_ptr(&self) -> *const u8 {
        unsafe { self.ptr.add(1) as *const u8 }
    }

    #[inline]
    pub fn buf_mut_ptr(&self) -> *mut u8 {
        unsafe { self.ptr.add(1) as *mut u8 }
    }

    #[inline]
    pub fn buf_slice(&self) -> *const [u8] {
        unsafe { 
            let buf_ptr = self.ptr.add(1) as *const _;
            let buf_size = self.size - std::mem::size_of::<T>();
            std::ptr::slice_from_raw_parts(buf_ptr, buf_size)
        }
    }

    #[inline]
    pub fn buf_mut_slice(&self) -> *mut [u8] {
        unsafe { 
            let buf_ptr = self.ptr.add(1) as *mut _;
            let buf_size = self.size - std::mem::size_of::<T>();
            std::ptr::slice_from_raw_parts_mut(buf_ptr, buf_size)
        }
    }

    #[inline]
    pub fn buf_size(&self) -> usize {
        self.size - std::mem::size_of::<T>()
    }
}

impl<T> Deref for TransportBuffer<T> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { 
            let buf_ptr = self.ptr.add(1) as *const _;
            let buf_size = self.size - std::mem::size_of::<T>();
            std::slice::from_raw_parts(buf_ptr, buf_size)
        }
    }
}

impl<T> DerefMut for TransportBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            let buf_ptr = self.ptr.add(1) as *mut _;
            let buf_size = self.size - std::mem::size_of::<T>();
            std::slice::from_raw_parts_mut(buf_ptr, buf_size)
        }
    }
}

impl<T> AsRef<[u8]> for TransportBuffer<T> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &*self
    }
}

impl<T> Drop for TransportBuffer<T> {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, self.align).unwrap();
        unsafe {
            std::ptr::drop_in_place(self.ptr);
            System.dealloc(self.ptr as *mut _, layout);
        }
    }
}

unsafe impl<T: Send + Sync> Send for TransportBuffer<T> { }
unsafe impl<T: Send + Sync> Sync for TransportBuffer<T> { }