use std::ffi::c_void;

use super::NUM_BUFFER_SLOTS;
const CACHE_LINE_SIZE: usize = 128;

#[repr(C, align(4096))]
pub struct SendBufMeta {
    pub head: u64,
    _pad1: [u8; CACHE_LINE_SIZE - std::mem::size_of::<u64>()],
    _ptr_exchange: *mut c_void,
    _reduce_op_arg_exchange: [u64; 2],
    _pad2:
        [u8; CACHE_LINE_SIZE - std::mem::size_of::<*mut c_void>() - 2 * std::mem::size_of::<u64>()],
    _slots_offsets: [i32; NUM_BUFFER_SLOTS],
}

static_assertions::const_assert_eq!(std::mem::size_of::<SendBufMeta>(), 4096);

impl SendBufMeta {
    pub fn new() -> Self {
        SendBufMeta {
            head: 0,
            _pad1: [0; CACHE_LINE_SIZE - std::mem::size_of::<u64>()],
            _ptr_exchange: std::ptr::null_mut(),
            _reduce_op_arg_exchange: [0; 2],
            _pad2: [0; CACHE_LINE_SIZE
                - std::mem::size_of::<*mut c_void>()
                - 2 * std::mem::size_of::<u64>()],
            _slots_offsets: [0; NUM_BUFFER_SLOTS],
        }
    }
}

impl Default for SendBufMeta {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C, align(4096))]
pub struct RecvBufMeta {
    pub tail: u64,
    _pad1: [u8; CACHE_LINE_SIZE - std::mem::size_of::<u64>()],
    pub slots_sizes: [i32; NUM_BUFFER_SLOTS],
    _slots_offsets: [i32; NUM_BUFFER_SLOTS],
    _flush: i32,
}

static_assertions::const_assert_eq!(std::mem::size_of::<RecvBufMeta>(), 4096);

impl Default for RecvBufMeta {
    fn default() -> Self {
        Self::new()
    }
}

impl RecvBufMeta {
    pub fn new() -> Self {
        RecvBufMeta {
            tail: 0,
            _pad1: [0; CACHE_LINE_SIZE - std::mem::size_of::<u64>()],
            slots_sizes: [0; NUM_BUFFER_SLOTS],
            _slots_offsets: [0; NUM_BUFFER_SLOTS],
            _flush: 0,
        }
    }
}
