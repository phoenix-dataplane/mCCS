use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicPtr};

use super::MAX_BUFFER_SLOTS;

pub struct ConnectionInfo {
   pub bufs: Vec<NonNull<u8>>,
   #[allow(unused)]
   pub head: *mut AtomicU64,
   #[allow(unused)]
   pub tail: *mut AtomicU64,

   pub _direct: bool,
   pub _shared: bool,
   pub _ptr_exchange: *mut AtomicPtr<c_void>,
   pub _red_op_arg_exchange: *mut AtomicU64,

   pub _slots_sizes: *mut [AtomicU32; MAX_BUFFER_SLOTS],
   pub _slots_offsets: *mut [AtomicU32; MAX_BUFFER_SLOTS],

   pub _step: u64,
   pub _ll_last_cleaning: u64,
}

// TBD
unsafe impl Send for ConnectionInfo {} 
unsafe impl Sync for ConnectionInfo {} 

pub struct TransportConnector {
    pub info: ConnectionInfo,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectorIdentifier {
    pub communicator_id: u32,
    pub sender_rank: usize,
    pub receiver_rank: usize,
    pub channel: u32,
}
