pub type WorkRequestSlot = [u8; 128];
pub type CompletionSlot = [u8; 64];

use serde::{Deserialize, Serialize};

use super::command::{AllGather, AllReduce};

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WorkRequest {
    AllReduce(AllReduce),
    AllGather(AllGather),
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WorkCompletion {
    AllReduce,
    AllGather,
}

mod sa {
    use super::*;
    use static_assertions::const_assert;
    use std::mem::size_of;
    const_assert!(size_of::<WorkRequest>() <= size_of::<WorkRequestSlot>());
    const_assert!(size_of::<WorkCompletion>() <= size_of::<CompletionSlot>());
}
