pub mod hmem;
pub mod buffer;
pub mod channel;
pub mod transporter;
pub mod engine;
pub mod message;
pub mod task;
pub mod op;
pub mod queue;
pub mod delegator;

pub const MAX_BUFFER_SLOTS: usize = 8;
// TBD
pub const NUM_PROTOCOLS: usize = 1;