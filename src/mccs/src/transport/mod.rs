use self::shm::transporter::ShmTransporter;

pub mod shm;
pub mod meta;
pub mod channel;
pub mod transporter;
pub mod engine;
pub mod message;
pub mod task;
pub mod op;
pub mod queue;
pub mod delegator;
pub mod catalog;

pub static SHM_TRANSPORTER: ShmTransporter = ShmTransporter;

pub const NUM_BUFFER_SLOTS: usize = 8;
pub const NUM_PROTOCOLS: usize = 1;

pub const PROTOCOL_SIMPLE: usize = 0;

pub const DEFAULT_BUFFER_SIZE: usize = 1 << 22;