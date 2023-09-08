use self::shm::transporter::ShmTransporter;

pub mod catalog;
pub mod channel;
pub mod delegator;
pub mod engine;
pub mod message;
pub mod meta;
pub mod op;
pub mod queue;
pub mod shm;
pub mod task;
pub mod transporter;

pub static SHM_TRANSPORTER: ShmTransporter = ShmTransporter;

pub const NUM_BUFFER_SLOTS: usize = 8;
pub const NUM_PROTOCOLS: usize = 1;

pub const PROTOCOL_SIMPLE: usize = 0;

pub const DEFAULT_BUFFER_SIZE: usize = 1 << 22;
