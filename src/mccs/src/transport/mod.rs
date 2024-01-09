use strum::{EnumCount, EnumIter};

use collectives_sys::MCCS_BUFFER_SLOTS;
use collectives_sys::{MCCS_NUM_PROTOCOLS, MCCS_PROTO_SIMPLE};

pub mod catalog;
pub mod channel;
pub mod delegator;
pub mod engine;
pub mod message;
pub mod meta;
pub mod net;
pub mod op;
pub mod queue;
pub mod setup;
pub mod shm;
pub mod task;
pub mod transporter;

use net::transporter::NetTransport;
use shm::transporter::ShmTransporter;
use transporter::Transporter;

pub static SHM_TRANSPORTER: ShmTransporter = ShmTransporter;
pub static NET_TRANSPORTER: NetTransport = NetTransport;
pub static ALL_TRANSPORTERS: &[&'static dyn Transporter] = &[&SHM_TRANSPORTER, &NET_TRANSPORTER];

pub const NUM_BUFFER_SLOTS: usize = MCCS_BUFFER_SLOTS as _;
pub const NUM_PROTOCOLS: usize = MCCS_NUM_PROTOCOLS as _;

#[derive(Debug, PartialEq, Eq, Clone, Copy, EnumIter, EnumCount)]
#[repr(usize)]
pub enum Protocol {
    Simple = MCCS_PROTO_SIMPLE as _,
}

static_assertions::const_assert_eq!(std::mem::variant_count::<Protocol>(), NUM_PROTOCOLS);

pub const DEFAULT_BUFFER_SIZE: usize = 1 << 22;
