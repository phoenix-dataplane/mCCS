pub struct ShmConfig {
    pub locality: ShmLocality,
    pub use_memcpy_send: bool,
    pub use_memcpy_recv: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShmLocality {
    Sender,
    Receiver,
}