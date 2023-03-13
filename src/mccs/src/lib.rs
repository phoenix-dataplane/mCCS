#![feature(peer_credentials_unix_socket)]
#![feature(drain_filter)]
#![feature(strict_provenance)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(int_roundings)]


pub mod config;
pub mod control;
pub mod daemon;
pub mod cuda;
pub mod transport;
pub mod comm;
pub mod proxy;
pub mod pattern;
pub mod registry;
pub mod message;
pub mod utils;