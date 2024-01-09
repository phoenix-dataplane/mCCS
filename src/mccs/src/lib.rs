#![feature(peer_credentials_unix_socket)]
#![feature(drain_filter)]
#![feature(strict_provenance)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(int_roundings)]
#![feature(variant_count)]
#![feature(is_some_and)]
#![feature(atomic_from_mut)]
// todo: temporary
#![allow(dead_code)]
#![allow(unused)]
#![feature(pointer_byte_offsets)]
#![feature(slice_ptr_get)]

pub mod bootstrap;
#[allow(dead_code)]
pub mod comm;
pub mod config;
pub mod control;
pub mod cuda;
pub mod daemon;
pub mod exchange;
pub mod message;
pub mod pattern;
pub mod proxy;
pub mod registry;
pub mod transport;
pub mod utils;
