use std::mem::size_of;
use std::net::TcpListener;
use std::{ffi::c_void, io::Write};

use cuda_runtime_sys::{
    cudaError, cudaIpcGetMemHandle, cudaIpcMemHandle_t, cudaMalloc, cudaMemcpy, cudaMemcpyKind,
};

const BUFFER_SIZE: usize = 1 * 1024 * 1024;

fn main() {
    let mut dev_ptr: *mut c_void = std::ptr::null_mut();
    let err = unsafe { cudaMalloc(&mut dev_ptr as *mut _, BUFFER_SIZE) };
    if err != cudaError::cudaSuccess {
        panic!("cudaMalloc failed")
    }

    let buf = vec![42i32; BUFFER_SIZE / size_of::<i32>()];
    let err = unsafe {
        cudaMemcpy(
            dev_ptr,
            buf.as_ptr() as *const _,
            BUFFER_SIZE,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
    };
    if err != cudaError::cudaSuccess {
        panic!("cudaMemcpy failed")
    }

    let mut handle = cudaIpcMemHandle_t::default();
    let err = unsafe { cudaIpcGetMemHandle(&mut handle as *mut _, dev_ptr) };
    if err != cudaError::cudaSuccess {
        panic!("cudaIpcGetMemHandle failed")
    }

    let listener = TcpListener::bind("localhost:2042").unwrap();
    match listener.accept() {
        Ok((mut socket, addr)) => {
            socket
                .write_all(unsafe {
                    std::slice::from_raw_parts(
                        &handle as *const _ as *const u8,
                        size_of::<cudaIpcMemHandle_t>(),
                    )
                })
                .unwrap();
            println!("new client: {addr:?}")
        }
        Err(e) => println!("couldn't get client: {e:?}"),
    }
    std::thread::sleep(std::time::Duration::from_secs(2));
}
