use std::ffi::c_void;
use std::io::Read;
use std::mem::size_of;
use std::net::TcpStream;

use cuda_runtime_sys::{
    cudaError, cudaIpcMemHandle_t, cudaIpcMemLazyEnablePeerAccess, cudaIpcOpenMemHandle,
    cudaMemcpy, cudaMemcpyKind,
};

const BUFFER_SIZE: usize = 1 * 1024 * 1024;

fn main() {
    let mut buf = vec![0i32; BUFFER_SIZE / size_of::<i32>()];
    buf.shrink_to_fit();
    assert_eq!(buf.capacity(), BUFFER_SIZE / size_of::<i32>());

    let mut handle = cudaIpcMemHandle_t::default();
    {
        let mut stream = TcpStream::connect("localhost:2042").unwrap();
        stream.set_nonblocking(false).unwrap();
        stream.set_nodelay(true).unwrap();
        stream
            .read_exact(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut handle as *mut _ as *mut u8,
                    size_of::<cudaIpcMemHandle_t>(),
                )
            })
            .unwrap();
    }
    let mut dev_ptr: *mut c_void = std::ptr::null_mut();
    let err = unsafe {
        cudaIpcOpenMemHandle(
            &mut dev_ptr as *mut _,
            handle,
            cudaIpcMemLazyEnablePeerAccess,
        )
    };
    if err != cudaError::cudaSuccess {
        panic!("cudaIpcOpenMemHandle failed")
    }
    let err = unsafe {
        cudaMemcpy(
            buf.as_mut_ptr() as *mut _,
            dev_ptr,
            BUFFER_SIZE,
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        )
    };
    if err != cudaError::cudaSuccess {
        panic!("cudaMemcpy failed")
    }

    for x in buf.iter() {
        assert_eq!(*x, 42, "CUDA IPC content mismatch");
    }
    println!("buf={}", buf[0]);
}
