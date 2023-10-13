use cuda_runtime_sys::cudaMemcpy;
use cuda_runtime_sys::{cudaError, cudaMemcpyKind};

use libmccs::memory::cuda_malloc;

const BUFFER_SIZE: usize = 1024 * 1024;

fn main() {
    let dev_ptr = cuda_malloc(0, BUFFER_SIZE).unwrap();
    let buf = vec![42i32; BUFFER_SIZE / std::mem::size_of::<i32>()];
    let err = unsafe {
        cudaMemcpy(
            dev_ptr.ptr,
            buf.as_ptr() as *const _,
            BUFFER_SIZE,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
    };
    if err != cudaError::cudaSuccess {
        panic!("cudaMemcpy failed")
    }

    println!("cudaMemcpy success");
}
