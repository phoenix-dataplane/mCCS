pub mod duplex_chan;
pub mod pool;
#[macro_export]
macro_rules! cuda_warning {
    ($cuda_op:expr) => {{
        let e = $cuda_op;
        if e != cuda_runtime_sys::cudaError::cudaSuccess {
            eprintln!("CUDA failed at {}:{}.", file!(), line!())
        }
    }};
    ($cuda_op:expr,$ctx:expr) => {{
        let e = $cuda_op;
        if e != cuda_runtime_sys::cudaError::cudaSuccess {
            eprintln!("CUDA failed at {}:{}. Context={}", file!(), line!(), $ctx)
        }
    }};
}
