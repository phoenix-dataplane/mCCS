pub mod duplex_chan;
pub mod gdr;
pub mod interfaces;
pub mod pool;
pub mod tcp;

#[macro_export]
macro_rules! cuda_warning {
    ($cuda_op:expr) => {{
        let e = $cuda_op;
        if e != cuda_runtime_sys::cudaError::cudaSuccess {
            log::error!(
                "CUDA runtime failed with {:?} at {}:{}.",
                e,
                file!(),
                line!()
            )
        }
    }};
    ($cuda_op:expr,$ctx:expr) => {{
        let e = $cuda_op;
        if e != cuda_runtime_sys::cudaError::cudaSuccess {
            log::error!(
                "CUDA runtime failed with {:?} at {}:{}. Context={}",
                e,
                file!(),
                line!(),
                $ctx
            )
        }
    }};
}

#[macro_export]
macro_rules! cu_warning {
    ($cu_op:expr) => {{
        let e = $cu_op;
        if e != cuda_driver_sys::CUresult::CUDA_SUCCESS {
            log::error!(
                "CUDA driver failed with {:?} at {}:{}.",
                e,
                file!(),
                line!()
            )
        }
    }};
    ($cu_op:expr,$ctx:expr) => {{
        let e = $cu_op;
        if e != cuda_driver_sys: CUresult::CUDA_SUCCESS {
            log::error!(
                "CUDA driver failed with {:?} at {}:{}. Context={}",
                e,
                file!(),
                line!(),
                $ctx
            )
        }
    }};
}

thread_local!(pub static CU_INIT: () = (|| unsafe {
    cu_warning!(cuda_driver_sys::cuInit(0));
    ()
})());
