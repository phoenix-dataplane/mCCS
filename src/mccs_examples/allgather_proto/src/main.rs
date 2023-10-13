use cuda_runtime_sys::{cudaError, cudaMemcpyKind};
use cuda_runtime_sys::{cudaMemcpy, cudaSetDevice};

const BUFFER_SIZE: usize = 8192;

fn main() {
    let comm_id = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(42);

    let handle = std::thread::spawn(move || {
        // device 1
        unsafe {
            let err = cudaSetDevice(1);
            if err != cudaError::cudaSuccess {
                panic!("cudaSetDevice");
            }
        }
        let dev_ptr = libmccs::cuda_malloc(1, BUFFER_SIZE).unwrap();
        let mut buf = vec![0i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()];
        buf.extend(vec![2042i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]);
        let err = unsafe {
            cudaMemcpy(
                dev_ptr.ptr,
                buf.as_ptr() as *const _,
                BUFFER_SIZE,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        };
        if err != cudaError::cudaSuccess {
            panic!("cudaMemcpy failed");
        }
        println!(
            "rank 1 - pre : buf[0]={}, buf[{}]={}",
            buf[0],
            BUFFER_SIZE / 2 / std::mem::size_of::<i32>(),
            buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]
        );

        let comm = libmccs::init_communicator_rank(comm_id, 1, 2, 1).unwrap();
        libmccs::all_gather(
            comm,
            dev_ptr.add(BUFFER_SIZE / 2).unwrap(),
            dev_ptr,
            BUFFER_SIZE / 2,
        )
        .unwrap();

        let mut buf = vec![0; BUFFER_SIZE];
        unsafe {
            let err = cudaMemcpy(
                buf.as_mut_ptr() as *mut _,
                dev_ptr.ptr,
                BUFFER_SIZE,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            if err != cudaError::cudaSuccess {
                panic!("cudaMemcpy failed");
            }
        };
        assert_eq!(buf[0], 1883);
        assert_eq!(buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()], 2042);
        println!(
            "rank 1 - post : buf[0]={}, buf[{}]={}",
            buf[0],
            BUFFER_SIZE / 2 / std::mem::size_of::<i32>(),
            buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]
        )
    });

    // device 0
    unsafe {
        let err = cudaSetDevice(0);
        if err != cudaError::cudaSuccess {
            panic!("cudaSetDevice");
        }
    }
    let dev_ptr = libmccs::cuda_malloc(0, BUFFER_SIZE).unwrap();
    let mut buf = vec![1883i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()];
    buf.extend(vec![0i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]);
    let err = unsafe {
        cudaMemcpy(
            dev_ptr.ptr,
            buf.as_ptr() as *const _,
            BUFFER_SIZE,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
    };
    if err != cudaError::cudaSuccess {
        panic!("cudaMemcpy failed");
    }
    println!(
        "rank 0 - pre : buf[0]={}, buf[{}]={}",
        buf[0],
        BUFFER_SIZE / 2 / std::mem::size_of::<i32>(),
        buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]
    );
    let comm = libmccs::init_communicator_rank(comm_id, 0, 2, 0).unwrap();
    libmccs::all_gather(comm, dev_ptr, dev_ptr, BUFFER_SIZE / 2).unwrap();
    let mut buf = vec![0; BUFFER_SIZE];
    unsafe {
        let err = cudaMemcpy(
            buf.as_mut_ptr() as *mut _,
            dev_ptr.ptr,
            BUFFER_SIZE,
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );
        if err != cudaError::cudaSuccess {
            panic!("cudaMemcpy failed");
        }
    };
    assert_eq!(buf[0], 1883);
    assert_eq!(buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()], 2042);
    println!(
        "rank 0 - post : buf[0]={}, buf[{}]={}",
        buf[0],
        BUFFER_SIZE / 2 / std::mem::size_of::<i32>(),
        buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]
    );

    handle.join().unwrap();
}
