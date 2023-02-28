use cuda_runtime_sys::{cudaMemcpy, cudaSetDevice};
use cuda_runtime_sys::{cudaMemcpyKind, cudaError};

use libmccs::collectives::all_gather;
use libmccs::communicator::init_communicator_rank;
use libmccs::memory::cuda_malloc;

const BUFFER_SIZE: usize = 8192;

fn main() {
    let handle = std::thread::spawn(|| {
        unsafe { 
            let err = cudaSetDevice(1);
            if err != cudaError::cudaSuccess {
                panic!("cudaSetDevice");
            }
        }
        let dev_ptr = cuda_malloc(1, BUFFER_SIZE).unwrap();
        let mut buf = vec![0i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()];
        buf.extend(vec![2042i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]);
        let err = unsafe { 
            cudaMemcpy(
                dev_ptr, 
                buf.as_ptr() as *const _, 
                BUFFER_SIZE,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        };
        if err != cudaError::cudaSuccess {
            panic!("cudaMemcpy failed");
        }
        println!("rank 1 - pre : buf[0]={}, buf[{}]={}", buf[0], BUFFER_SIZE / 2 / std::mem::size_of::<i32>(), buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]);

        let comm = init_communicator_rank(
            42,
            1,
            2,
            1,
        ).unwrap();
        all_gather(comm, (), (), BUFFER_SIZE / 2).unwrap();

        let mut buf = vec![0; BUFFER_SIZE];
        unsafe { 
            let err = cudaMemcpy(
                buf.as_mut_ptr() as *mut _, 
                dev_ptr,
                BUFFER_SIZE,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            if err != cudaError::cudaSuccess {
                panic!("cudaMemcpy failed");
            }
        };
        assert_eq!(buf[0], 1883);
        assert_eq!(buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()], 2042);
        println!("rank 1 - post : buf[0]={}, buf[{}]={}", buf[0], BUFFER_SIZE / 2 / std::mem::size_of::<i32>(), buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()])
    });

    unsafe { 
        let err = cudaSetDevice(0);
        if err != cudaError::cudaSuccess {
            panic!("cudaSetDevice");
        }
    }
    let dev_ptr = cuda_malloc(0, BUFFER_SIZE).unwrap();
    let mut buf = vec![1883i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()];
    buf.extend(vec![0i32; BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]);
    let err = unsafe { 
        cudaMemcpy(
            dev_ptr, 
            buf.as_ptr() as *const _, 
            BUFFER_SIZE,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
    };
    if err != cudaError::cudaSuccess {
        panic!("cudaMemcpy failed");
    }
    println!("rank 0 - pre : buf[0]={}, buf[{}]={}", buf[0], BUFFER_SIZE / 2 / std::mem::size_of::<i32>(), buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]);
    let comm = init_communicator_rank(
        42,
        0,
        2,
        0,
    ).unwrap();
    all_gather(comm, (), (), BUFFER_SIZE / 2).unwrap();
    let mut buf = vec![0; BUFFER_SIZE];
    unsafe { 
        let err = cudaMemcpy(
            buf.as_mut_ptr() as *mut _, 
            dev_ptr,
            BUFFER_SIZE,
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );
        if err != cudaError::cudaSuccess {
            panic!("cudaMemcpy failed");
        }
    };
    assert_eq!(buf[0], 1883);
    assert_eq!(buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()], 2042);
    println!("rank 0 - post : buf[0]={}, buf[{}]={}", buf[0], BUFFER_SIZE / 2 / std::mem::size_of::<i32>(), buf[BUFFER_SIZE / 2 / std::mem::size_of::<i32>()]);

    handle.join().unwrap();
}

