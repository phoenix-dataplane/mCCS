use std::net::IpAddr;
use std::process::ExitCode;

use structopt::StructOpt;

use cuda_runtime_sys::{cudaError, cudaMemcpyKind, cudaStream_t};
use cuda_runtime_sys::{cudaMemcpy, cudaSetDevice};

#[derive(Debug, Clone, StructOpt)]
#[structopt(name = "AllGather prototype")]
struct Opts {
    #[structopt(long)]
    root_addr: IpAddr,
    #[structopt(long)]
    rank: usize,
    #[structopt(long)]
    num_ranks: usize,
    #[structopt(long)]
    cuda_device_idx: i32,
    #[structopt(short, long, default_value = "42")]
    communicator: u32,
    #[structopt(short, long, default_value = "128")]
    size: usize,
}

fn main() -> ExitCode {
    let base_val = 2042;
    let opts = Opts::from_args();
    let buffer_size = opts.size * 1024 * 1024;
    let rank = opts.rank;
    let num_ranks = opts.num_ranks;

    unsafe {
        let err = cudaSetDevice(opts.cuda_device_idx);
        if err != cudaError::cudaSuccess {
            panic!("cudaSetDevice");
        }
    }
    let dev_ptr = libmccs::cuda_malloc(opts.cuda_device_idx, buffer_size * num_ranks).unwrap();
    let mut buf = vec![0i32; buffer_size * num_ranks / std::mem::size_of::<i32>()];
    buf[rank * buffer_size / std::mem::size_of::<i32>()
        ..(rank + 1) * buffer_size / std::mem::size_of::<i32>()]
        .fill(base_val + rank as i32);
    let err = unsafe {
        cudaMemcpy(
            dev_ptr.ptr,
            buf.as_ptr() as *const _,
            buffer_size * num_ranks,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
    };
    if err != cudaError::cudaSuccess {
        panic!("cudaMemcpy failed");
    }
    println!("rank {} - buffer initialized", rank);
    for r in 0..num_ranks {
        println!(
            "buf[{}]={}",
            r * buffer_size / std::mem::size_of::<i32>(),
            buf[r * buffer_size / std::mem::size_of::<i32>()]
        );
    }
    println!("**********");
    let comm = libmccs::init_communicator_rank(
        opts.communicator,
        rank,
        num_ranks,
        opts.cuda_device_idx,
        opts.root_addr,
    )
    .unwrap();
    println!("rank {} - communicator initialized", rank);

    libmccs::register_stream(opts.cuda_device_idx, 0 as cudaStream_t).unwrap();
    println!("rank {} - stream registered", rank);

    libmccs::all_gather(
        comm,
        dev_ptr.add(rank * buffer_size).unwrap(),
        dev_ptr,
        buffer_size,
        0 as cudaStream_t,
    )
    .unwrap();

    println!("rank {} - all gather issued", rank);

    let mut buf = vec![0; buffer_size * num_ranks / std::mem::size_of::<i32>()];
    unsafe {
        let err = cudaMemcpy(
            buf.as_mut_ptr() as *mut _,
            dev_ptr.ptr,
            buffer_size * num_ranks,
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );
        if err != cudaError::cudaSuccess {
            panic!("cudaMemcpy failed");
        }
    };

    println!("rank {}- all gather completed", rank);
    for r in 0..num_ranks {
        println!(
            "buf[{}]={}",
            r * buffer_size / std::mem::size_of::<i32>(),
            buf[r * buffer_size / std::mem::size_of::<i32>()]
        );
    }
    for r in 0..num_ranks {
        let data = buf[r * buffer_size / std::mem::size_of::<i32>()];
        let expected = base_val + r as i32;
        if data != expected {
            eprintln!("Rank{}: expected {}, got {}", r, expected, data);
            return ExitCode::FAILURE;
        }
    }
    return ExitCode::SUCCESS;
}
