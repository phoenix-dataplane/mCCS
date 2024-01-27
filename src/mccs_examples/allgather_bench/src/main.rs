use std::net::IpAddr;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use structopt::StructOpt;

use cuda_runtime_sys::{cudaError, cudaMemcpyKind, cudaStream_t};
use cuda_runtime_sys::{cudaMemcpy, cudaSetDevice};

#[derive(Debug, Clone, StructOpt)]
#[structopt(name = "AllGather Benchmark")]
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
    #[structopt(long)]
    size_in_byte: bool,
    #[structopt(long, default_value = "20")]
    round: usize,
    #[structopt(long, default_value = "1")]
    epoch: usize,
    #[structopt(long)]
    name: Option<String>,
}

fn get_prefix(name: Option<&str>) -> String {
    match name {
        Some(name) => format!("[{}] ", name),
        None => String::new(),
    }
}

fn main() -> ExitCode {
    let base_val = 2042;
    let opts = Opts::from_args();
    let buffer_size = if opts.size_in_byte {
        opts.size
    } else {
        opts.size * 1024 * 1024
    };
    let rank = opts.rank;
    let num_ranks = opts.num_ranks;
    let prefix = get_prefix(opts.name.as_deref());

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
    println!("{}Rank {}: warm phase started", prefix, rank);
    for r in 0..num_ranks {
        println!(
            "{}buf[{}]={}",
            prefix,
            r * buffer_size / std::mem::size_of::<i32>(),
            buf[r * buffer_size / std::mem::size_of::<i32>()]
        );
    }
    let comm = libmccs::init_communicator_rank(
        opts.communicator,
        rank,
        num_ranks,
        opts.cuda_device_idx,
        opts.root_addr,
    )
    .unwrap();
    libmccs::register_stream(opts.cuda_device_idx, 0 as cudaStream_t).unwrap();
    println!("{}Rank {}: start issuing", prefix, rank);

    libmccs::all_gather(
        comm,
        dev_ptr.add(rank * buffer_size).unwrap(),
        dev_ptr,
        buffer_size,
        0 as cudaStream_t,
    )
    .unwrap();
    println!("{}Rank {}: warmup call returned", prefix, rank);
    let mut buf2 = vec![0; buffer_size * num_ranks / std::mem::size_of::<i32>()];
    unsafe {
        let err = cudaMemcpy(
            buf2.as_mut_ptr() as *mut _,
            dev_ptr.ptr,
            buffer_size * num_ranks,
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );
        if err != cudaError::cudaSuccess {
            panic!("cudaMemcpy failed");
        }
    };
    for r in 0..num_ranks {
        let data = buf2[r * buffer_size / std::mem::size_of::<i32>()];
        let expected = base_val + r as i32;
        if data != expected {
            eprintln!("{}Rank{}: expected {}, got {}", prefix, r, expected, data);
            return ExitCode::FAILURE;
        }
    }
    for _ in 0..5 {
        libmccs::all_gather(
            comm,
            dev_ptr.add(rank * buffer_size).unwrap(),
            dev_ptr,
            buffer_size,
            0 as cudaStream_t,
        )
        .unwrap();
    }
    unsafe {
        let err = cuda_runtime_sys::cudaStreamSynchronize(0 as cudaStream_t);
        if err != cudaError::cudaSuccess {
            panic!("cudaStreamSynchronize failed");
        }
    }
    println!("{}Rank {}: warmup phase finished", prefix, opts.rank);
    // start testing
    for e in 0..opts.epoch {
        let start = Instant::now();
        for _ in 0..opts.round {
            libmccs::all_gather(
                comm,
                dev_ptr.add(rank * buffer_size).unwrap(),
                dev_ptr,
                buffer_size,
                0 as cudaStream_t,
            )
            .unwrap();
        }
        unsafe {
            let err = cuda_runtime_sys::cudaStreamSynchronize(0 as cudaStream_t);
            if err != cudaError::cudaSuccess {
                panic!("cudaStreamSynchronize failed");
            }
        }
        let end = Instant::now();
        let dura = end.duration_since(start);
        let tput =
            (buffer_size * num_ranks * opts.round) as f64 / 1e9 / (dura.as_micros() as f64 / 1.0e6);
        let bus = tput * (num_ranks - 1) as f64 / num_ranks as f64;
        if opts.epoch > 1 {
            println!(
                "{}[Epoch={}] [Rank {}/{}] Algorithm bandwidth: {:.} GB/s, Bus bandwidth: {:.} GB/s",
                prefix, e, rank, num_ranks, tput, bus
            );
        } else {
            println!(
                "{}[Rank {}/{}] Algorithm bandwidth: {:.} GB/s, Bus bandwidth: {:.} GB/s",
                prefix, rank, num_ranks, tput, bus
            );
        }
    }
    return ExitCode::SUCCESS;
}

#[allow(dead_code)]
fn get_stats(mut durations: Vec<Duration>) -> (Duration, Duration, Duration, Duration) {
    durations.sort();

    let sum: Duration = durations.iter().sum();
    let mean = sum / durations.len() as u32;

    let median = if durations.len() % 2 == 0 {
        let mid = durations.len() / 2;
        (durations[mid - 1] + durations[mid]) / 2
    } else {
        durations[durations.len() / 2]
    };

    let &min = durations.first().unwrap();
    let &max = durations.last().unwrap();

    (mean, median, min, max)
}
