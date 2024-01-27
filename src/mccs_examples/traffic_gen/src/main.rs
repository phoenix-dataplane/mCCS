use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use structopt::StructOpt;

use cuda_runtime_sys::{cudaError, cudaMemcpyKind, cudaStream_t};
use cuda_runtime_sys::{cudaMemcpy, cudaSetDevice};

const CUDA_FLOAT16_SIZE: usize = 2;

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OpType {
    #[serde(alias = "all_reduce")]
    AllReduce,
    #[serde(alias = "all_gather")]
    AllGather,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OperationTrace {
    // Message size in bytes
    #[serde(alias = "size")]
    message_size: usize,
    #[serde(alias = "type")]
    op_type: OpType,
    // Simulated compute interval: microseconds
    compute_interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    // Global communicator id
    #[serde(alias = "comm_id")]
    communicator_id: u32,
    // Array with num_ranks elements, rank -> cuda dev mapping
    #[serde(alias = "cuda_devices")]
    communicator_devices: Vec<i32>,
    traces: Vec<OperationTrace>,
}

impl Config {
    fn from_path<P: AsRef<Path>>(path: P) -> Config {
        let content = std::fs::read_to_string(path).unwrap();
        let config = toml::from_str(&content).unwrap();
        config
    }
}

#[derive(Debug, Clone, StructOpt)]
#[structopt(name = "Trace traffic generator")]
struct Opts {
    #[structopt(long, short = "a")]
    root_addr: IpAddr,
    #[structopt(long, short = "r")]
    rank: usize,
    // Path to toml traffic trace
    #[structopt(long, short = "c")]
    config: PathBuf,
    // Number of iterations
    #[structopt(long, short = "i", default_value = "20")]
    iters: usize,
    #[structopt(long, short = "s")]
    save_path: Option<PathBuf>,
    #[structopt(long, short = "v")]
    verbose: bool,
}

fn main() -> ExitCode {
    let opts = Opts::from_args();
    let rank = opts.rank;
    let num_iters = opts.iters;
    let config = Config::from_path(opts.config);

    let num_ranks = config.communicator_devices.len();
    let cuda_device_idx = config.communicator_devices[rank];
    let comm_id = config.communicator_id;

    unsafe {
        let err = cudaSetDevice(cuda_device_idx);
        if err != cudaError::cudaSuccess {
            panic!("cudaSetDevice failed {:?}", err);
        }
    }
    let traces = config.traces;
    let mut buffer_size = traces
        .iter()
        .map(|x| match x.op_type {
            OpType::AllReduce => x.message_size,
            OpType::AllGather => x.message_size * num_ranks,
        })
        .max()
        .unwrap();
    buffer_size = buffer_size.div_ceil(num_ranks) * num_ranks;

    let dev_ptr = libmccs::cuda_malloc(cuda_device_idx, buffer_size).unwrap();
    let buf = vec![80u8; buffer_size];
    unsafe {
        let err = cudaMemcpy(
            dev_ptr.ptr,
            buf.as_ptr() as *const _,
            buffer_size,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );
        if err != cudaError::cudaSuccess {
            panic!("cudaMemcpy failed {:?}", err);
        }
    };
    let comm =
        libmccs::init_communicator_rank(comm_id, rank, num_ranks, cuda_device_idx, opts.root_addr)
            .unwrap();
    libmccs::register_stream(cuda_device_idx, 0 as cudaStream_t).unwrap();
    println!("Rank {}: start warmup", rank);

    for _ in 0..5 {
        libmccs::all_gather(
            comm,
            dev_ptr.add(rank * (buffer_size / num_ranks)).unwrap(),
            dev_ptr,
            buffer_size / num_ranks,
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
    println!("Rank {}: warmup phase AG finished", rank);

    for _ in 0..5 {
        libmccs::all_reduce(
            comm,
            dev_ptr,
            dev_ptr,
            buffer_size / CUDA_FLOAT16_SIZE,
            libmccs::AllReduceDataType::Float16,
            libmccs::AllReduceOpType::Sum,
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
    println!("Rank {}: warmup phase finished", rank);

    // start testing
    let mut start = Instant::now();
    let mut round_times = Vec::with_capacity(num_iters);
    for _ in 0..num_iters {
        for op in traces.iter() {
            spin_sleep::sleep(Duration::from_micros(op.compute_interval));
            match op.op_type {
                OpType::AllReduce => {
                    libmccs::all_reduce(
                        comm,
                        dev_ptr,
                        dev_ptr,
                        op.message_size / CUDA_FLOAT16_SIZE,
                        libmccs::AllReduceDataType::Float16,
                        libmccs::AllReduceOpType::Sum,
                        0 as cudaStream_t,
                    )
                    .unwrap();
                }
                OpType::AllGather => {
                    libmccs::all_gather(
                        comm,
                        dev_ptr.add(rank * (buffer_size / num_ranks)).unwrap(),
                        dev_ptr,
                        op.message_size,
                        0 as cudaStream_t,
                    )
                    .unwrap();
                }
            }
            unsafe {
                let err = cuda_runtime_sys::cudaStreamSynchronize(0 as cudaStream_t);
                if err != cudaError::cudaSuccess {
                    panic!("cudaStreamSynchronize failed");
                }
            }
        }
        let end = Instant::now();
        let dura = end.duration_since(start);
        start = end;
        let round_time = dura.as_micros() as u64;
        if opts.verbose && opts.rank == 0 {
            println!("Iter time: {} ms", round_time / 1000);
        }
        round_times.push(round_time);
    }
    if opts.rank == 0 && opts.save_path.is_some() {
        let mut wtr = csv::Writer::from_path(opts.save_path.unwrap()).unwrap();
        wtr.serialize(round_times).unwrap();
    }

    return ExitCode::SUCCESS;
}
