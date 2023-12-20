use std::collections::VecDeque;

use crate::cuda::ptr::DeviceNonNull;
use crate::message::CudaStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskFuncType {
    Boadcast = 0,
    Reduce = 1,
    AllGather = 2,
    ReduceScatter = 3,
    AllReduce = 4,
    SendRecv = 5,
    FuncSend = 6,
    FuncRecv = 7,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskDataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
}

impl TaskDataType {
    pub fn count_bytes(&self) -> usize {
        match self {
            TaskDataType::Int8 | TaskDataType::Uint8 => 1,
            TaskDataType::Float16 => 2,
            TaskDataType::Int32 | TaskDataType::Uint32 | TaskDataType::Float32 => 4,
            TaskDataType::Int64 | TaskDataType::Uint64 | TaskDataType::Float64 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskReduceOpType {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    PreMulSum = 4,
    SumPostDiv = 5,
}

pub struct TaskReduceOp {
    pub op: TaskReduceOpType,
    pub arg: u64,
}

pub struct CollTask {
    pub func: TaskFuncType,
    pub send_buf: DeviceNonNull<u8>,
    pub recv_buf: DeviceNonNull<u8>,
    pub count: usize,
    pub root: usize,
    pub data_type: TaskDataType,
    pub reduce_op: Option<TaskReduceOp>,
    pub chunk_steps: u32,
    pub slice_steps: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskAlgorithm {
    Ring = 0,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskProtocol {
    Simple = 0,
}

#[derive(Debug, Clone)]
pub struct TaskSchema {
    pub algorithm: TaskAlgorithm,
    pub protocol: TaskProtocol,
    pub work_func_index: u16,
    pub num_channels: u32,
    pub num_threads: u32,
}

pub struct TaskQueue {
    pub coll_queue: VecDeque<CollTask>,
}
