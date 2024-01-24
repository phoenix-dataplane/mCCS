use std::collections::VecDeque;

use ipc::mccs::command::{AllReduceDataType, AllReduceOpType};

use crate::cuda::ptr::DeviceNonNull;

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

impl From<AllReduceDataType> for TaskDataType {
    fn from(value: AllReduceDataType) -> Self {
        match value {
            AllReduceDataType::Float16 => Self::Float16,
            AllReduceDataType::Int32 => Self::Int32,
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

impl From<AllReduceOpType> for TaskReduceOpType {
    fn from(value: AllReduceOpType) -> Self {
        match value {
            AllReduceOpType::Sum => TaskReduceOpType::Sum,
            AllReduceOpType::Prod => TaskReduceOpType::Prod,
            AllReduceOpType::Max => TaskReduceOpType::Max,
            AllReduceOpType::Min => TaskReduceOpType::Min,
            AllReduceOpType::PreMulSum => TaskReduceOpType::PreMulSum,
            AllReduceOpType::SumPostDiv => TaskReduceOpType::SumPostDiv,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskReduceOp {
    pub op: TaskReduceOpType,
    pub arg: u64,
}

#[derive(Debug, Clone)]
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

impl CollTask {
    pub fn total_bytes(&self, n_rank: usize) -> usize {
        let n_bytes = self.count * self.data_type.count_bytes();
        match self.func {
            TaskFuncType::AllGather => n_bytes * n_rank,
            TaskFuncType::AllReduce => n_bytes,
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskAlgorithm {
    Ring = 0,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskProtocol {
    Simple = 0,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskPattern {
    Ring,
    RingTwice,
    PipelineFrom,
    PipelineTo,
    TreeUp,
    TreeDown,
    TreeUpDown,
    CollnetChain,
    CollnetDirect,
    Send,
    Recv,
}

#[derive(Debug, Clone)]
pub struct TaskSchema {
    pub algorithm: TaskAlgorithm,
    pub protocol: TaskProtocol,
    pub coll_func: TaskFuncType,
    pub work_func_index: u16,
    pub num_channels: u32,
    pub num_threads: u32,
}

impl TaskSchema {
    fn get_pattern(&self) -> TaskPattern {
        match self.coll_func {
            TaskFuncType::AllGather => TaskPattern::Ring,
            TaskFuncType::AllReduce => {
                // if self.algorithm==TaskAlgorithm::Tree{TaskPattern::TreeUpDown}
                TaskPattern::RingTwice
            }
            _ => unimplemented!(),
        }
    }
    pub fn get_num_chunks_per_loop(&self, n_ranks: u32) -> u32 {
        match self.get_pattern() {
            TaskPattern::Ring => n_ranks,
            TaskPattern::RingTwice => n_ranks,
            _ => unimplemented!(),
        }
    }

    pub fn get_num_steps_per_loop(&self, n_ranks: u32) -> u32 {
        match self.get_pattern() {
            TaskPattern::Ring => n_ranks - 1,
            TaskPattern::RingTwice => 2 * (n_ranks - 1),
            _ => unimplemented!(),
        }
    }
}
pub struct TaskQueue {
    pub coll_queue: VecDeque<CollTask>,
}
