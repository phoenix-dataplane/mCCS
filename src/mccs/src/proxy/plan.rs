use itertools::Itertools;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic;
use std::sync::atomic::{AtomicPtr, AtomicU32};
use std::{collections::VecDeque, mem::MaybeUninit};

use collectives_sys::{
    mccsDevComm, mccsDevWork, mccsDevWorkElem, mccsDevWorkHeader, mccsDevWorkType,
    mccsDevWork__bindgen_ty_1, mccsKernel_AllGather_RING_SIMPLE_Sum_int8_t,
};
use cuda_runtime_sys::{cudaEventRecord, cudaLaunchKernel, cudaMemcpy, cudaMemcpyKind};

use super::task::{CollTask, TaskAlgorithm, TaskProtocol, TaskSchema};
use crate::comm::{MCCS_MAX_CHANNELS, MCCS_WORK_FIFO_DEPTH};
use crate::transport::channel::{ChannelId, CommChannel};
use crate::transport::task::TransportTask;
use crate::{
    comm::Communicator,
    cuda::{alloc::DeviceAlloc, ptr::DeviceNonNull},
    cuda_warning,
};

const MCCS_MAX_ELEMENTS_PER_WORK: usize = 10;
const MCCS_SIMPLE_MAX_N_THREADS: usize = 512;
const MCCS_SIMPLE_THREAD_THRESHOLD: usize = 64;
const WARP_SIZE: usize = 32;

#[derive(Clone)]
pub struct WorkElem {
    num_warps: u8,
    send_buf: DeviceNonNull<c_void>,
    recv_buf: DeviceNonNull<c_void>,

    count: usize,
    root: u32,
    // block id
    bid: u8,
    num_channels: u8,
}

pub struct KernelWork {
    func_index: u16,
    work_elems: Vec<WorkElem>,
}

pub struct ChanWorkSchedule {
    pub coll_bytes: usize,
    pub work_queue: Vec<KernelWork>,
    pub agent_task_queue: Vec<TransportTask>,
}

impl ChanWorkSchedule {
    fn enqueue_work_elem_coll(
        &mut self,
        elem: WorkElem,
        work_func_index: u16,
        data_type_bytes: usize,
    ) {
        if let Some(tail) = self.work_queue.last_mut() {
            // accumulate same type work_elems
            if work_func_index == tail.func_index
                && elem.num_warps == tail.work_elems[0].num_warps
                && tail.work_elems.len() < MCCS_MAX_ELEMENTS_PER_WORK
            {
                tail.work_elems.push(elem);
                return;
            }
        }
        self.coll_bytes += elem.count * data_type_bytes;
        let work = KernelWork {
            func_index: work_func_index,
            work_elems: vec![elem],
        };
        self.work_queue.push(work);
    }
}

pub struct KernelPlan {
    kernel_fn: *const c_void,
    dev_work_head: DeviceNonNull<mccsDevWork>,

    channel_upper_bound: u32,
    channel_count: u32,
    channel_mask: u64,

    thread_per_block: usize,
}

impl Communicator {
    pub fn pre_launch_schedule(&mut self) {
        let first_task = self.task_queue.coll_queue.pop_front().unwrap();
        // todo: proxy
        self.compute_coll_work(&first_task);
        while let Some(coll_task) = self.task_queue.coll_queue.front() {
            if first_task.func == coll_task.func
                && first_task.data_type == coll_task.data_type
                && first_task.reduce_op.as_ref().is_some_and(|op| {
                    coll_task
                        .reduce_op
                        .as_ref()
                        .is_some_and(|op2| op.op == op2.op)
                })
            {
                let coll_task = self.task_queue.coll_queue.pop_front().unwrap();
                self.compute_coll_work(&coll_task);
            } else {
                break;
            }
        }
        let plan = self.finalize_one_plan();
        self.unlaunched_plans.push_back(plan);
    }

    // convert one task to different WorkElem object to different channel
    fn compute_coll_work(&mut self, task: &CollTask) {
        let schema = get_task_schema(
            task,
            TaskAlgorithm::Ring,
            TaskProtocol::Simple,
            self.channels.len(),
        );
        let num_wraps = schema.num_threads / 32;
        self.select_best_channels(schema.num_channels)
            .into_iter()
            .enumerate()
            .for_each(|(block_id, chan_id)| {
                let elem = WorkElem {
                    num_warps: num_wraps as _,
                    send_buf: task.send_buf.cast(),
                    recv_buf: task.recv_buf.cast(),
                    count: task.count,
                    root: task.root as _,
                    bid: block_id as u8,
                    num_channels: schema.num_channels as _,
                };
                self.plan_schedule
                    .get_mut(&chan_id)
                    .unwrap()
                    .enqueue_work_elem_coll(
                        elem,
                        schema.work_func_index,
                        task.data_type.count_bytes(),
                    );
            })
    }

    // block id -> ChannelId
    fn select_best_channels(&self, num: u32) -> Vec<ChannelId> {
        self.channels
            .iter()
            .map(|(chan_id, chan)| ChannelLoad {
                id: *chan_id,
                coll_bytes: self.plan_schedule.get(chan_id).unwrap().coll_bytes,
            })
            .k_smallest(num as usize)
            .map(|load| load.id)
            .collect()
    }

    fn finalize_one_plan(&mut self) -> KernelPlan {
        let ptr = mccsKernel_AllGather_RING_SIMPLE_Sum_int8_t;
        let mut chan_list = Vec::with_capacity(MCCS_MAX_ELEMENTS_PER_WORK);
        let mut channel_upper_bound = 0;
        let mut channel_mask = 0u64;
        let mut work_count = 0;
        for (idx, chan) in self.plan_schedule.iter_mut() {
            if !chan.work_queue.is_empty() {
                chan_list.push(*idx);
                channel_upper_bound = idx.0 + 1;
                channel_mask |= 1 << idx.0;
                work_count += chan.work_queue.len();
            }
        }
        // todo
        let dev_work = self.upload_work(&chan_list, channel_upper_bound, channel_mask, work_count);
        KernelPlan {
            kernel_fn: ptr as _,
            dev_work_head: dev_work,

            channel_count: chan_list.len() as u32,
            channel_upper_bound,
            channel_mask,

            thread_per_block: 512, //FIXME: should be bigger than maximum thread. Otherwise the program will hang
        }
    }

    // See NCCL enqueue.cc:waitWorkFifoAvailable()
    fn wait_work_queue(&mut self, target: u32) {
        // wrap around happens
        if rolling_less_u32(
            self.work_queue_acked_min + MCCS_WORK_FIFO_DEPTH as u32,
            target,
        ) {
            loop {
                let done_ptr = self.dev_resources.sync.work_queue_done.as_ptr_host();
                let mut ackd = [0u32; MCCS_MAX_CHANNELS];
                for i in 0..MCCS_MAX_CHANNELS {
                    unsafe {
                        ackd[i] =
                            AtomicU32::from_mut(done_ptr.offset(i as isize).as_mut().unwrap())
                                .load(atomic::Ordering::Relaxed);
                    }
                }
                atomic::compiler_fence(atomic::Ordering::SeqCst);
                let mut ackd_all = self.work_queue_next_available;
                for (id, chan) in self.channels.iter() {
                    if ackd[id.0 as usize] != chan.work_queue_next_available {
                        ackd_all = rolling_min_u32(ackd_all, ackd[id.0 as usize])
                    }
                }
                atomic::compiler_fence(atomic::Ordering::SeqCst);
                for (id, chan) in self.channels.iter() {
                    if ackd[id.0 as usize] == chan.work_queue_next_available {
                        unsafe {
                            AtomicU32::from_mut(done_ptr.offset(id.0 as isize).as_mut().unwrap())
                                .store(ackd_all, atomic::Ordering::Relaxed)
                        }
                    }
                }
                self.work_queue_acked_min = ackd_all;
                if !rolling_less_u32(
                    self.work_queue_acked_min + MCCS_WORK_FIFO_DEPTH as u32,
                    target,
                ) {
                    return;
                }
                std::thread::yield_now();
            }
        }
    }

    fn upload_work(
        &mut self,
        chan_list: &[ChannelId],
        channel_upper_bound: u32,
        channel_mask: u64,
        work_count: usize,
    ) -> DeviceNonNull<mccsDevWork> {
        let work_queue_mask = self.dev_resources.sync.work_queue_heap.size() as u32 - 1;
        let channel_count = chan_list.len() as u32;
        let new_first_chan_start = {
            let mut new_start = self.work_queue_next_available;
            if ((new_start + channel_count - 1) & work_queue_mask) < (new_start & work_queue_mask) {
                // wrap around happens
                new_start = (new_start + work_queue_mask) & !work_queue_mask;
                self.work_queue_next_available = new_start;
            }
            self.wait_work_queue(new_start + work_count as u32);
            new_start
        };

        // fill work
        let mut new_subsequent_start = new_first_chan_start + channel_count;
        let work_queue_heap_ptr = self.dev_resources.sync.work_queue_heap.as_ptr_host();
        for (nth_chan, chan_id) in chan_list.iter().enumerate() {
            let chan = self.plan_schedule.get(chan_id).unwrap();
            let work_len = chan.work_queue.len();
            chan.work_queue
                .iter()
                .enumerate()
                .for_each(|(work_id, work)| {
                    let dev_work = if work_id == work_len - 1 {
                        self.channels
                            .get_mut(chan_id)
                            .unwrap()
                            .work_queue_next_available = new_subsequent_start + 1;
                        work_elem_conversion(
                            work,
                            true,
                            false, /*todo:?*/
                            DevWorkHeaderUnion::DoneAcks(new_subsequent_start + 1),
                        )
                    } else {
                        work_elem_conversion(
                            work,
                            false,
                            false,
                            DevWorkHeaderUnion::WorkNext(
                                (if work_id == 0 {
                                    new_subsequent_start
                                } else {
                                    new_subsequent_start + 1
                                } & work_queue_mask) as i32
                                    - (new_first_chan_start & work_queue_mask) as i32,
                            ),
                        )
                    };
                    unsafe {
                        *work_queue_heap_ptr
                            .offset((new_subsequent_start & work_queue_mask) as isize) = dev_work;
                    }
                    if work_id != 0 {
                        new_subsequent_start += 1;
                    }
                });
        }
        self.work_queue_next_available = new_subsequent_start;
        // todo: GDR fence
        DeviceNonNull::new(unsafe {
            self.dev_resources
                .sync
                .work_queue_heap
                .as_ptr_dev()
                .offset((new_first_chan_start & work_queue_mask) as isize)
        })
        .unwrap()
    }
}

#[derive(Clone, Copy)]
enum DevWorkHeaderUnion {
    WorkNext(i32),
    DoneAcks(u32),
}

fn work_elem_conversion(
    work: &KernelWork,
    in_fifo: bool,
    is_last: bool,
    union_field: DevWorkHeaderUnion,
) -> mccsDevWork {
    debug_assert!(work.work_elems.len() <= MCCS_MAX_ELEMENTS_PER_WORK);

    let elems = unsafe {
        let elems = [MaybeUninit::zeroed(); MCCS_MAX_ELEMENTS_PER_WORK];
        let mut elems = MaybeUninit::array_assume_init(elems);
        for (idx, work_elem) in work.work_elems.iter().enumerate() {
            elems[idx] = unsafe {
                let mut elem = mccsDevWorkElem {
                    _bitfield_align_1: Default::default(),
                    _bitfield_1: Default::default(),
                    nWarps: work_elem.num_warps,
                    sendbuff: work_elem.send_buf.as_ptr(),
                    recvbuff: work_elem.recv_buf.as_ptr(),
                    count: work_elem.count,
                    root: work_elem.root,
                    bid: work_elem.bid,
                    nChannels: work_elem.num_channels,
                    redOpArg: 0, // todo
                };
                elem.set_isUsed(1);
                elem
            };
        }
        elems
    };
    let dev_work_elems = mccsDevWork__bindgen_ty_1 { elems };
    let dev_work_header = unsafe {
        let uninit = MaybeUninit::<mccsDevWorkHeader>::zeroed();
        let mut init = uninit.assume_init();
        init.funcIndex = work.func_index;
        init.type_ = mccsDevWorkType::mccsDevWorkTypeColl;
        init.set_inFifo(in_fifo as u8);
        init.set_isLast(is_last as u8);
        match union_field {
            DevWorkHeaderUnion::WorkNext(next) => init.__bindgen_anon_1.workNext = next,
            DevWorkHeaderUnion::DoneAcks(acks) => init.__bindgen_anon_1.doneAcks = acks,
        }
        init
    };

    mccsDevWork {
        header: dev_work_header,
        __bindgen_anon_1: dev_work_elems,
    }
}

fn get_task_schema(
    task: &CollTask,
    algo: TaskAlgorithm,
    proto: TaskProtocol,
    mut num_channel: usize,
) -> TaskSchema {
    use super::task::TaskDataType;

    assert_eq!(task.data_type, TaskDataType::Uint8);
    let mut num_thread = MCCS_SIMPLE_MAX_N_THREADS;
    let thread_th = MCCS_SIMPLE_THREAD_THRESHOLD;
    while task.count * task.data_type.count_bytes() < num_channel * num_thread * thread_th {
        if num_channel >= 2 {
            num_channel -= 1;
        } else if (num_thread % 128) == 0 {
            num_thread /= 2;
        } else {
            break;
        }
    }
    // todo: determine if "Extra warp for sync" necessary to be added when exceeding 512
    num_thread = if num_thread + WARP_SIZE > MCCS_SIMPLE_MAX_N_THREADS {
        MCCS_SIMPLE_MAX_N_THREADS
    } else {
        num_thread + WARP_SIZE
    }; // warning: should not exceed thread_per_block?
    TaskSchema {
        algorithm: algo,
        protocol: proto,
        work_func_index: todo!(),
        num_channels: num_channel as _,
        num_threads: num_thread as _,
    }
}

impl Communicator {
    pub fn launch_plan(&mut self) {
        use cuda_runtime_sys::dim3;
        let mut plan = self.unlaunched_plans.pop_front().unwrap();
        let mut dev_comm = self.dev_resources.get_dev_comm_ptr().as_ptr();
        let mut dev_work = plan.dev_work_head.as_ptr();
        let mut args = [
            &mut dev_comm as *mut *mut mccsDevComm as *mut c_void,
            &mut plan.channel_mask as *mut u64 as *mut c_void,
            &mut dev_work as *mut *mut mccsDevWork as *mut c_void,
        ];
        let grid = dim3 {
            x: plan.channel_count,
            y: 1,
            z: 1,
        };
        let block = dim3 {
            x: plan.thread_per_block as _,
            y: 1,
            z: 1,
        };
        unsafe {
            cuda_warning!(cudaLaunchKernel(
                plan.kernel_fn,
                grid,
                block,
                args.as_mut_ptr(),
                0,
                self.stream,
            ));
            cuda_warning!(cudaEventRecord(self.event, self.stream));
        }
    }
}
use std::cmp::Ordering;
#[derive(PartialEq, Eq)]
struct ChannelLoad {
    id: ChannelId,
    coll_bytes: usize,
}
impl PartialOrd for ChannelLoad {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.coll_bytes.partial_cmp(&other.coll_bytes) {
            Some(Ordering::Equal) => self.id.0.partial_cmp(&other.id.0),
            ord => ord,
        }
    }
}
impl Ord for ChannelLoad {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.coll_bytes.cmp(&other.coll_bytes) {
            Ordering::Equal => {}
            ord => return ord,
        }
        self.id.0.cmp(&other.id.0)
    }
}

fn rolling_less_u32(a: u32, b: u32) -> bool {
    a - b > i32::MAX as u32
}
fn rolling_min_u32(a: u32, b: u32) -> u32 {
    if (b - a) <= i32::MAX as u32 {
        a
    } else {
        b
    }
}
