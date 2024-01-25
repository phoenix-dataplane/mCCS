use std::collections::VecDeque;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::atomic;
use std::sync::atomic::{AtomicPtr, AtomicU32};

use crossbeam::channel::Sender;
use itertools::Itertools;

use collectives_sys::{
    mccsDevComm, mccsDevWork, mccsDevWorkElem, mccsDevWorkHeader, mccsDevWorkType,
    mccsDevWork__bindgen_ty_1, mccsKernel_AllGather_RING_SIMPLE_Sum_int8_t,
    mccsKernel_AllReduce_RING_SIMPLE_Prod_half, mccsKernel_AllReduce_RING_SIMPLE_Prod_int32_t,
    mccsKernel_AllReduce_RING_SIMPLE_Sum_half, mccsKernel_AllReduce_RING_SIMPLE_Sum_int32_t,
};
use cuda_runtime_sys::{cudaEventRecord, cudaLaunchKernel, cudaMemcpy, cudaMemcpyKind};

use super::task::{
    CollTask, TaskAlgorithm, TaskDataType, TaskFuncType, TaskProtocol, TaskReduceOp,
    TaskReduceOpType, TaskSchema,
};
use crate::comm::{ChannelCommPattern, MCCS_MAX_CHANNELS, MCCS_WORK_FIFO_DEPTH};
use crate::pattern::MCCS_STEP;
use crate::transport::channel::{ChannelId, CommChannel, ConnType, PeerConnId};
use crate::transport::engine::TransportEngineId;
use crate::transport::message::TransportEngineRequest;
use crate::transport::op::{TransportOp, TransportOpState};
use crate::transport::task::TransportTask;
use crate::transport::transporter::TransportAgentId;
use crate::transport::Protocol;
use crate::{
    comm::Communicator,
    cuda::{alloc::DeviceAlloc, ptr::DeviceNonNull},
    cuda_warning,
};

const MCCS_MAX_ELEMENTS_PER_WORK: usize = 10;
const MCCS_SIMPLE_MAX_N_THREADS: usize = 512;
const MCCS_SIMPLE_THREAD_THRESHOLD: usize = 64;
const WARP_SIZE: usize = 32;

#[derive(Clone, Debug)]
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
        self.coll_bytes += elem.count * data_type_bytes;
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
        let work = KernelWork {
            func_index: work_func_index,
            work_elems: vec![elem],
        };
        self.work_queue.push(work);
    }

    fn clear(&mut self) {
        self.coll_bytes = 0;
        self.work_queue.clear();
        self.agent_task_queue.clear();
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
    pub fn pre_launch_schedule(
        &mut self,
        transport_tx: &mut HashMap<TransportEngineId, Sender<TransportEngineRequest>>,
        device_id: i32,
    ) {
        let first_task = self.task_queue.coll_queue.pop_front().unwrap();
        let mut max_threads_per_block = 0;
        max_threads_per_block = std::cmp::max(
            max_threads_per_block,
            self.compute_coll_work(&first_task, device_id),
        );
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
                max_threads_per_block = std::cmp::max(
                    max_threads_per_block,
                    self.compute_coll_work(&coll_task, device_id),
                );
                log::trace!("Compute more coll task");
            } else {
                break;
            }
        }
        let func = match first_task.func {
            TaskFuncType::AllGather => mccsKernel_AllGather_RING_SIMPLE_Sum_int8_t as *const _,
            TaskFuncType::AllReduce => match (
                first_task.data_type,
                first_task.reduce_op.expect("AllReduce missing op").op,
            ) {
                (TaskDataType::Float16, TaskReduceOpType::Sum) => {
                    mccsKernel_AllReduce_RING_SIMPLE_Sum_half as *const _
                }

                (TaskDataType::Float16, TaskReduceOpType::Prod) => {
                    mccsKernel_AllReduce_RING_SIMPLE_Prod_half as *const _
                }
                (TaskDataType::Int32, TaskReduceOpType::Sum) => {
                    mccsKernel_AllReduce_RING_SIMPLE_Sum_int32_t as *const _
                }

                (TaskDataType::Int32, TaskReduceOpType::Prod) => {
                    mccsKernel_AllReduce_RING_SIMPLE_Prod_int32_t as *const _
                }

                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        };
        let plan = self.finalize_one_plan(func, max_threads_per_block, transport_tx);
        self.unlaunched_plans.push_back(plan);
    }

    // convert one task to different WorkElem objects and append them to different channels
    fn compute_coll_work(&mut self, task: &CollTask, device_id: i32) -> u32 {
        let schema = get_task_schema(
            task,
            TaskAlgorithm::Ring,
            TaskProtocol::Simple,
            self.num_ranks,
            self.channels.len(),
        );
        log::debug!("task schema: {:?}", schema);
        log::debug!("CollTask: {:?}", task);
        let num_wraps = schema.num_threads / 32;

        let (chunk_steps, slice_steps, num_step) = {
            // computeColl()
            let (chunk_steps, slice_steps) = if schema.algorithm == TaskAlgorithm::Ring
                && schema.protocol == TaskProtocol::Simple
            {
                (task.chunk_steps, task.slice_steps)
            } else {
                (1, 1)
            };
            let step_size = self.profile.buff_sizes[Protocol::Simple as usize] as u32 / MCCS_STEP;

            let chunk_size = chunk_steps * step_size;
            let n_loops = {
                let total_bytes = task.total_bytes(self.num_ranks);
                let per_loop_size = schema.num_channels as usize
                    * schema.get_num_chunks_per_loop(self.num_ranks as u32) as usize
                    * chunk_size as usize;
                log::debug!(
                    "total_bytes={} per_loop_size={}",
                    total_bytes,
                    per_loop_size
                );
                // DIVUP
                ((total_bytes + per_loop_size - 1) / per_loop_size) as u32
            };

            log::debug!("nloops={}, num_channels={}", n_loops, schema.num_channels);
            (
                chunk_steps,
                slice_steps,
                schema.get_num_steps_per_loop(self.num_ranks as u32) * n_loops * chunk_steps,
            )
        };
        log::debug!(
            "chunk_steps={}, slice_steps={}, num_step={}",
            chunk_steps,
            slice_steps,
            num_step
        );
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
                log::debug!("Chan[{}]: WorkElem={:?}", chan_id.0, elem);
                let schedule = self.plan_schedule.get_mut(&chan_id).unwrap();
                // work queue
                schedule.enqueue_work_elem_coll(
                    elem,
                    schema.work_func_index,
                    task.data_type.count_bytes(), // FIXME: may buggy due to ncclInfoSetDerived
                );
                debug_assert!(schema.algorithm == TaskAlgorithm::Ring);
                // proxy queue
                {
                    let tx_op = TransportOp::new(
                        self.id,
                        num_step,
                        slice_steps,
                        chunk_steps,
                        match schema.protocol {
                            TaskProtocol::Simple => crate::transport::Protocol::Simple,
                        },
                    );
                    let send_agent_id = TransportAgentId {
                        communicator_id: self.id,
                        client_rank: self.rank,
                        client_cuda_dev: device_id,
                        peer_conn: PeerConnId {
                            peer_rank: self.channels.get(&chan_id).unwrap().ring.next,
                            channel: chan_id,
                            conn_index: 0,
                            conn_type: ConnType::Send,
                        },
                    };
                    let recv_agent_id = TransportAgentId {
                        communicator_id: self.id,
                        client_rank: self.rank,
                        client_cuda_dev: device_id,
                        peer_conn: PeerConnId {
                            peer_rank: self.channels.get(&chan_id).unwrap().ring.prev,
                            channel: chan_id,
                            conn_index: 0,
                            conn_type: ConnType::Recv,
                        },
                    };
                    schedule.agent_task_queue.push(TransportTask {
                        agent_id: send_agent_id,
                        op: tx_op.clone(),
                    });
                    schedule.agent_task_queue.push(TransportTask {
                        agent_id: recv_agent_id,
                        op: tx_op,
                    });
                }
            });
        schema.num_threads
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

    fn finalize_one_plan(
        &mut self,
        kernel_func: *const c_void,
        thread_per_block: u32,
        transport_tx: &mut HashMap<TransportEngineId, Sender<TransportEngineRequest>>,
    ) -> KernelPlan {
        let mut chan_list = Vec::with_capacity(MCCS_MAX_ELEMENTS_PER_WORK);
        let mut channel_upper_bound = 0;
        let mut channel_mask = 0u64;
        let mut work_count = 0;
        // warning: should not clear now
        for (idx, chan) in self.plan_schedule.iter_mut() {
            if !chan.work_queue.is_empty() {
                chan_list.push(*idx);
                channel_upper_bound = idx.0 + 1;
                channel_mask |= 1 << idx.0;
                work_count += chan.work_queue.len();
                // upload ProxyOp
                chan.agent_task_queue.iter().for_each(|task| {
                    let TransportTask { agent_id, op } = task;
                    let peer_conn = self.channels.
                        get(&agent_id.peer_conn.channel)
                        .unwrap()
                        .peers
                        .get(&agent_id.peer_conn.peer_rank)
                        .unwrap();
                    let connector = match agent_id.peer_conn.conn_type {
                        ConnType::Send => &peer_conn.send[agent_id.peer_conn.conn_index as usize],
                        ConnType::Recv => &peer_conn.recv[agent_id.peer_conn.conn_index as usize],
                    }.as_ref().unwrap();
                    if connector.transporter.need_op() {
                        let tx_engine_id = connector.transport_agent_engine.unwrap();
                        log::debug!(
                            "Submit new TxOp: tx_engine_id={:?}, agent_id={:?}, op={{num_steps={:?}}}",
                            tx_engine_id,
                            agent_id,
                            op.num_steps,
                        );
                        transport_tx.get_mut(&tx_engine_id)
                            .expect("Channels to transport engine should be established after communicator init")
                            .send(TransportEngineRequest::AgentTransportOp(*agent_id, op.clone()))
                            .unwrap();
                    }else{
                        log::trace!(
                            "No Transporter Needed: agent_id={:?}, op={{num_steps={:?}}}",
                            agent_id,
                            op.num_steps,
                        );
                    }
                })
            }
        }
        let dev_work = self.upload_work(&chan_list, channel_upper_bound, channel_mask, work_count);
        self.plan_schedule
            .iter_mut()
            .for_each(|(_, schedule)| schedule.clear());
        log::debug!(
            "Finalized one KernelPlan: [{}/{}/{:b}; {}]",
            chan_list.len(),
            channel_upper_bound,
            channel_mask,
            thread_per_block
        );
        KernelPlan {
            kernel_fn: kernel_func as _,
            dev_work_head: dev_work,

            channel_count: chan_list.len() as u32,
            channel_upper_bound,
            channel_mask,

            thread_per_block: thread_per_block as usize,
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
        log::trace!("Upload to {:?} with {} works", chan_list, work_count);
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
        log::trace!(
            "work queue ptr from {} to {}",
            new_first_chan_start,
            new_first_chan_start + work_count as u32
        );

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
                    log::trace!("Work: channel {}, id {}", nth_chan, work_id);
                    let dev_work = if work_id == work_len - 1 {
                        self.channels
                            .get_mut(chan_id)
                            .unwrap()
                            .work_queue_next_available = new_subsequent_start + 1;
                        work_elem_conversion(
                            work,
                            true,
                            true,
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
                    let current_offset = if work_id == 0 {
                        new_first_chan_start + nth_chan as u32
                    } else {
                        let offset = new_subsequent_start;
                        new_subsequent_start += 1;
                        offset
                    };
                    unsafe {
                        log::trace!(
                            "Adding work to {}",
                            (current_offset & work_queue_mask) as isize
                        );
                        *work_queue_heap_ptr.offset((current_offset & work_queue_mask) as isize) =
                            dev_work;
                    }
                });
        }
        self.work_queue_next_available = new_subsequent_start;
        // todo: GDR fence
        // unsafe {
        //     for i in new_first_chan_start..(new_first_chan_start + work_count as u32) {
        //         let work = *work_queue_heap_ptr.offset(i as isize);
        //         let mut s = format!(
        //             "\n[isLast:{} inFifo:{} {} type:{:?} count:{}]\n",
        //             work.header.isLast(),
        //             work.header.inFifo(),
        //             if work.header.isLast() != 0 {
        //                 format!("DoneAcks: {}", work.header.__bindgen_anon_1.doneAcks)
        //             } else {
        //                 format!("WorkNext: {}", work.header.__bindgen_anon_1.workNext)
        //             },
        //             work.header.type_,
        //             work_count
        //         );
        //         for (idx, elem) in work.__bindgen_anon_1.elems.iter().enumerate() {
        //             if elem.isUsed() != 0 {
        //                 s += format!(
        //                     "\t {}: (nWarps:{} count:{} bid:{} nChannels:{} root:{})",
        //                     idx, elem.nWarps, elem.count, elem.bid, elem.nChannels, elem.root
        //                 )
        //                 .as_str();
        //             }
        //         }
        //         log::debug!("Upload work:{}", s);
        //     }
        // }
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
        init.funcIndex = work.func_index; // seems not used in common.h
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
    n_rank: usize,
    mut num_channel: usize,
) -> TaskSchema {
    let mut num_thread = MCCS_SIMPLE_MAX_N_THREADS;
    let thread_th = MCCS_SIMPLE_THREAD_THRESHOLD;
    while task.total_bytes(n_rank) < num_channel * num_thread * thread_th {
        if num_channel >= 2 {
            num_channel -= 1;
        } else if (num_thread % 128) == 0 {
            num_thread /= 2;
        } else {
            break;
        }
    }
    num_thread += WARP_SIZE;
    if num_thread / WARP_SIZE < 3 {
        num_thread = WARP_SIZE * 3
    }
    // warning: should not exceed thread_per_block?
    debug_assert!(num_thread <= MCCS_SIMPLE_MAX_N_THREADS + WARP_SIZE);
    log::debug!("num_thread={}, num_channel={}", num_thread, num_channel);
    TaskSchema {
        algorithm: algo,
        protocol: proto,
        work_func_index: 0, // FIXME
        num_channels: num_channel as _,
        num_threads: num_thread as _,
        coll_func: task.func,
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
