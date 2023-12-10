use itertools::Itertools;
use std::collections::HashMap;
use std::ffi::c_void;
use std::{collections::VecDeque, mem::MaybeUninit};

use collectives_sys::{
    mccsDevComm, mccsDevWork, mccsDevWorkElem, mccsDevWorkHeader, mccsDevWorkType,
    mccsDevWork__bindgen_ty_1, mccsKernel_AllGather_RING_SIMPLE_Sum_int8_t,
};
use cuda_runtime_sys::{cudaEventRecord, cudaLaunchKernel, cudaMemcpy, cudaMemcpyKind};

use super::task::{CollTask, TaskAlgorithm, TaskProtocol, TaskSchema};
use crate::transport::channel::{ChannelId, CommChannel};
use crate::transport::task::TransportTask;
use crate::{
    comm::Communicator,
    cuda::{alloc::DeviceAlloc, ptr::DeviceNonNull},
    cuda_warning,
};

const MCCS_MAX_ELEMENTS_PER_WORK: usize = 10;

#[derive(Clone)]
pub struct WorkElemColl {
    num_warps: u8,
    send_buf: DeviceNonNull<c_void>,
    recv_buf: DeviceNonNull<c_void>,

    count: usize,
    root: u32,
    // block id
    bid: u8,
    num_channels: u8,
}

pub enum KernelWork {
    Coll {
        func_index: u16,
        work_elems: Vec<WorkElemColl>,
    },
}

pub struct ChanWorkSchedule {
    pub coll_bytes: usize,
    pub work_queue: Vec<KernelWork>,
    pub agent_task_queue: Vec<TransportTask>,
}

impl ChanWorkSchedule {
    fn enqueue_work_elem_coll(&mut self, elem: WorkElemColl, work_func_index: u16) {
        if let Some(tail) = self.work_queue.last_mut() {
            if let KernelWork::Coll {
                func_index,
                work_elems,
            } = tail
            {
                // accumulate same type work_elems
                if work_func_index == *func_index
                    && elem.num_warps == work_elems[0].num_warps
                    && work_elems.len() < MCCS_MAX_ELEMENTS_PER_WORK
                {
                    work_elems.push(elem);
                    return;
                }
            }
        }
        let work = KernelWork::Coll {
            func_index: work_func_index,
            work_elems: vec![elem],
        };
        self.work_queue.push(work);
        self.coll_bytes += elem.count * todo!("depends on work_func_index");
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
        let coll_task = self.task_queue.coll_queue.pop_front().unwrap();
        self.compute_coll_work(&coll_task);
        self.unlaunched_plans.push_back(self.finalize_one_plan());
    }

    fn compute_coll_work(&mut self, task: &CollTask) {
        let schema = get_task_schema(task);
        let num_wraps = schema.num_threads / 32;
        self.select_best_channels(schema.num_channels)
            .into_iter()
            .enumerate()
            .for_each(|(block_id, chan_id)| {
                let elem = WorkElemColl {
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
                    .enqueue_work_elem_coll(elem, todo!());
            })
    }

    // block id -> ChannelId
    fn select_best_channels(&self, num: u32) -> Vec<ChannelId> {
        self.channels
            .iter()
            .enumerate()
            .map(|(chan_id, chan)| ChannelLoad {
                id: ChannelId(chan_id as u32),
                coll_bytes: self
                    .plan_schedule
                    .get(&ChannelId(chan_id as u32))
                    .unwrap()
                    .coll_bytes,
            })
            .k_smallest(num as usize)
            .map(|load| load.id)
            .collect()
    }

    fn finalize_one_plan(&mut self) -> KernelPlan {
        let ptr = mccsKernel_AllGather_RING_SIMPLE_Sum_int8_t;
        let dev_work = upload_work(
            self.plan_schedule
                .get_mut(&ChannelId(0))
                .unwrap()
                .work_queue
                .pop_front()
                .unwrap(),
        );
        let plan = KernelPlan {
            kernel_fn: ptr as _,
            dev_work_head: dev_work,

            channel_count: 1,
            channel_upper_bound: 1,
            channel_mask: 1,

            thread_per_block: 512, //FIXME: should be bigger than maximum thread. Otherwise the program will hang
        };
    }
}

fn get_task_schema(task: &CollTask) -> TaskSchema {
    use super::task::TaskDataType;

    let algorithm = TaskAlgorithm::Ring;
    let protocol = TaskProtocol::Simple;
    assert_eq!(task.data_type, TaskDataType::Uint8);
    let mut nc = 1;
    let mut nt = 512;
    let thread_th = 64;
    while task.count < nc * nt * thread_th {
        if nc >= 2 {
            nc -= 1;
        } else if (nt % 128) == 0 {
            nt /= 2;
        } else {
            break;
        }
    }
    nt = if nt + 32 > 512 { 512 } else { nt + 32 }; // warning: should not exceed thread_per_block
    TaskSchema {
        algorithm,
        protocol,
        num_channels: nc as _,
        num_threads: nt as _,
    }
}

fn upload_work(work: KernelWork) -> DeviceNonNull<mccsDevWork> {
    let mut dev_work_content = match work {
        KernelWork::Coll {
            func_index,
            mut work_elems,
        } => {
            debug_assert!(work_elems.len() <= MCCS_MAX_ELEMENTS_PER_WORK);

            let elems = unsafe {
                let elems = [MaybeUninit::zeroed(); MCCS_MAX_ELEMENTS_PER_WORK];
                let mut elems = MaybeUninit::array_assume_init(elems);
                for (idx, work_elem) in work_elems.into_iter().enumerate() {
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
                init.funcIndex = func_index;
                init.set_isLast(1);
                init.set_inFifo(0);
                init.type_ = mccsDevWorkType::mccsDevWorkTypeColl;
                init
            };

            mccsDevWork {
                header: dev_work_header,
                __bindgen_anon_1: dev_work_elems,
            }
        } // TODO
    };

    let dev_work = DeviceAlloc::new(1);
    let ptr = unsafe { DeviceNonNull::new_unchecked(dev_work.as_ptr()) };
    let _guard = std::mem::ManuallyDrop::new(dev_work);
    unsafe {
        cuda_warning!(cudaMemcpy(
            ptr.as_ptr() as _,
            &mut dev_work_content as *mut mccsDevWork as _,
            std::mem::size_of::<mccsDevWork>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        ));
    }
    ptr
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
