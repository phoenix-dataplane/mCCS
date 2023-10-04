use std::ffi::c_void;
use std::{collections::VecDeque, mem::MaybeUninit};

use collectives_sys::{
    mccsDevComm, mccsDevWork, mccsDevWorkElem, mccsDevWorkHeader, mccsDevWorkType,
    mccsDevWork__bindgen_ty_1, mccsKernel_AllGather_RING_SIMPLE_Sum_int8_t,
};
use cuda_runtime_sys::{cudaEventRecord, cudaLaunchKernel, cudaMemcpy, cudaMemcpyKind};

use super::task::{CollTask, TaskAlgorithm, TaskProtocol, TaskSchema};
use crate::{
    comm::Communicator,
    cuda::{alloc::DeviceAlloc, ptr::DeviceNonNull},
};

#[derive(Clone)]
pub struct WorkElemColl {
    num_warps: u8,
    send_buf: DeviceNonNull<c_void>,
    recv_buf: DeviceNonNull<c_void>,

    count: usize,
    root: u32,
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
    pub work_queue: VecDeque<KernelWork>,
}

pub struct KernelPlan {
    kernel_fn: *const c_void,
    dev_work_head: DeviceNonNull<mccsDevWork>,

    channel_upper_bound: u32,
    channel_count: u32,
    channel_mask: u64,

    thread_per_block: usize,
}

// TODO: following implementations are temporary
// used for a single coll task launch
// need to be replaced by following NCCL's enqueue system

impl Communicator {
    pub fn pre_launch_schedule(&mut self) {
        self.schedule_coll_tasks();
        self.finalize_plan();
    }

    pub fn schedule_coll_tasks(&mut self) {
        let coll_task = self.task_queue.coll_queue.pop_front().unwrap();
        let work = compute_coll_work(&coll_task);
        enqueue_coll_work_to_chans(&work, self);
    }

    pub fn finalize_plan(&mut self) {
        let ptr = mccsKernel_AllGather_RING_SIMPLE_Sum_int8_t;
        let dev_work = self.upload_work();
        let plan = KernelPlan {
            kernel_fn: ptr as _,
            dev_work_head: dev_work,

            channel_count: 1,
            channel_upper_bound: 1,
            channel_mask: 1,

            thread_per_block: 512,
        };
        self.unlaunched_plans.push_back(plan);
    }

    pub fn upload_work(&mut self) -> DeviceNonNull<mccsDevWork> {
        let work = self
            .plan_schedule
            .get_mut(&0)
            .unwrap()
            .work_queue
            .pop_front()
            .unwrap();
        let mut dev_work_content = match work {
            KernelWork::Coll {
                func_index: _,
                mut work_elems,
            } => {
                let work_elem = work_elems.pop().unwrap();
                let mut dev_work_elem = unsafe {
                    let elem = MaybeUninit::<mccsDevWorkElem>::zeroed();
                    elem.assume_init()
                };
                dev_work_elem.bid = 0;
                dev_work_elem.nChannels = work_elem.num_channels;
                dev_work_elem.nWarps = work_elem.num_warps;
                dev_work_elem.count = work_elem.count;
                dev_work_elem.root = work_elem.root;
                dev_work_elem.recvbuff = work_elem.recv_buf.as_ptr();
                dev_work_elem.sendbuff = work_elem.send_buf.as_ptr();
                dev_work_elem.set_isUsed(1);

                let elems = unsafe {
                    let elems = [MaybeUninit::zeroed(); 10];
                    let mut elems = MaybeUninit::array_assume_init(elems);
                    elems[0] = dev_work_elem;
                    elems
                };
                let dev_work_elems = mccsDevWork__bindgen_ty_1 { elems };
                let dev_work_header = unsafe {
                    let uninit = MaybeUninit::<mccsDevWorkHeader>::zeroed();
                    let mut init = uninit.assume_init();
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
            cudaMemcpy(
                ptr.as_ptr() as _,
                &mut dev_work_content as *mut mccsDevWork as _,
                std::mem::size_of::<mccsDevWork>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
        }
        ptr
    }

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
            cudaLaunchKernel(
                plan.kernel_fn,
                grid,
                block,
                args.as_mut_ptr(),
                0,
                self.stream,
            );
            cudaEventRecord(self.event, self.stream);
        }
    }
}

pub fn compute_coll_work(task: &CollTask) -> WorkElemColl {
    let schema = get_task_schema(task);
    let num_wraps = schema.num_threads / 32;
    WorkElemColl {
        num_warps: num_wraps as _,
        send_buf: task.send_buf.cast(),
        recv_buf: task.recv_buf.cast(),
        count: task.count,
        root: task.root as _,
        bid: 0,
        num_channels: schema.num_channels as _,
    }
}

pub fn enqueue_coll_work_to_chans(elem: &WorkElemColl, comm: &mut Communicator) {
    let schedule = comm.plan_schedule.get_mut(&0).unwrap();
    let work = KernelWork::Coll {
        func_index: 0,
        work_elems: vec![elem.clone()],
    };
    schedule.work_queue.push_back(work);
}

pub fn get_task_schema(task: &CollTask) -> TaskSchema {
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
    nt += 32;
    TaskSchema {
        algorithm,
        protocol,
        num_channels: nc as _,
        num_threads: 128, //FIXME
    }
}
