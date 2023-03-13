use std::mem::MaybeUninit;
use std::sync::atomic::Ordering;

use cuda_runtime_sys::{cudaEventCreate, cudaEventRecord, cudaEventQuery};
use cuda_runtime_sys::cudaStreamCreateWithFlags;
use cuda_runtime_sys::cudaMemcpyAsync;
use cuda_runtime_sys::cudaError;
use cuda_runtime_sys::cudaStreamNonBlocking;
use cuda_runtime_sys::cudaMemcpyKind::{cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost};

use crate::cuda::alloc::{DeviceHostMapped, DeviceAlloc};
use crate::cuda::ptr::DeviceNonNull;
use crate::transport::{PROTOCOL_SIMPLE, NUM_BUFFER_SLOTS};
use crate::transport::op::{TransportOp, TransportOpState};
use crate::transport::transporter::{AgentMessage, AnyResources};

use super::config::ShmLocality;
use super::resources::{ShmAgentRequest, ShmAgentReply, ShmAgentResources};

pub async fn shm_agent_connect(
    agent_request: AgentMessage,
) -> (AnyResources, AgentMessage) {
    let request = *agent_request.unwrap()
        .downcast::<ShmAgentRequest>()
        .unwrap();

    let buf = match request.locality {
        ShmLocality::Sender => request.sender_meta.buf_mut_ptr(),
        ShmLocality::Receiver => request.receiver_meta.buf_mut_ptr(),
    };
    let buf_size = request.buf_sizes[PROTOCOL_SIMPLE];
    let buf_offset = request.buf_sizes[0..PROTOCOL_SIMPLE]
        .iter()
        .copied()
        .sum();
    let host_buf = unsafe { buf.add(buf_offset) };

    let meta_sync = DeviceHostMapped::alloc(1);
    let meta_sync_ptr = unsafe {
        DeviceNonNull::new_unchecked(meta_sync.as_ptr_dev())
    };
    let device_buf = DeviceAlloc::new(buf_size);
    let dev_buf_ptr = unsafe { 
        DeviceNonNull::new_unchecked(device_buf.as_ptr()) 
    };
    
    let mut stream =  MaybeUninit::uninit();
    unsafe {
        cudaStreamCreateWithFlags(stream.as_mut_ptr(), cudaStreamNonBlocking);
    };
    let stream = unsafe { stream.assume_init() };
    let mut events = MaybeUninit::uninit_array();
    for i in 0..events.len() {
        unsafe {
            cudaEventCreate(events[i].as_mut_ptr());
        };
    }
    let events = unsafe { MaybeUninit::array_assume_init(events) };

    let agent_resources = ShmAgentResources {
        meta_sync,
        host_buf,
        device_buf,
        buf_size,
        sender_meta: request.sender_meta,
        receiver_meta: request.receiver_meta,
        step: 0,
        stream,
        events,
    };
    let boxed_resources = Box::new(agent_resources);
    let reply = ShmAgentReply {
        meta_sync: meta_sync_ptr,
        device_buf: dev_buf_ptr,
    };
    let boxed_reply = Box::new(reply);

    (boxed_resources, Some(boxed_reply))
}

pub fn shm_agent_send_progress(
    resources: &mut AnyResources, 
    op: &mut TransportOp,
) {
    let resources = resources.downcast_mut::<ShmAgentResources>().unwrap();
    if op.state == TransportOpState::Init {
        op.base = resources.step.div_ceil(op.chunk_steps as u64);
        op.posted = 0;
        op.transmitted = 0;
        op.done = 0;
        op.state = TransportOpState::InProgress;
        if op.protocol == PROTOCOL_SIMPLE {
            resources.step = op.base + op.num_steps as u64;
            op.state = TransportOpState::Completed;
            return;
        }
    }

    let step_size = resources.buf_size / NUM_BUFFER_SLOTS;
    if (op.transmitted < op.done + NUM_BUFFER_SLOTS as u64) && (op.transmitted < op.num_steps as u64) {
        let buf_slot = ((op.base + op.transmitted) % NUM_BUFFER_SLOTS as u64) as usize;
        let meta = resources.meta_sync.as_ptr_host();
        let tail = unsafe { (&*meta).tail };
        if tail > op.base + op.transmitted {
            let size = unsafe { (&*meta).slots_sizes[buf_slot as usize] };
            let offset = buf_slot * step_size;
            unsafe { 
                let device_buf = resources.device_buf.as_ptr().add(offset);
                let host_buf = resources.host_buf.add(offset);
                cudaMemcpyAsync(
                    host_buf as _,
                    device_buf as _,
                    size as usize,
                    cudaMemcpyDeviceToHost,
                    resources.stream,
                );
                cudaEventRecord(
                    resources.events[buf_slot],
                    resources.stream,
                );
                let receiver_meta = resources.receiver_meta.get_meta_mut();
                receiver_meta.slots_sizes[buf_slot] = size;
                // TODO: check whether it is equivalent to 
                // GCC's __sync_synchronize
                std::sync::atomic::fence(Ordering::SeqCst);
                op.transmitted += op.slice_steps as u64;
            }
        }
    }
    if op.done < op.transmitted {
        let buf_slot = ((op.base + op.done) % NUM_BUFFER_SLOTS as u64) as usize;
        unsafe {
            let res = cudaEventQuery(resources.events[buf_slot]);
            if res == cudaError::cudaSuccess {
                op.done += op.slice_steps as u64;
                let receiver_meta = resources.receiver_meta.get_meta_mut();
                receiver_meta.tail = op.base + op.done;
            }
            if op.done == op.num_steps as u64  {
                resources.step = op.base + op.num_steps as u64;
                op.state == TransportOpState::Completed;
            }
        }
    }
}

pub fn shm_agent_recv_progress(
    resources: &mut AnyResources, 
    op: &mut TransportOp,
) {
    let resources = resources.downcast_mut::<ShmAgentResources>().unwrap();
    if op.state == TransportOpState::Init {
        op.base = resources.step.div_ceil(op.chunk_steps as u64);
        op.posted = 0;
        op.transmitted = 0;
        op.done = 0;
        op.state == TransportOpState::InProgress;
        if op.protocol == PROTOCOL_SIMPLE {
            resources.step = op.base + op.num_steps as u64;
            op.state = TransportOpState::Completed;
            return;
        }
    }
    
    let step_size = resources.buf_size / NUM_BUFFER_SLOTS;
    if (op.transmitted < op.done + NUM_BUFFER_SLOTS as u64) && (op.transmitted < op.num_steps as u64) {
        let buf_slot = ((op.base + op.transmitted) % NUM_BUFFER_SLOTS as u64) as usize;
        let tail = unsafe { resources.receiver_meta.get_meta_mut().tail };
        if tail > op.base + op.transmitted {
            let size = unsafe { resources.receiver_meta.get_meta_mut().slots_sizes[buf_slot] };
            let offset = buf_slot * step_size;
            unsafe {
                let device_buf = resources.device_buf.as_ptr().add(offset);
                let host_buf = resources.host_buf.add(offset);
                cudaMemcpyAsync(
                    device_buf as _,
                    host_buf as _,
                    size as usize,
                    cudaMemcpyHostToDevice,
                    resources.stream,
                );
                cudaEventRecord(
                    resources.events[buf_slot],
                    resources.stream,
                );
                op.transmitted += op.slice_steps as u64;
            }
        }
    }
    if op.done < op.transmitted {
        let buf_slot = ((op.base + op.done) % NUM_BUFFER_SLOTS as u64) as usize;
        unsafe {
            let res = cudaEventQuery(resources.events[buf_slot]);
            if res == cudaError::cudaSuccess {
                op.done += op.slice_steps as u64;
                let meta = resources.meta_sync.as_ptr_host();
                (&mut *meta).tail = op.base + op.done; 
                if op.done == op.num_steps as u64 {
                    resources.step = op.base + op.num_steps as u64;
                    op.state = TransportOpState::Completed;
                }
            }
        }
    }
}