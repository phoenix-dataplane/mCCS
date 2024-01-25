use crate::checked_cuda;
use crate::DevicePtr;
use crate::Error;
use crate::MccsCommunicatorHandle;
use crate::MCCS_CTX;
use crate::MCCS_STREAM_SYNC;
use cuda_runtime_sys::cudaStreamWaitEvent;
use cuda_runtime_sys::{cudaEventRecord, cudaStream_t};
use ipc::mccs::command::AllGather;
use ipc::mccs::command::AllReduce;
use ipc::mccs::command::AllReduceDataType;
use ipc::mccs::command::AllReduceOpType;
use ipc::mccs::dp;

pub fn all_gather(
    comm: MccsCommunicatorHandle,
    send_buf: DevicePtr,
    recv_buf: DevicePtr,
    size: usize,
    stream: cudaStream_t,
) -> Result<(), Error> {
    unsafe {
        let user_event = MCCS_STREAM_SYNC.with_borrow(|sync| *sync.get(&stream).unwrap());
        checked_cuda!(cudaEventRecord(user_event, stream));
    };
    let op = AllGather {
        comm: comm.comm_handle,
        send_buf: send_buf.backup_mem,
        recv_buf: recv_buf.backup_mem,
        size,
        user_stream: stream as usize,
    };
    MCCS_CTX.with(move |ctx| {
        let mut sent = false;
        while !sent {
            ctx.service
                .enqueue_wr_with(|ptr, _count| unsafe {
                    ptr.cast::<dp::WorkRequest>()
                        .write(dp::WorkRequest::AllGather(op));
                    sent = true;
                    1
                })
                .expect("channel to backend corrupted");
        }

        let mut recv = false;
        while !recv {
            ctx.service
                .dequeue_wc_with(|ptr, count| unsafe {
                    for i in 0..count {
                        let c = ptr.add(i).cast::<dp::WorkCompletion>().read();
                        match c {
                            dp::WorkCompletion::AllGather => {
                                recv = true;
                            }
                            _ => {
                                panic!("unexpected work completion: {:?}", c);
                            }
                        }
                    }
                    count
                })
                .map_err(Error::Service)?;
        }
        Ok(()) as Result<(), Error>
    })?;

    unsafe {
        checked_cuda!(cudaStreamWaitEvent(stream, comm.backend_event, 0));
    }

    Ok(())
}

pub fn all_reduce(
    comm: MccsCommunicatorHandle,
    send_buf: DevicePtr,
    recv_buf: DevicePtr,
    size: usize,
    data_type: AllReduceDataType,
    op_type: AllReduceOpType,
    stream: cudaStream_t,
) -> Result<(), Error> {
    unsafe {
        let user_event = MCCS_STREAM_SYNC.with_borrow(|sync| *sync.get(&stream).unwrap());
        checked_cuda!(cudaEventRecord(user_event, stream));
    };
    let op = AllReduce {
        comm: comm.comm_handle,
        send_buf: send_buf.backup_mem,
        recv_buf: recv_buf.backup_mem,
        size,
        data_type,
        op_type,
        user_stream: stream as usize,
    };

    MCCS_CTX.with(move |ctx| {
        let mut sent = false;
        while !sent {
            ctx.service
                .enqueue_wr_with(|ptr, _count| unsafe {
                    ptr.cast::<dp::WorkRequest>()
                        .write(dp::WorkRequest::AllReduce(op));
                    sent = true;
                    1
                })
                .expect("channel to backend corrupted");
        }

        let mut recv = false;
        while !recv {
            ctx.service
                .dequeue_wc_with(|ptr, count| unsafe {
                    for i in 0..count {
                        let c = ptr.add(i).cast::<dp::WorkCompletion>().read();
                        match c {
                            dp::WorkCompletion::AllReduce => {
                                recv = true;
                            }
                            _ => {
                                panic!("unexpected work completion: {:?}", c);
                            }
                        }
                    }
                    count
                })
                .map_err(Error::Service)?;
        }
        Ok(()) as Result<(), Error>
    })?;

    unsafe {
        checked_cuda!(cudaStreamWaitEvent(stream, comm.backend_event, 0));
    }

    Ok(())
}
