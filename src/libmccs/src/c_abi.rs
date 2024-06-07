use std::{
    ffi::{c_char, c_void},
    net::IpAddr,
    str::FromStr,
    sync::{Mutex, OnceLock},
};

use cuda_runtime_sys::{cudaGetDevice, cudaStream_t};

use crate::{all_gather, DevicePtr, MccsCommunicatorHandle};

static REVERSE_MAPPING: OnceLock<Mutex<Vec<DevicePtr>>> = OnceLock::new();

#[repr(C)]
pub struct UniqueId([c_char; 128]);

pub type NcclCommType = *mut c_void;

#[no_mangle]
#[allow(non_snake_case)]
pub fn ncclGetUniqueId(id: *mut UniqueId) -> i32 {
    eprintln!("Debug: ncclGetUniqueId");
    unsafe {
        let id = id.as_mut().unwrap();
        id.0 = [0; 128];
        id.0[0] = 42;
    }
    return 0;
}

#[no_mangle]
#[allow(non_snake_case)]
pub fn ncclCommInitRank(
    comm: *mut NcclCommType,
    nranks: i32,
    commId: *const UniqueId,
    rank: i32,
) -> i32 {
    eprintln!(
        "Debug: ncclCommInitRank {:?} {:?} {:?} {:?}",
        comm, nranks, commId, rank
    );
    match crate::init_communicator_rank(
        42,
        commId as usize,
        nranks as usize,
        get_device(),
        IpAddr::from_str("127.0.0.1").unwrap(),
    ) {
        Ok(comm_handle) => {
            let comm_box = Box::new(comm_handle);
            eprintln!("Debug: ncclCommInitRank assigning");
            unsafe {
                *comm = Box::into_raw(comm_box) as *mut c_void;
                eprintln!("Debug: ncclCommInitRank assigned = {:?}", *comm)
            }
            eprintln!("Debug: ncclCommInitRank done");
            0
        }
        Err(_) => -1,
    }
}

#[no_mangle]
#[allow(non_snake_case)]
pub fn ncclCommDestroy(_comm: NcclCommType) -> i32 {
    eprintln!("Debug: ncclCommDestroy");
    return 0;
}

#[no_mangle]
#[allow(non_snake_case)]
pub fn ncclCommGetAsyncError(_comm: NcclCommType, error: *mut i32) -> i32 {
    // eprintln!("Debug ncclCommGetAsyncError");
    unsafe {
        *error = 0;
    }
    return 0;
}

#[no_mangle]
#[allow(non_snake_case)]
pub fn ncclGroupStart() -> i32 {
    eprintln!("Debug: ncclGroupStart");
    return 0;
}

#[no_mangle]
#[allow(non_snake_case)]
pub fn ncclGroupEnd() -> i32 {
    eprintln!("Debug: ncclGroupEnd");
    return 0;
}

#[no_mangle]
#[allow(non_snake_case)]
pub fn ncclAllGather(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    _datatype: i32, // ncclDataType_t, enum
    comm: NcclCommType,
    stream: cudaStream_t,
) -> i32 {
    // eprintln!("Debug: ncclAllGather: sendbuff={:?}, recvbuff={:?}, count={:?}, datatype={:?} comm={:?}, stream={:?}", sendbuff, recvbuff, count, _datatype, comm, stream);
    if crate::register_stream(get_device(), stream).is_err() {
        eprintln!("Error: ncclAllGather");
        return -1;
    }
    let handle = unsafe { *(comm as *mut MccsCommunicatorHandle) };
    let sendbuff = lookup_global_ptr_mapping(sendbuff).unwrap();
    let recvbuff = lookup_global_ptr_mapping(recvbuff).unwrap();
    match all_gather(handle, sendbuff, recvbuff, count, stream) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
#[allow(non_snake_case)]
pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32 {
    eprintln!("Debug: cudaMalloc");
    match crate::cuda_malloc(get_device(), size) {
        Ok(ptr) => {
            unsafe { *dev_ptr = ptr.ptr };
            push_global_ptr_mapping(ptr);
            0
        }
        Err(_) => {
            unsafe { *dev_ptr = std::ptr::null_mut() };
            -1
        }
    }
}

#[no_mangle]
#[allow(non_snake_case)]
pub fn cudaFree(_ptr: *mut c_void) {
    eprintln!("Debug: cudaFree");
    let ptr = lookup_global_ptr_mapping(_ptr).unwrap();
    crate::cuda_free(ptr.backup_mem);
}

fn push_global_ptr_mapping(ptr: DevicePtr) {
    eprintln!("Debug: push_global_ptr_mapping: {:?}", ptr);
    REVERSE_MAPPING
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .push(ptr);
}

fn lookup_global_ptr_mapping(ptr: *const c_void) -> Option<DevicePtr> {
    let list = REVERSE_MAPPING
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    for i in list.iter() {
        if ptr >= i.ptr && ptr < unsafe { i.ptr.add(i.backup_mem.len) } {
            let mut new_ptr = *i;
            new_ptr.backup_mem.offset = ptr as usize - i.ptr as usize;
            new_ptr.ptr = ptr as *mut c_void;
            return Some(new_ptr);
        }
    }
    None
}

fn get_device() -> i32 {
    let mut device_idx = 0;
    let r = unsafe { cudaGetDevice(&mut device_idx as *mut _) };
    if r != cuda_runtime_sys::cudaError::cudaSuccess {
        eprintln!("Error: get_device: {:?}", r);
    }
    device_idx
}
