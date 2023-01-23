use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use thiserror::Error;

use cuda_runtime_sys::{cudaHostRegister, cudaHostRegisterMapped, cudaHostGetDevicePointer};

use crate::transport::buffer::{SendBufMeta, RecvBufMeta, TransportBuffer};
use crate::transport::registry::TransportSetupRegistry;
use crate::transport::connector::{ConnectorIdentifier, TransportConnector, ConnectionInfo};

use super::resources::{HostMemTptResource, HostMemTptSetupSender, HostMemTptSetupReceiver};
use super::config::{HostMemTptConfig, MemLocality};

#[derive(Error, Clone, Debug)]
pub enum Error {
    #[error("Pending remote setup")]
    PendingRemote,
}

pub struct HostMemTptEndpoint {
    pub resources: HostMemTptResource,
    pub connector: TransportConnector,
}

pub fn hmem_sender_setup(
    registry: &TransportSetupRegistry,
    identifier: ConnectorIdentifier,
    config: &HostMemTptConfig,
) {
    let mut buf_size = std::mem::size_of::<SendBufMeta>();
    if config.locality == MemLocality::SenderSide {
        buf_size += config.buff_sizes.iter().sum::<usize>();
    }

    let send_buf_meta = SendBufMeta::new();
    let send_buf = TransportBuffer::new(
        send_buf_meta,
        buf_size,
        std::mem::align_of::<SendBufMeta>()
    );
    let sender_setup = HostMemTptSetupSender {
        host_mem: Arc::new(send_buf),
        config: config.clone(),
    };
    registry.insert_hmem_sender(identifier, sender_setup).unwrap();
}

pub fn hmem_receiver_setup(
    registry: &TransportSetupRegistry,
    identifier: ConnectorIdentifier,
    config: &HostMemTptConfig,
) {
    let mut buf_size = std::mem::size_of::<RecvBufMeta>();
    if config.locality == MemLocality::ReceiverSide {
        buf_size += config.buff_sizes.iter().sum::<usize>();
    }
    let recv_buf_meta = RecvBufMeta::new();
    let recv_buf = TransportBuffer::new(
        recv_buf_meta,
        buf_size,
        std::mem::align_of::<RecvBufMeta>(),
    );
    let receiver_setup = HostMemTptSetupReceiver {
        host_mem: Arc::new(recv_buf),
        config: config.clone(),
    };
    registry.insert_hmem_receiver(identifier, receiver_setup).unwrap();
}

pub fn hmem_sender_connect(
    registry: &TransportSetupRegistry,
    identifier: &ConnectorIdentifier,
) -> Result<HostMemTptEndpoint, Error> {
    let sender = registry.hmem_senders.get(identifier).unwrap();
    let receiver = registry.hmem_receivers.get(identifier).ok_or(Error::PendingRemote)?;

    let send_buf_device_ptr = unsafe { 
        let buf_ptr = sender.host_mem.buf_mut_ptr() as *mut c_void;
        let buf_size = sender.host_mem.buf_size();
        cudaHostRegister(buf_ptr, buf_size, cudaHostRegisterMapped);
        let mut buf_device_ptr: *mut c_void = std::ptr::null_mut();
        cudaHostGetDevicePointer(&mut buf_device_ptr as *mut *mut c_void, buf_ptr, 0);
        buf_device_ptr as *mut SendBufMeta
    };
    let receiver_buf_device_ptr = unsafe { 
        let buf_ptr = receiver.host_mem.buf_mut_ptr() as *mut c_void;
        let buf_size = receiver.host_mem.buf_size();
        cudaHostRegister(buf_ptr, buf_size, cudaHostRegisterMapped);
        let mut buf_device_ptr: *mut c_void = std::ptr::null_mut();
        cudaHostGetDevicePointer(&mut buf_device_ptr as *mut *mut c_void, buf_ptr, 0);
        buf_device_ptr as *mut RecvBufMeta
    };

    let sender = HostMemTptResource {
        sender_host_mem: Arc::clone(&sender.host_mem),
        sender_device_mem: send_buf_device_ptr,
        receiver_host_mem: Arc::clone(&receiver.host_mem),
        receiver_device_mem: receiver_buf_device_ptr,
    };
    let buf = if receiver.config.locality == MemLocality::SenderSide {
        // send_buf_device_ptr.add(1) as *mut u8
        sender.sender_host_mem.buf_mut_ptr()
    }
    else {
        // receiver_buf_device_ptr.add(1) as *mut u8
        sender.receiver_host_mem.buf_mut_ptr()
    };
    let buf = NonNull::new(buf).unwrap();

    // let head = unsafe { &mut (*send_buf_device_ptr).head as *mut _ };
    // let tail = unsafe { &mut (*receiver_buf_device_ptr).tail as *mut _ };
    let head = unsafe { &mut sender.sender_host_mem.get_meta_mut().head as *mut _ };
    let tail = unsafe { &mut sender.receiver_host_mem.get_meta_mut().tail as *mut _ };

    let info = ConnectionInfo {
        bufs: vec![buf],
        head,
        tail,
        _direct: false,
        _shared: false,
        _ptr_exchange: std::ptr::null_mut(),
        _red_op_arg_exchange: std::ptr::null_mut(),
        _slots_sizes: std::ptr::null_mut(),
        _slots_offsets: std::ptr::null_mut(),
        _step: 0,
        _ll_last_cleaning: 0,
    };
    let conn = TransportConnector {
        info,
    };

    let endpoint = HostMemTptEndpoint{
        resources: sender,
        connector: conn,
    };
    
    Ok(endpoint)
}

pub fn hmem_receiver_connect(
    registry: &TransportSetupRegistry,
    identifier: &ConnectorIdentifier,
) -> Result<HostMemTptEndpoint, Error> {
    let receiver = registry.hmem_receivers.get(identifier).unwrap();
    let sender = registry.hmem_senders.get(identifier).ok_or(Error::PendingRemote)?;

    let send_buf_device_ptr = unsafe { 
        let buf_ptr = sender.host_mem.buf_mut_ptr() as *mut c_void;
        let buf_size = sender.host_mem.buf_size();
        cudaHostRegister(buf_ptr, buf_size, cudaHostRegisterMapped);
        let mut buf_device_ptr: *mut c_void = std::ptr::null_mut();
        cudaHostGetDevicePointer(&mut buf_device_ptr as *mut *mut c_void, buf_ptr, 0);
        buf_device_ptr as *mut SendBufMeta
    };
    let receiver_buf_device_ptr = unsafe { 
        let buf_ptr = receiver.host_mem.buf_mut_ptr() as *mut c_void;
        let buf_size = receiver.host_mem.buf_size();
        cudaHostRegister(buf_ptr, buf_size, cudaHostRegisterMapped);
        let mut buf_device_ptr: *mut c_void = std::ptr::null_mut();
        cudaHostGetDevicePointer(&mut buf_device_ptr as *mut *mut c_void, buf_ptr, 0);
        buf_device_ptr as *mut RecvBufMeta
    };

    let receiver = HostMemTptResource {
        sender_host_mem: Arc::clone(&sender.host_mem),
        sender_device_mem: send_buf_device_ptr,
        receiver_host_mem: Arc::clone(&receiver.host_mem),
        receiver_device_mem: receiver_buf_device_ptr,
    };

    let buf = if sender.config.locality == MemLocality::SenderSide {
        receiver.sender_host_mem.buf_mut_ptr()
    }
    else {
        receiver.receiver_host_mem.buf_mut_ptr()
    };
    let buf = NonNull::new(buf).unwrap();

    let head = unsafe { &mut receiver.sender_host_mem.get_meta_mut().head as *mut _ };
    let tail = unsafe { &mut receiver.receiver_host_mem.get_meta_mut().tail as *mut _ };

    let info = ConnectionInfo {
        bufs: vec![buf],
        head,
        tail,
        _direct: false,
        _shared: false,
        _ptr_exchange: std::ptr::null_mut(),
        _red_op_arg_exchange: std::ptr::null_mut(),
        _slots_sizes: std::ptr::null_mut(),
        _slots_offsets: std::ptr::null_mut(),
        _step: 0,
        _ll_last_cleaning: 0,
    };
    let conn = TransportConnector {
        info,
    };

    let endpoint = HostMemTptEndpoint{
        resources: receiver,
        connector: conn,
    };
    Ok(endpoint)
}