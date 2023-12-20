use std::any::Any;
use std::ffi::c_void;
use std::os::fd::RawFd;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;

use crate::transport::transporter::{ConnectHandle, ConnectHandleError};

pub type AnyNetComm = Box<dyn Any + Send>;
pub type AnyMrHandle = Box<dyn Any + Send>;

#[derive(Debug, Clone)]
pub(crate) struct NetProperties {
    pub(crate) name: String,
    // Path to the PCI device
    pub(crate) pci_path: String,
    // Unique identifier for the NIC chip. Important for
    // cards with multiple PCI functions (Physical or virtual).
    pub(crate) guid: u64,
    pub(crate) speed: u32,
    pub(crate) port: u16,
    pub(crate) latency: f32,
    pub(crate) max_comms: usize,
    pub(crate) max_recvs: usize,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub(crate) enum MrType {
    Host,
    Device,
}

#[derive(Clone, Copy)]
pub(crate) struct MemoryRegion {
    pub(crate) data: *mut c_void,
    pub(crate) size: usize,
    pub(crate) mr_type: MrType,
}

unsafe impl Send for MemoryRegion {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct NetRequestId(u32);

pub(crate) struct NetListener<NetHandle> {
    pub(crate) handle: NetHandle,
    pub(crate) recv_comm: AnyNetComm,
}

// https://github.com/NVIDIA/nccl/blob/v2.17.1-1/src/include/nccl_net.h#L87
// closeSend, closeRecv and closeListen should be implemented as Drop of the NetComm types
#[async_trait]
pub(crate) trait NetProvider: Send + Sync {
    type NetError: std::error::Error + Send + Sync + 'static;
    type NetHandle: Serialize + DeserializeOwned + Send + Sync + 'static;

    // Initialize the network transport provider.
    // Any state modifications should be achieved with interior mutability
    fn init(&self);
    // Return the number of network adapters
    fn get_num_devices(&self) -> usize;
    // Get various device properties
    fn get_properties<'a>(&'a self, device: usize) -> &'a NetProperties;
    // Create a receiving object and provide a handle to connect to it.
    // The handle will be exchanged between ranks to createa a connection
    async fn listen(&self, device: usize) -> Result<NetListener<Self::NetHandle>, Self::NetError>;
    // Connect to a handle and return a sending comm object for that peer.
    async fn connect(&self, handle: Self::NetHandle) -> Result<AnyNetComm, Self::NetError>;
    // Finalize connection establishment after remote peer has called connect.
    async fn accept(&self, listen_comm: AnyNetComm) -> Result<AnyNetComm, Self::NetError>;
    // Register memory. Comm can be either a sendComm or a recvComm.
    async fn register_mr(
        &self,
        comm: &mut AnyNetComm,
        mr: MemoryRegion,
    ) -> Result<AnyMrHandle, Self::NetError>;
    // DMA-BUF support
    async fn register_mr_dma_buf(
        &self,
        comm: &mut AnyNetComm,
        mr: MemoryRegion,
        offset: u64,
        fd: RawFd,
    ) -> Result<AnyMrHandle, Self::NetError>;
    // Deregister memory
    async fn deregister_mr(
        &self,
        comm: &mut AnyNetComm,
        handle: AnyMrHandle,
    ) -> Result<(), Self::NetError>;
    // Initiate an asynchronous send operation to a peer.
    fn initiate_send(
        &self,
        send_comm: &mut AnyNetComm,
        data: *mut c_void,
        size: usize,
        tag: u32,
        mr_handle: &AnyMrHandle,
    ) -> Option<NetRequestId>;
    // Initiate an asynchronous recv operation from a eer.
    fn initiate_recv(
        &self,
        recv_comm: &mut AnyNetComm,
        data: &[*mut c_void],
        sizes: &[usize],
        tags: &[u32],
        mr_handles: &[&AnyMrHandle],
    ) -> Option<NetRequestId>;
    // Initiate an asynchronous flush/fence to make sure all data received with
    // MemoryRegionType::Device is visible to the GPU.
    fn initiate_flush(
        &self,
        recv_comm: &mut AnyNetComm,
        data: &[*mut c_void],
        sizes: &[usize],
        mr_handle: &[&AnyMrHandle],
    ) -> Option<NetRequestId>;
    // Test whether a request is complete.
    fn test(&self, request: NetRequestId, sizes: &mut [usize]) -> bool;
}

pub(crate) struct NetListenerErased {
    pub(crate) handle: ConnectHandle,
    pub(crate) recv_comm: AnyNetComm,
}

mod private {
    pub(crate) trait Sealed {}
}

#[derive(Debug, Error)]
pub enum NetProviderError {
    #[error("Connection handle: {0}")]
    ConnectionHandle(#[from] ConnectHandleError),
    #[error("Net provider: {0}")]
    NetProvider(#[from] anyhow::Error),
}

#[async_trait]
pub(crate) trait NetProvierWrap: private::Sealed {
    fn init(&self);
    fn get_num_devices(&self) -> usize;
    fn get_properties<'a>(&'a self, device: usize) -> &'a NetProperties;
    async fn listen(&self, device: usize) -> Result<NetListenerErased, NetProviderError>;
    async fn connect(&self, handle: &ConnectHandle) -> Result<AnyNetComm, NetProviderError>;
    async fn accept(&self, listen_comm: AnyNetComm) -> Result<AnyNetComm, NetProviderError>;
    async fn register_mr(
        &self,
        comm: &mut AnyNetComm,
        mr: MemoryRegion,
    ) -> Result<AnyMrHandle, NetProviderError>;
    async fn register_mr_dma_buf(
        &self,
        comm: &mut AnyNetComm,
        mr: MemoryRegion,
        offset: u64,
        fd: RawFd,
    ) -> Result<AnyMrHandle, NetProviderError>;
    async fn deregister_mr(
        &self,
        comm: &mut AnyNetComm,
        handle: AnyMrHandle,
    ) -> Result<(), NetProviderError>;
    fn initiate_send(
        &self,
        send_comm: &mut AnyNetComm,
        data: *mut c_void,
        size: usize,
        tag: u32,
        mr_handle: &AnyMrHandle,
    ) -> Option<NetRequestId>;
    fn initiate_recv(
        &self,
        recv_comm: &mut AnyNetComm,
        data: &[*mut c_void],
        sizes: &[usize],
        tags: &[u32],
        mr_handles: &[&AnyMrHandle],
    ) -> Option<NetRequestId>;
    fn initiate_flush(
        &self,
        recv_comm: &mut AnyNetComm,
        data: &[*mut c_void],
        sizes: &[usize],
        mr_handle: &[&AnyMrHandle],
    ) -> Option<NetRequestId>;
    fn test(&self, request: NetRequestId, sizes: &mut [usize]) -> bool;
}

impl<T: NetProvider> private::Sealed for T {}

#[async_trait]
impl<T: NetProvider> NetProvierWrap for T {
    #[inline]
    fn init(&self) {
        <Self as NetProvider>::init(self);
    }
    #[inline]
    fn get_num_devices(&self) -> usize {
        <Self as NetProvider>::get_num_devices(self)
    }
    #[inline]
    fn get_properties<'a>(&'a self, device: usize) -> &'a NetProperties {
        <Self as NetProvider>::get_properties(self, device)
    }
    #[inline]
    async fn listen(&self, device: usize) -> Result<NetListenerErased, NetProviderError> {
        let listener = <Self as NetProvider>::listen(self, device)
            .await
            .map_err(anyhow::Error::new)?;
        let serialized_handle = ConnectHandle::serialize_from(listener.handle)?;
        let erased = NetListenerErased {
            handle: serialized_handle,
            recv_comm: listener.recv_comm,
        };
        Ok(erased)
    }
    #[inline]
    async fn connect(&self, handle: &ConnectHandle) -> Result<AnyNetComm, NetProviderError> {
        let handle = handle.deserialize_to::<<Self as NetProvider>::NetHandle>()?;
        let send_comm = <Self as NetProvider>::connect(self, handle)
            .await
            .map_err(anyhow::Error::new)?;
        Ok(send_comm)
    }
    #[inline]
    async fn accept(&self, listen_comm: AnyNetComm) -> Result<AnyNetComm, NetProviderError> {
        let recv_comm = <Self as NetProvider>::accept(self, listen_comm)
            .await
            .map_err(anyhow::Error::new)?;
        Ok(recv_comm)
    }
    #[inline]
    async fn register_mr(
        &self,
        comm: &mut AnyNetComm,
        mr: MemoryRegion,
    ) -> Result<AnyMrHandle, NetProviderError> {
        let mr_handle = <Self as NetProvider>::register_mr(self, comm, mr)
            .await
            .map_err(anyhow::Error::new)?;
        Ok(mr_handle)
    }
    #[inline]
    async fn register_mr_dma_buf(
        &self,
        comm: &mut AnyNetComm,
        mr: MemoryRegion,
        offset: u64,
        fd: RawFd,
    ) -> Result<AnyMrHandle, NetProviderError> {
        let mr_handle = <Self as NetProvider>::register_mr_dma_buf(self, comm, mr, offset, fd)
            .await
            .map_err(anyhow::Error::new)?;
        Ok(mr_handle)
    }
    #[inline]
    async fn deregister_mr(
        &self,
        comm: &mut AnyNetComm,
        handle: AnyMrHandle,
    ) -> Result<(), NetProviderError> {
        <Self as NetProvider>::deregister_mr(self, comm, handle)
            .await
            .map_err(anyhow::Error::new)?;
        Ok(())
    }
    #[inline]
    fn initiate_send(
        &self,
        send_comm: &mut AnyNetComm,
        data: *mut c_void,
        size: usize,
        tag: u32,
        mr_handle: &AnyMrHandle,
    ) -> Option<NetRequestId> {
        <Self as NetProvider>::initiate_send(self, send_comm, data, size, tag, mr_handle)
    }
    #[inline]
    fn initiate_recv(
        &self,
        recv_comm: &mut AnyNetComm,
        data: &[*mut c_void],
        sizes: &[usize],
        tags: &[u32],
        mr_handles: &[&AnyMrHandle],
    ) -> Option<NetRequestId> {
        <Self as NetProvider>::initiate_recv(self, recv_comm, data, sizes, tags, mr_handles)
    }
    #[inline]
    fn initiate_flush(
        &self,
        recv_comm: &mut AnyNetComm,
        data: &[*mut c_void],
        sizes: &[usize],
        mr_handle: &[&AnyMrHandle],
    ) -> Option<NetRequestId> {
        <Self as NetProvider>::initiate_flush(self, recv_comm, data, sizes, mr_handle)
    }
    #[inline]
    fn test(&self, request: NetRequestId, sizes: &mut [usize]) -> bool {
        <Self as NetProvider>::test(self, request, sizes)
    }
}