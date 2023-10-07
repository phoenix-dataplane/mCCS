use std::mem::MaybeUninit;
use std::sync::Arc;

use async_trait::async_trait;
use memoffset::raw_field;

use crate::comm::CommProfile;
use crate::cuda::alloc::DeviceHostMapped;
use crate::cuda::ptr::DeviceNonNull;
use crate::transport::catalog::TransportCatalog;
use crate::transport::channel::{PeerConnId, PeerConnInfo};
use crate::transport::meta::{RecvBufMeta, SendBufMeta};
use crate::transport::op::TransportOp;
use crate::transport::transporter::{AgentMessage, TransportAgentId};
use crate::transport::transporter::{
    AnyResources, ConnectInfo, TransportConnect, TransportSetup, Transporter,
};
use crate::transport::{NUM_PROTOCOLS, PROTOCOL_SIMPLE};

use super::agent::{shm_agent_connect, shm_agent_recv_progress, shm_agent_send_progress};
use super::buffer::TransportBuffer;
use super::config::{ShmConfig, ShmLocality};
use super::resources::{
    ShmAgentReply, ShmAgentRequest, ShmConnectedResources, ShmRecvSetupResources,
    ShmSendSetupResources,
};

pub struct ShmTransporter;

#[async_trait]
impl Transporter for ShmTransporter {
    fn send_setup(
        &self,
        profile: &CommProfile,
        _conn_id: &PeerConnId,
        catalog: &TransportCatalog,
    ) -> TransportSetup {
        let mut buf_size = std::mem::size_of::<SendBufMeta>();
        let config = catalog.get_config::<ShmConfig>("ShmTransport").unwrap();
        if config.locality == ShmLocality::Sender {
            buf_size += profile.buff_sizes.iter().sum::<usize>();
        }
        buf_size += 8 * 1024 * 1024;
        let send_buf_meta = SendBufMeta::new();
        let send_buf =
            TransportBuffer::new(send_buf_meta, buf_size, std::mem::align_of::<SendBufMeta>());

        let sender_resources = ShmSendSetupResources {
            buf: Arc::new(send_buf),
            buf_sizes: profile.buff_sizes,
            locality: config.locality,
            use_memcpy: config.use_memcpy_send,
        };
        let connect_info = sender_resources.clone();

        TransportSetup::Setup {
            peer_connect_info: Box::new(connect_info),
            setup_resources: Some(Box::new(sender_resources)),
        }
    }

    fn send_connect(
        &self,
        _conn_id: &PeerConnId,
        connect_info: ConnectInfo,
        setup_resources: Option<AnyResources>,
    ) -> TransportConnect {
        let receiver = *connect_info.downcast::<ShmRecvSetupResources>().unwrap();
        let sender = *setup_resources
            .unwrap()
            .downcast::<ShmSendSetupResources>()
            .unwrap();

        let send_buf_mapped =
            DeviceHostMapped::register(sender.buf.meta_mut_ptr() as *mut u8, sender.buf.size())
                .unwrap()
                .cast::<SendBufMeta>();
        let recv_buf_mapped =
            DeviceHostMapped::register(receiver.buf.meta_mut_ptr() as *mut u8, receiver.buf.size())
                .unwrap()
                .cast::<RecvBufMeta>();

        let shm_resources = ShmConnectedResources {
            sender_buf: sender.buf,
            sender_buf_dev: send_buf_mapped,
            receiver_buf: receiver.buf,
            receiver_buf_dev: recv_buf_mapped,
            buf_sizes: sender.buf_sizes,
            locality: sender.locality,
        };
        if sender.use_memcpy {
            let agent_request = ShmAgentRequest {
                locality: sender.locality,
                buf_sizes: sender.buf_sizes,
                sender_meta: Arc::clone(&shm_resources.sender_buf),
                receiver_meta: Arc::clone(&shm_resources.receiver_buf),
            };
            TransportConnect::PreAgentCb {
                agent_request: Some(Box::new(agent_request)),
                transport_resources: Some(Box::new(shm_resources)),
            }
        } else {
            let head = unsafe {
                let ptr = raw_field!(shm_resources.sender_buf_dev.as_ptr_dev(), SendBufMeta, head);
                DeviceNonNull::new_unchecked(ptr as _)
            };
            let tail = unsafe {
                let ptr = raw_field!(
                    shm_resources.receiver_buf_dev.as_ptr_dev(),
                    RecvBufMeta,
                    tail
                );
                DeviceNonNull::new_unchecked(ptr as _)
            };

            let slots_size = if receiver.use_memcpy {
                let meta = shm_resources.receiver_buf_dev.as_ptr_dev();
                let sizes_ptr = raw_field!(meta, RecvBufMeta, slots_sizes);
                let dev_ptr = unsafe { DeviceNonNull::new_unchecked(sizes_ptr as _) };
                Some(dev_ptr)
            } else {
                None
            };

            let mut bufs = MaybeUninit::uninit_array();
            let mut buf_curr = match sender.locality {
                ShmLocality::Sender => unsafe {
                    shm_resources.sender_buf_dev.as_ptr_dev().add(1) as *mut u8
                },
                ShmLocality::Receiver => unsafe {
                    shm_resources.receiver_buf_dev.as_ptr_dev().add(1) as *mut u8
                },
            };
            for (proto, buf) in bufs.iter_mut().enumerate().take(NUM_PROTOCOLS) {
                unsafe {
                    let dev_ptr = DeviceNonNull::new_unchecked(buf_curr);
                    buf.write(dev_ptr);
                    buf_curr = buf_curr.add(sender.buf_sizes[proto]);
                }
            }
            let bufs = unsafe { MaybeUninit::array_assume_init(bufs) };

            let info = PeerConnInfo {
                bufs,
                head,
                tail,
                slots_size,
            };
            TransportConnect::Connect {
                conn_info: info,
                transport_resources: Box::new(shm_resources),
            }
        }
    }

    fn send_connect_agent_callback(
        &self,
        _conn_id: &PeerConnId,
        agent_reply: AgentMessage,
        transport_resources: Option<AnyResources>,
    ) -> TransportConnect {
        let reply = *agent_reply.unwrap().downcast::<ShmAgentReply>().unwrap();
        let resources = transport_resources
            .unwrap()
            .downcast::<ShmConnectedResources>()
            .unwrap();

        let meta_sync_dev = reply.meta_sync.as_ptr();
        let head = unsafe {
            let ptr = raw_field!(resources.sender_buf_dev.as_ptr_dev(), SendBufMeta, head);
            DeviceNonNull::new_unchecked(ptr as _)
        };
        let tail = unsafe {
            let ptr = raw_field!(meta_sync_dev, RecvBufMeta, tail);
            DeviceNonNull::new_unchecked(ptr as _)
        };
        let sizes_ptr = raw_field!(meta_sync_dev, RecvBufMeta, slots_sizes);
        let sizes_dev_ptr = unsafe { DeviceNonNull::new_unchecked(sizes_ptr as _) };

        let mut bufs = MaybeUninit::uninit_array();
        let mut buf_curr = match resources.locality {
            ShmLocality::Sender => unsafe {
                resources.sender_buf_dev.as_ptr_dev().add(1) as *mut u8
            },
            ShmLocality::Receiver => unsafe {
                resources.receiver_buf_dev.as_ptr_dev().add(1) as *mut u8
            },
        };
        for (proto, buf) in bufs.iter_mut().enumerate().take(NUM_PROTOCOLS) {
            unsafe {
                let dev_ptr = DeviceNonNull::new_unchecked(buf_curr);
                buf.write(dev_ptr);
                buf_curr = buf_curr.add(resources.buf_sizes[proto]);
            }
        }
        bufs[PROTOCOL_SIMPLE].write(reply.device_buf);
        let bufs = unsafe { MaybeUninit::array_assume_init(bufs) };

        let info = PeerConnInfo {
            bufs,
            head,
            tail,
            slots_size: Some(sizes_dev_ptr),
        };
        TransportConnect::Connect {
            conn_info: info,
            transport_resources: resources,
        }
    }

    fn recv_setup(
        &self,
        profile: &CommProfile,
        _conn_id: &PeerConnId,
        catalog: &TransportCatalog,
    ) -> TransportSetup {
        let mut buf_size = std::mem::size_of::<RecvBufMeta>();
        let config = catalog.get_config::<ShmConfig>("ShmTransport").unwrap();
        if config.locality == ShmLocality::Receiver {
            buf_size += profile.buff_sizes.iter().sum::<usize>();
        }
        buf_size += 8 * 1024 * 1024;
        let recv_buf_meta = RecvBufMeta::new();
        let recv_buf =
            TransportBuffer::new(recv_buf_meta, buf_size, std::mem::align_of::<RecvBufMeta>());

        let sender_resources = ShmRecvSetupResources {
            buf: Arc::new(recv_buf),
            buf_sizes: profile.buff_sizes,
            locality: config.locality,
            use_memcpy: config.use_memcpy_send,
        };
        let connect_info = sender_resources.clone();

        TransportSetup::Setup {
            peer_connect_info: Box::new(connect_info),
            setup_resources: Some(Box::new(sender_resources)),
        }
    }

    fn recv_connect(
        &self,
        _conn_id: &PeerConnId,
        connect_info: ConnectInfo,
        setup_resources: Option<AnyResources>,
    ) -> TransportConnect {
        let sender = *connect_info.downcast::<ShmSendSetupResources>().unwrap();
        let receiver = *setup_resources
            .unwrap()
            .downcast::<ShmRecvSetupResources>()
            .unwrap();

        let send_buf_mapped =
            DeviceHostMapped::register(sender.buf.meta_mut_ptr() as *mut u8, sender.buf.size())
                .unwrap()
                .cast::<SendBufMeta>();
        let recv_buf_mapped =
            DeviceHostMapped::register(receiver.buf.meta_mut_ptr() as *mut u8, receiver.buf.size())
                .unwrap()
                .cast::<RecvBufMeta>();

        let shm_resources = ShmConnectedResources {
            sender_buf: sender.buf,
            sender_buf_dev: send_buf_mapped,
            receiver_buf: receiver.buf,
            receiver_buf_dev: recv_buf_mapped,
            buf_sizes: sender.buf_sizes,
            locality: sender.locality,
        };
        if receiver.use_memcpy {
            let agent_request = ShmAgentRequest {
                locality: receiver.locality,
                buf_sizes: receiver.buf_sizes,
                sender_meta: Arc::clone(&shm_resources.sender_buf),
                receiver_meta: Arc::clone(&shm_resources.receiver_buf),
            };
            TransportConnect::PreAgentCb {
                agent_request: Some(Box::new(agent_request)),
                transport_resources: Some(Box::new(shm_resources)),
            }
        } else {
            let head = unsafe {
                let ptr = raw_field!(shm_resources.sender_buf_dev.as_ptr_dev(), SendBufMeta, head);
                DeviceNonNull::new_unchecked(ptr as _)
            };
            let tail = unsafe {
                let ptr = raw_field!(
                    shm_resources.receiver_buf_dev.as_ptr_dev(),
                    RecvBufMeta,
                    tail
                );
                DeviceNonNull::new_unchecked(ptr as _)
            };

            let mut bufs = MaybeUninit::uninit_array();
            let mut buf_curr = match sender.locality {
                ShmLocality::Sender => unsafe {
                    shm_resources.sender_buf_dev.as_ptr_dev().add(1) as *mut u8
                },
                ShmLocality::Receiver => unsafe {
                    shm_resources.receiver_buf_dev.as_ptr_dev().add(1) as *mut u8
                },
            };
            for (proto, buf) in bufs.iter_mut().enumerate().take(NUM_PROTOCOLS) {
                unsafe {
                    let dev_ptr = DeviceNonNull::new_unchecked(buf_curr);
                    buf.write(dev_ptr);
                    buf_curr = buf_curr.add(sender.buf_sizes[proto]);
                }
            }
            let bufs = unsafe { MaybeUninit::array_assume_init(bufs) };

            let info = PeerConnInfo {
                bufs,
                head,
                tail,
                slots_size: None,
            };
            TransportConnect::Connect {
                conn_info: info,
                transport_resources: Box::new(shm_resources),
            }
        }
    }

    fn recv_connect_agent_callback(
        &self,
        _conn_id: &PeerConnId,
        agent_reply: AgentMessage,
        transport_resources: Option<AnyResources>,
    ) -> TransportConnect {
        let reply = *agent_reply.unwrap().downcast::<ShmAgentReply>().unwrap();
        let resources = transport_resources
            .unwrap()
            .downcast::<ShmConnectedResources>()
            .unwrap();

        let meta_sync_dev = reply.meta_sync.as_ptr();
        let head = unsafe {
            let ptr = raw_field!(resources.sender_buf_dev.as_ptr_dev(), SendBufMeta, head);
            DeviceNonNull::new_unchecked(ptr as _)
        };
        let tail = unsafe {
            let ptr = raw_field!(meta_sync_dev, RecvBufMeta, tail);
            DeviceNonNull::new_unchecked(ptr as _)
        };

        let mut bufs = MaybeUninit::uninit_array();
        let mut buf_curr = match resources.locality {
            ShmLocality::Sender => unsafe {
                resources.sender_buf_dev.as_ptr_dev().add(1) as *mut u8
            },
            ShmLocality::Receiver => unsafe {
                resources.receiver_buf_dev.as_ptr_dev().add(1) as *mut u8
            },
        };
        for (proto, buf) in bufs.iter_mut().enumerate().take(NUM_PROTOCOLS) {
            unsafe {
                let dev_ptr = DeviceNonNull::new_unchecked(buf_curr);
                buf.write(dev_ptr);
                buf_curr = buf_curr.add(resources.buf_sizes[proto]);
            }
        }
        bufs[PROTOCOL_SIMPLE].write(reply.device_buf);
        let bufs = unsafe { MaybeUninit::array_assume_init(bufs) };

        let info = PeerConnInfo {
            bufs,
            head,
            tail,
            slots_size: None,
        };
        TransportConnect::Connect {
            conn_info: info,
            transport_resources: resources,
        }
    }

    async fn agent_send_setup(
        &self,
        _id: TransportAgentId,
        _agent_request: AgentMessage,
    ) -> (AnyResources, AgentMessage) {
        unimplemented!("SHM transport agent does not require setup")
    }

    async fn agent_send_connect(
        &self,
        _id: TransportAgentId,
        agent_request: AgentMessage,
        _setup_resources: Option<AnyResources>,
    ) -> (AnyResources, AgentMessage) {
        shm_agent_connect(agent_request).await
    }

    async fn agent_recv_setup(
        &self,
        _id: TransportAgentId,
        _agent_request: AgentMessage,
    ) -> (AnyResources, AgentMessage) {
        unimplemented!("SHM transport agent does not require setup")
    }

    async fn agent_recv_connect(
        &self,
        _id: TransportAgentId,
        agent_request: AgentMessage,
        _setup_resources: Option<AnyResources>,
    ) -> (AnyResources, AgentMessage) {
        shm_agent_connect(agent_request).await
    }

    fn agent_send_progress_op(&self, op: &mut TransportOp, resources: &mut AnyResources) {
        shm_agent_send_progress(resources, op);
    }

    fn agent_recv_progress_op(&self, op: &mut TransportOp, resources: &mut AnyResources) {
        shm_agent_recv_progress(resources, op)
    }
}
