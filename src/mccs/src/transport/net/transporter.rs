use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;

use async_trait::async_trait;
use memoffset::raw_field;
use qos_service::QosSchedule;
use strum::IntoEnumIterator;

use cuda_runtime_sys::cudaError;
use cuda_runtime_sys::{cudaDeviceEnablePeerAccess, cudaGetDevice, cudaGetLastError};

use super::agent;
use super::config::NetTransportConfig;
use super::resources::NetSendSetup;
use super::resources::{AgentRecvConnectReply, AgentRecvSetupReply, AgentSendConnectReply};
use super::resources::{AgentRecvConnectRequest, AgentSendConnectRequest, AgentSetupRequest};
use super::resources::{AgentRecvResources, AgentSendResources};
use super::resources::{AgentRecvSetup, AgentSendSetup};
use super::resources::{NetRecvResources, NetSendResources};
use super::{NetAgentError, NetTransportError};
use crate::comm::{CommProfile, PeerInfo};
use crate::cuda::ptr::DeviceNonNull;
use crate::cuda_warning;
use crate::transport::catalog::TransportCatalog;
use crate::transport::channel::{PeerConnId, PeerConnInfo, ConnType};
use crate::transport::meta::{RecvBufMeta, SendBufMeta};
use crate::transport::op::TransportOp;
use crate::transport::transporter::TransporterError;
use crate::transport::transporter::{AgentMessage, AnyResources};
use crate::transport::transporter::{ConnectHandle, TransportAgentId, Transporter};
use crate::transport::transporter::{TransportConnect, TransportSetup};
use crate::transport::Protocol;

pub struct NetTransport;
pub static NET_TRANSPORT: NetTransport = NetTransport;

fn net_send_setup(
    conn_id: &PeerConnId,
    my_info: &PeerInfo,
    _peer_info: &PeerInfo,
    profile: &CommProfile,
    config: &NetTransportConfig,
) -> Result<TransportSetup, NetTransportError> {
    // NVLink forward is not supported yet
    // 8 GPUs GPT trace
    // 2 -> 3 -> 1 -> 5 -> 2: Bad (25G) (5->2) corresponds to rank 7 -> rank 0
    // 5 -> 1 -> 3 -> 2 -> 5: Good (100G)
    let (net_dev, _) = profile.get_network_device(conn_id.channel, my_info.rank, conn_id.peer_rank);
    let udp_sport = profile.get_udp_sport(&conn_id);
    let mut tc = profile.get_tc();

    // if conn_id.peer_rank == 0 && my_info.rank == 7 && conn_id.conn_type == ConnType::Send && my_info.host == IpAddr::V4(Ipv4Addr::new(192, 168, 211, 162)) {
    //     log::warn!("Force to use TC 66 for 25G connection.");
    //     tc = Some(66);
    // }

    let agent_cuda_dev = my_info.cuda_device_idx;
    let use_gdr = profile.check_gdr(my_info.rank, net_dev, true) && config.gdr_enable;
    let provider = profile.get_net_provider();

    let setup_resources = NetSendSetup {
        agent_rank: my_info.rank,
    };
    let setup_request = AgentSetupRequest {
        rank: my_info.rank,
        local_rank: my_info.local_rank,
        remote_rank: conn_id.peer_rank,
        net_device: net_dev,
        use_gdr,
        need_flush: false,
        buffer_sizes: profile.buff_sizes,
        provider,
        udp_sport,
        tc,
    };
    let setup = TransportSetup::PreAgentCb {
        agent_cuda_dev,
        agent_request: Some(Box::new(setup_request)),
        setup_resources: Some(Box::new(setup_resources)),
    };

    let net_name = provider.get_properties(net_dev)?.name;
    log::info!(
        "Channel {:0>2}/{}: {} -> {} [send] via NET/{}/{}, udp_sport={:?}, GDRDMA={}",
        conn_id.channel,
        conn_id.conn_index,
        my_info.rank,
        conn_id.peer_rank,
        net_name,
        net_dev,
        udp_sport,
        use_gdr,
    );
    Ok(setup)
}

fn net_send_setup_agent_callback(
    setup_resources: Option<AnyResources>,
) -> Result<TransportSetup, NetTransportError> {
    let setup_resources = setup_resources
        .ok_or(NetTransportError::DowncastSetupResources)?
        .downcast::<NetSendSetup>()
        .map_err(|_| NetTransportError::DowncastSetupResources)?;

    let handle = ConnectHandle::serialize_from(setup_resources.agent_rank)?;
    let setup = TransportSetup::Setup {
        peer_connect_handle: handle,
        setup_resources: None,
    };
    Ok(setup)
}

fn net_recv_setup(
    conn_id: &PeerConnId,
    my_info: &PeerInfo,
    _peer_info: &PeerInfo,
    profile: &CommProfile,
    config: &NetTransportConfig,
) -> Result<TransportSetup, NetTransportError> {
    // Use myInfo->rank as the receiver uses its own NIC
    let (net_dev, _) = profile.get_network_device(conn_id.channel, my_info.rank, my_info.rank);
    let udp_sport = profile.get_udp_sport(&conn_id);
    let tc = profile.get_tc();

    let proxy_cuda_dev = my_info.cuda_device_idx;
    let use_gdr = profile.check_gdr(my_info.rank, net_dev, false) && config.gdr_enable;
    let need_flush = profile.check_gdr_need_flush(my_info.rank);
    let provider = profile.get_net_provider();

    let setup_request = AgentSetupRequest {
        rank: my_info.rank,
        local_rank: my_info.local_rank,
        remote_rank: conn_id.peer_rank,
        net_device: net_dev,
        use_gdr,
        need_flush,
        buffer_sizes: profile.buff_sizes,
        provider,
        udp_sport,
        tc,
    };
    let setup = TransportSetup::PreAgentCb {
        agent_cuda_dev: proxy_cuda_dev,
        agent_request: Some(Box::new(setup_request)),
        setup_resources: None,
    };

    let net_name = provider.get_properties(net_dev)?.name;
    log::info!(
        "Channel {:0>2}/{}: {} -> {} [recv] via NET/{}/{}, GDRDMA={}",
        conn_id.channel,
        conn_id.conn_index,
        conn_id.peer_rank,
        my_info.rank,
        net_name,
        net_dev,
        use_gdr,
    );
    Ok(setup)
}

fn net_recv_setup_agent_callback(
    agent_reply: AgentMessage,
) -> Result<TransportSetup, NetTransportError> {
    let agent_reply = *agent_reply
        .ok_or(NetTransportError::DowncastAgentReply)?
        .downcast::<AgentRecvSetupReply>()
        .map_err(|_| NetTransportError::DowncastAgentReply)?;
    let setup = TransportSetup::Setup {
        peer_connect_handle: agent_reply.handle,
        setup_resources: None,
    };
    Ok(setup)
}

fn net_send_connect(connect_handle: ConnectHandle) -> Result<TransportConnect, NetTransportError> {
    let request = AgentSendConnectRequest {
        handle: connect_handle,
    };
    let connect = TransportConnect::PreAgentCb {
        agent_request: Some(Box::new(request)),
        transport_resources: None,
    };
    Ok(connect)
}

fn net_send_connect_agent_callback(
    agent_reply: AgentMessage,
) -> Result<TransportConnect, NetTransportError> {
    let agent_reply = *agent_reply
        .ok_or(NetTransportError::DowncastAgentReply)?
        .downcast::<AgentSendConnectReply>()
        .map_err(|_| NetTransportError::DowncastAgentReply)?;

    let device = unsafe {
        let mut dev = 0;
        cuda_warning!(cudaGetDevice(&mut dev));
        dev
    };
    if device != agent_reply.agent_cuda_dev {
        unsafe {
            let err = cudaDeviceEnablePeerAccess(agent_reply.agent_cuda_dev, 0);
            if err == cudaError::cudaErrorPeerAccessAlreadyEnabled {
                cudaGetLastError();
            } else if err != cudaError::cudaSuccess {
                log::error!("CUDA failed with {:?} at {}:{}.", err, file!(), line!());
            }
        }
    }

    let send_mem = agent_reply
        .map
        .get_send_mem_meta()
        .ok_or(NetTransportError::InvalidAgentReply)?;
    let head = if let Some(gdc_mem) = agent_reply.map.get_gdc_mem_gpu_ptr() {
        gdc_mem
    } else {
        let base_ptr = send_mem.as_ptr_dev();
        let head_ptr = raw_field!(base_ptr, SendBufMeta, head);
        unsafe { DeviceNonNull::new_unchecked(head_ptr as _) }
    };

    let recv_mem = agent_reply
        .map
        .get_recv_mem_meta()
        .ok_or(NetTransportError::InvalidAgentReply)?;
    let tail = unsafe {
        let base_ptr = recv_mem.as_ptr_dev();
        let tail_ptr = raw_field!(base_ptr, RecvBufMeta, tail);
        DeviceNonNull::new_unchecked(tail_ptr as _)
    };
    let sizes_fifo = unsafe {
        let base_ptr = recv_mem.as_ptr_dev();
        let sizes_fifo_ptr = raw_field!(base_ptr, RecvBufMeta, slots_sizes);
        DeviceNonNull::new_unchecked(sizes_fifo_ptr as _)
    };

    let mut buffers = std::mem::MaybeUninit::uninit_array();
    for (proto, buf) in Protocol::iter().zip(buffers.iter_mut()) {
        let ptr = agent_reply
            .map
            .get_buffer_gpu_ptr(proto)
            .ok_or(NetTransportError::InvalidAgentReply)?
            .cast();
        buf.write(ptr);
    }
    let buffers = unsafe { std::mem::MaybeUninit::array_assume_init(buffers) };

    let conn_info = PeerConnInfo {
        bufs: buffers,
        head,
        tail,
        slots_size: Some(sizes_fifo),
    };

    let resources = NetSendResources {
        map: agent_reply.map,
    };
    let connect = TransportConnect::Connect {
        conn_info,
        transport_resources: Box::new(resources),
    };
    Ok(connect)
}

fn net_recv_connect(connect_handle: ConnectHandle) -> Result<TransportConnect, NetTransportError> {
    let send_agent_rank = connect_handle.deserialize_to::<usize>()?;
    log::debug!("send_agent_rank={send_agent_rank}");
    let request = AgentRecvConnectRequest { send_agent_rank };
    let connect = TransportConnect::PreAgentCb {
        agent_request: Some(Box::new(request)),
        transport_resources: None,
    };
    Ok(connect)
}

fn net_recv_connect_agent_callback(
    agent_reply: AgentMessage,
) -> Result<TransportConnect, NetTransportError> {
    let agent_reply = *agent_reply
        .ok_or(NetTransportError::DowncastAgentReply)?
        .downcast::<AgentRecvConnectReply>()
        .map_err(|_| NetTransportError::DowncastAgentReply)?;

    let send_mem = agent_reply
        .map
        .get_send_mem_meta()
        .ok_or(NetTransportError::InvalidAgentReply)?;
    let head = unsafe {
        let base_ptr = send_mem.as_ptr_dev();
        let head_ptr = raw_field!(base_ptr, SendBufMeta, head);
        DeviceNonNull::new_unchecked(head_ptr as _)
    };

    let recv_mem = agent_reply
        .map
        .get_recv_mem_meta()
        .ok_or(NetTransportError::InvalidAgentReply)?;
    let tail = if let Some(gdc_mem) = agent_reply.map.get_gdc_mem_gpu_ptr() {
        gdc_mem
    } else {
        let base_ptr = recv_mem.as_ptr_dev();
        let tail_ptr = raw_field!(base_ptr, SendBufMeta, head);
        unsafe { DeviceNonNull::new_unchecked(tail_ptr as _) }
    };

    let sizes_fifo = unsafe {
        let base_ptr = recv_mem.as_ptr_dev();
        let sizes_fifo_ptr = raw_field!(base_ptr, RecvBufMeta, slots_sizes);
        DeviceNonNull::new_unchecked(sizes_fifo_ptr as _)
    };

    let mut buffers = std::mem::MaybeUninit::uninit_array();
    for (proto, buf) in Protocol::iter().zip(buffers.iter_mut()) {
        let ptr = agent_reply
            .map
            .get_buffer_gpu_ptr(proto)
            .ok_or(NetTransportError::InvalidAgentReply)?
            .cast();
        buf.write(ptr);
    }
    let buffers = unsafe { std::mem::MaybeUninit::array_assume_init(buffers) };

    let conn_info = PeerConnInfo {
        bufs: buffers,
        head,
        tail,
        slots_size: Some(sizes_fifo),
    };

    let resources = NetRecvResources {
        map: agent_reply.map,
    };
    let connect = TransportConnect::Connect {
        conn_info,
        transport_resources: Box::new(resources),
    };
    Ok(connect)
}

#[async_trait]
impl Transporter for NetTransport {
    #[inline]
    fn can_connect(
        &self,
        _send_peer: &PeerInfo,
        _recv_peer: &PeerInfo,
        _profile: &CommProfile,
        _catalog: &TransportCatalog,
    ) -> bool {
        true
    }

    fn send_setup(
        &self,
        conn_id: &PeerConnId,
        my_info: &PeerInfo,
        peer_info: &PeerInfo,
        profile: &CommProfile,
        catalog: &TransportCatalog,
    ) -> Result<TransportSetup, TransporterError> {
        let config = catalog.get_config::<NetTransportConfig>("NetTransport")?;
        let setup = net_send_setup(conn_id, my_info, peer_info, profile, &config)?;
        Ok(setup)
    }

    fn send_setup_agent_callback(
        &self,
        _rank: usize,
        _conn_id: &PeerConnId,
        _agent_reply: AgentMessage,
        setup_resources: Option<AnyResources>,
    ) -> Result<TransportSetup, TransporterError> {
        let setup = net_send_setup_agent_callback(setup_resources)?;
        Ok(setup)
    }

    fn send_connect(
        &self,
        _conn_id: &PeerConnId,
        connect_handle: ConnectHandle,
        _setup_resources: Option<AnyResources>,
    ) -> Result<TransportConnect, TransporterError> {
        let connect = net_send_connect(connect_handle)?;
        Ok(connect)
    }

    fn send_connect_agent_callback(
        &self,
        _conn_id: &PeerConnId,
        agent_reply: AgentMessage,
        _transport_resources: Option<AnyResources>,
    ) -> Result<TransportConnect, TransporterError> {
        let connect = net_send_connect_agent_callback(agent_reply)?;
        Ok(connect)
    }

    fn recv_setup(
        &self,
        conn_id: &PeerConnId,
        my_info: &PeerInfo,
        peer_info: &PeerInfo,
        profile: &CommProfile,
        catalog: &TransportCatalog,
    ) -> Result<TransportSetup, TransporterError> {
        let config = catalog.get_config::<NetTransportConfig>("NetTransport")?;
        let setup = net_recv_setup(conn_id, my_info, peer_info, profile, &config)?;
        Ok(setup)
    }

    fn recv_setup_agent_callback(
        &self,
        _rank: usize,
        _conn_id: &PeerConnId,
        agent_reply: AgentMessage,
        _setup_resources: Option<AnyResources>,
    ) -> Result<TransportSetup, TransporterError> {
        let setup = net_recv_setup_agent_callback(agent_reply)?;
        Ok(setup)
    }

    fn recv_connect(
        &self,
        _conn_id: &PeerConnId,
        connect_handle: ConnectHandle,
        _setup_resources: Option<AnyResources>,
    ) -> Result<TransportConnect, TransporterError> {
        let connect = net_recv_connect(connect_handle)?;
        Ok(connect)
    }

    fn recv_connect_agent_callback(
        &self,
        _conn_id: &PeerConnId,
        agent_reply: AgentMessage,
        _transport_resources: Option<AnyResources>,
    ) -> Result<TransportConnect, TransporterError> {
        let connect = net_recv_connect_agent_callback(agent_reply)?;
        Ok(connect)
    }

    async fn agent_send_setup(
        &self,
        _id: TransportAgentId,
        agent_request: AgentMessage,
        catalog: Arc<TransportCatalog>,
    ) -> Result<(AnyResources, AgentMessage), TransporterError> {
        let request = *agent_request
            .ok_or(NetAgentError::DowncastAgentRequest)?
            .downcast::<AgentSetupRequest>()
            .map_err(|_| NetAgentError::DowncastAgentRequest)?;
        let agent_resources = agent::net_agent_send_setup(request, &catalog).await?;
        Ok((Box::new(agent_resources), None))
    }

    async fn agent_send_connect(
        &self,
        _id: TransportAgentId,
        agent_request: AgentMessage,
        setup_resources: Option<AnyResources>,
    ) -> Result<(AnyResources, AgentMessage), TransporterError> {
        let request = *agent_request
            .ok_or(NetAgentError::DowncastAgentRequest)?
            .downcast::<AgentSendConnectRequest>()
            .map_err(|_| NetAgentError::DowncastAgentRequest)?;
        let resources = *setup_resources
            .ok_or(NetAgentError::DowncastAgentResources)?
            .downcast::<AgentSendSetup>()
            .map_err(|_| NetAgentError::DowncastAgentResources)?;
        let (reply, agent_resources) = agent::net_agent_send_connect(request, resources).await?;
        Ok((Box::new(agent_resources), Some(Box::new(reply))))
    }

    async fn agent_recv_setup(
        &self,
        _id: TransportAgentId,
        agent_request: AgentMessage,
        catalog: Arc<TransportCatalog>,
    ) -> Result<(AnyResources, AgentMessage), TransporterError> {
        let request = *agent_request
            .ok_or(NetAgentError::DowncastAgentRequest)?
            .downcast::<AgentSetupRequest>()
            .map_err(|_| NetAgentError::DowncastAgentRequest)?;
        let (reply, agent_resources) = agent::net_agent_recv_setup(request, &catalog).await?;
        Ok((Box::new(agent_resources), Some(Box::new(reply))))
    }

    async fn agent_recv_connect(
        &self,
        _id: TransportAgentId,
        agent_request: AgentMessage,
        setup_resources: Option<AnyResources>,
    ) -> Result<(AnyResources, AgentMessage), TransporterError> {
        let request = *agent_request
            .ok_or(NetAgentError::DowncastAgentRequest)?
            .downcast::<AgentRecvConnectRequest>()
            .map_err(|_| NetAgentError::DowncastAgentRequest)?;
        let resources = *setup_resources
            .ok_or(NetAgentError::DowncastAgentResources)?
            .downcast::<AgentRecvSetup>()
            .map_err(|_| NetAgentError::DowncastAgentResources)?;
        let (reply, agent_resources) = agent::net_agent_recv_connect(request, resources).await?;
        Ok((Box::new(agent_resources), Some(Box::new(reply))))
    }

    fn agent_send_progress_op(
        &self,
        op: &mut TransportOp,
        resources: &mut AnyResources,
        schedule: &QosSchedule,
    ) -> Result<(), TransporterError> {
        let resources = resources
            .downcast_mut::<AgentSendResources>()
            .ok_or_else(|| NetAgentError::DowncastAgentResources)?;
        agent::net_agent_send_progress(resources, op, schedule)?;
        Ok(())
    }

    fn agent_recv_progress_op(
        &self,
        op: &mut TransportOp,
        resources: &mut AnyResources,
        _schedule: &QosSchedule,
    ) -> Result<(), TransporterError> {
        let resources = resources
            .downcast_mut::<AgentRecvResources>()
            .ok_or_else(|| NetAgentError::DowncastAgentResources)?;
        agent::net_agent_recv_progress(resources, op)?;
        Ok(())
    }
}
