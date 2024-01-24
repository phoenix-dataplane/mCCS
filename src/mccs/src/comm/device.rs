use std::collections::BTreeMap;
use std::mem::MaybeUninit;

use collectives_sys::{
    mccsDevChannel, mccsDevChannelPeer, mccsDevComm, mccsDevCommAndChannels, mccsDevConnInfo,
    mccsDevRing,
};
use collectives_sys::{mccsDevWork, MCCS_MAX_NCHANNELS};
use cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice;
use cuda_runtime_sys::{cudaMalloc, cudaMemcpy};

use crate::cuda::alloc::{DeviceAlloc, DeviceHostMapped};
use crate::cuda::ptr::DeviceNonNull;
use crate::cuda_warning;
use crate::transport::channel::{ChannelId, CommChannel, PeerConnInfo};

use super::CommProfile;

struct ChanDevStorage {
    peers: DeviceAlloc<mccsDevChannelPeer>,
    ring_user_ranks: DeviceAlloc<i32>,
}

impl ChanDevStorage {
    fn new(num_ranks: usize) -> Self {
        let peers = DeviceAlloc::new(num_ranks);
        let ring_user_ranks = DeviceAlloc::new(num_ranks);
        ChanDevStorage {
            peers,
            ring_user_ranks,
        }
    }
}

fn conn_info_to_dev(conn_info: &PeerConnInfo) -> mccsDevConnInfo {
    let bufs = conn_info.bufs.map(|x| x.as_ptr() as *mut _);
    let tail = conn_info.tail.as_ptr();
    let head = conn_info.head.as_ptr();
    let sizes = if let Some(slots_size) = conn_info.slots_size {
        slots_size.as_ptr() as *mut _
    } else {
        std::ptr::null_mut()
    };
    mccsDevConnInfo {
        buffs: bufs,
        tail,
        head,
        sizesFifo: sizes,
        offsFifo: std::ptr::null_mut(),
        step: 0,
    }
}

pub(crate) struct DevHostSyncResources {
    pub(crate) work_queue_heap: DeviceHostMapped<mccsDevWork>,
    pub(crate) work_queue_done: DeviceHostMapped<u32>,
    // maps index in work_queue_done to its corresponding channel id
    chan_mapping: Vec<ChannelId>,
}

impl DevHostSyncResources {
    fn new(num_channels: usize, queue_depth: usize) -> Self {
        let work_queue_heap = DeviceHostMapped::alloc(queue_depth);
        let work_queue_done = DeviceHostMapped::alloc(num_channels);
        DevHostSyncResources {
            work_queue_heap,
            work_queue_done,
            chan_mapping: Vec::with_capacity(num_channels),
        }
    }
}

pub struct CommDevResources {
    comm_dev: DeviceAlloc<mccsDevCommAndChannels>,

    chan_storage: Vec<ChanDevStorage>,
    pub(crate) sync: DevHostSyncResources,
}

impl CommDevResources {
    pub fn new(
        rank: usize,
        num_ranks: usize,
        queue_depth: usize, // must be a power of 2
        profile: &CommProfile,
        channels: &BTreeMap<ChannelId, CommChannel>,
    ) -> Self {
        debug_assert_eq!(queue_depth & (queue_depth - 1), 0);
        let buf_sizes = profile.buff_sizes.map(|x| x as _);
        let mut dev_host_sync = DevHostSyncResources::new(channels.len(), queue_depth);
        let mut dev_channels = [MaybeUninit::zeroed(); MCCS_MAX_NCHANNELS as usize];
        let mut dev_chan_stroage = Vec::with_capacity(channels.len());
        for (idx, (chan_id, chan)) in channels.iter().enumerate() {
            let storage = ChanDevStorage::new(num_ranks);
            let mut dev_chan_peers = vec![MaybeUninit::zeroed(); num_ranks];
            for (peer_rank, peer_conn) in chan.peers.iter() {
                let mut dev_chan_peer =
                    unsafe { MaybeUninit::<mccsDevChannelPeer>::zeroed().assume_init() };
                for (conn_index, send_conn) in peer_conn.send.iter().enumerate() {
                    if let Some(send_conn) = send_conn {
                        let conn_info = conn_info_to_dev(&send_conn.conn_info);
                        dev_chan_peer.send[conn_index as usize] = conn_info;
                    }
                }
                for (conn_index, recv_conn) in peer_conn.recv.iter().enumerate() {
                    if let Some(recv_conn) = recv_conn {
                        let conn_info = conn_info_to_dev(&recv_conn.conn_info);
                        dev_chan_peer.recv[conn_index as usize] = conn_info;
                    }
                }
                dev_chan_peers[*peer_rank] = MaybeUninit::new(dev_chan_peer);
            }
            let user_ranks = chan
                .ring
                .user_ranks
                .iter()
                .map(|x| *x as i32)
                .collect::<Vec<_>>();
            unsafe {
                cuda_warning!(cudaMemcpy(
                    storage.ring_user_ranks.as_ptr() as _,
                    user_ranks.as_ptr() as _,
                    num_ranks * std::mem::size_of::<i32>(),
                    cudaMemcpyHostToDevice,
                ));
                cuda_warning!(cudaMemcpy(
                    storage.peers.as_ptr() as _,
                    dev_chan_peers.as_ptr() as _,
                    num_ranks * std::mem::size_of::<mccsDevChannelPeer>(),
                    cudaMemcpyHostToDevice,
                ));
            }
            let dev_ring = mccsDevRing {
                prev: chan.ring.prev as _,
                next: chan.ring.next as _,
                userRanks: storage.ring_user_ranks.as_ptr(),
                index: chan.ring.index as _,
            };
            let work_done = unsafe { dev_host_sync.work_queue_done.as_ptr_dev().add(idx) };
            dev_host_sync.chan_mapping.push(*chan_id);
            let dev_chan = mccsDevChannel {
                peers: storage.peers.as_ptr(),
                ring: dev_ring,
                workFifoDone: work_done,
            };
            dev_channels[chan_id.0 as usize].write(dev_chan);
            dev_chan_stroage.push(storage);
        }

        let dev_channels = unsafe { MaybeUninit::array_assume_init(dev_channels) };
        let mut dev_comm = mccsDevComm {
            rank: rank as _,
            nRanks: num_ranks as _,
            buffSizes: buf_sizes,
            abortFlag: std::ptr::null_mut(),
        };
        cuda_warning!(unsafe {
            cudaMalloc(
                &mut dev_comm.abortFlag as *mut *mut u32 as *mut *mut _,
                std::mem::size_of_val(&dev_comm.abortFlag),
            )
        });
        let mut dev_comm_chans = mccsDevCommAndChannels {
            comm: dev_comm,
            __bindgen_padding_0: 0,
            channels: dev_channels,
        };

        let dev_comm_storage = DeviceAlloc::new(1);
        unsafe {
            cuda_warning!(cudaMemcpy(
                dev_comm_storage.as_ptr() as _,
                &mut dev_comm_chans as *mut mccsDevCommAndChannels as _,
                std::mem::size_of::<mccsDevCommAndChannels>(),
                cudaMemcpyHostToDevice,
            ));
        }
        CommDevResources {
            comm_dev: dev_comm_storage,
            chan_storage: dev_chan_stroage,
            sync: dev_host_sync,
        }
    }

    pub fn get_dev_comm_ptr(&self) -> DeviceNonNull<mccsDevComm> {
        let ptr = self.comm_dev.as_ptr() as *mut mccsDevComm;
        unsafe { DeviceNonNull::new_unchecked(ptr) }
    }
}
