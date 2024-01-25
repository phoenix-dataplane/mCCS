use super::{Protocol, NUM_BUFFER_SLOTS};
use crate::comm::CommunicatorId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportOpState {
    Init,
    InProgress,
    Completed,
}

/*
NOTE: ncclProxySubArgs is only used by NCCL proxy
some elements in ncclProxyOp are duplicated in ncclProxySubArgs and ncclProxyArgs
these elements (e.g., peer) are used in e.g.,
ncclProxySaveOp to determine whether a proxyOp is needed

struct ncclProxySubArgs {
  // used to locate transportResources, not needed for mCCS
  struct ncclProxyConnection* connection;
  // for debug only
  int channelId;
  int nsteps;
  // this is only used for P2P
  // for collective it is always proxyOp->nbytes = stepSize*proxyOp->sliceSteps
  ssize_t nbytes;
  // for debug only
  int peer;

  // not used
  // sub groupping is only used for P2P
  //   if (shared && args->opCount == op->opCount)
  //    NCCLCHECK(ncclProxyOpToArgs(op, args, args->nsubs));
  int groupSize; // Number of consecutive sub operations sharing the same recvComm
  uint64_t base;
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;
  uint64_t transmitted;
  uint64_t done;
  // not used
  uint64_t end;
  // keep NET requests here
  void* requests[NCCL_STEPS];
  // not needed
  void* profilingEvents[NCCL_STEPS];
};


struct ncclProxyArgs {
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];
  // correspond to the proxy progress function
  proxyProgressFunc_t progress;
  // total num of sub args
  int nsubs;
  // how many sub args are done
  int done;
  // used only for debug and element linking
  // not sued by the proxy
  uint64_t opCount;
  int sliceSteps;
  int chunkSteps;
  // used only by coll net
  int chunkSize;
  // used only for coll net
  uint8_t /*ncclDataType_t*/ dtype;
  // used only for coll net
  uint8_t /*ncclDevRedOp_t*/ redOp;
  // used only for coll net
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t protocol;
  int state;
  // used only for P2P
  char* sharedBuff[NCCL_STEPS];
  // used only for P2P
  int sharedSize[NCCL_STEPS];

  // whether progress has been made by the proxy
  int idle;

  // Element linking
  // not needed for mCCS
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
  struct ncclProxyArgs** proxyAppendPtr;
};
*/

// NCCL supports groupping multiple ncclProxySubArgs
// inside a ncclProxyArgs
// each ncclProxySubArgs could correspond to a different connection
// (each connection is identified by channel, peer, connIndex)
// we can also have TransportSubOp, and put AgentResources in each sub op
#[derive(Clone, Debug)]
pub struct TransportOp {
    pub communicator_id: CommunicatorId,
    pub num_steps: u32,

    pub slice_steps: u32,
    pub chunk_steps: u32,

    pub protocol: Protocol,

    pub state: TransportOpState,

    pub requests_id: [Option<u32>; NUM_BUFFER_SLOTS],
    pub idle: bool,

    pub base: u64,
    pub posted: u64,
    pub received: u64,
    pub flushed: u64,
    pub transmitted: u64,
    pub done: u64,
    pub debug_id: u64,
}

static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl TransportOp {
    pub fn new(
        communicator_id: CommunicatorId,
        num_steps: u32,
        slice_steps: u32,
        chunk_steps: u32,
        protocol: Protocol,
    ) -> Self {
        Self {
            communicator_id,
            num_steps,
            slice_steps,
            chunk_steps,
            protocol,
            state: TransportOpState::Init,
            requests_id: Default::default(),
            idle: true,
            base: 0,
            posted: 0,
            received: 0,
            flushed: 0,
            transmitted: 0,
            done: 0,
            debug_id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }
}
