
#ifndef MCCS_DEVICE_H_
#define MCCS_DEVICE_H_

#include "align.h"
#include <stdint.h>
#include <stddef.h>

#define MCCS_NUM_FUNCTIONS 5 // Send/Recv not included for now
typedef enum { 
  mccsFuncBoadcast, 
  mccsFuncReduce, 
  mccsFuncAllGather, 
  mccsFuncReduceScatter, 
  mccsFuncAllReduce, 
  mccsFuncSendRecv, 
  mccsFuncSend, 
  mccsFuncRecv, 
  mccsNumFuncs
} mccsDevFunc_t;

#define MCCS_NUM_ALGORITHMS 1
#define MCCS_ALGO_RING 0

#define MCCS_NUM_PROTOCOLS 1
#define MCCS_PROTO_SIMPLE 0

#define MCCS_MAX_OPS 2048
#define MCCS_BUFFER_SLOTS 8

#define WARP_SIZE 32
#define MCCS_MAX_NCHANNELS 32
#define MCCS_MAX_NTHREADS 640
#define MCCS_SIMPLE_MAX_NTHREADS 512

struct mccsDevConnInfo {
  // Regular comm mechanism
  char *buffs[MCCS_NUM_PROTOCOLS]; // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int *sizesFifo;     // Sizes fifo from GPU to proxy
  int *offsFifo;      // Buffer fifo from proxy to GPU

  uint64_t step;      // Keep where we are
};

struct mccsDevRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal mccs index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;

  int index; // This rank's index in the ring
};

#define MCCS_MAX_CONNS 2

/* mccsDevWork is to be a power of two, currently 8x64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of mccsDevWorkElem. */
#define MCCS_WORK_SIZE 512
enum mccsDevWorkType : uint8_t {
   mccsDevWorkTypeUnused=0,
   mccsDevWorkTypeColl=1,
   mccsDevWorkTypeP2p=2,
};
enum mccsDevWorkP2PType : uint8_t {
  mccsDevWorkP2pTypeUnused=0,
  mccsDevWorkP2pTypeSend,
  mccsDevWorkP2pTypeRecv
};

struct mccsDevWorkHeader {
  union {
    int32_t workNext;  // when isLast=0: Offset from kernel argument workHead
    uint32_t doneAcks; // when isLast=1: Monotonic (mod 1<<32) ack value to send back.
  };
  uint16_t funcIndex;
  uint8_t isLast:1; // last work for this kernel
  uint8_t inFifo:1; // is this work in the fifo
  enum mccsDevWorkType type;
};

struct mccsDevWorkElem {
  uint8_t isUsed:1;
  uint8_t nWarps;

  const void * sendbuff;
  void * recvbuff;

  size_t count;
  uint32_t root;
  uint8_t bid;
  uint8_t nChannels;
  uint64_t redOpArg;
};

#define MCCS_MAX_WORK_ELEMENTS ((MCCS_WORK_SIZE - alignUp(sizeof(mccsDevWorkHeader), alignof(mccsDevWorkElem)))/sizeof(mccsDevWorkElem))
static_assert(MCCS_MAX_WORK_ELEMENTS == 10, "Sanity check: MCCS_MAX_WORK_ELEMENTS == 10");

struct mccsDevWorkElemP2p {
  int peer : 30;
  int proto : 2;

  enum mccsDevWorkP2PType p2pType;
  uint8_t nWarps;
  uint8_t warpStart;
  uint8_t ngroups;
  // Important not to use any fields with greater than 4-byte alignment since
  // we need sizeof(mccsWorkElemP2p)==28, but that would be padded up to 32 if
  // there were 8-byte fields.
  //void* buff;
  uint32_t buffHi32, buffLo32; // buff = buffHi32<<32 | buffLo32;
  //size_t count;
  uint32_t countHi32, countLo32; // count = countHi32<<32 | countLo32;
  int chunkSize;
};

static_assert(((MCCS_WORK_SIZE - alignUp(sizeof(mccsDevWorkHeader), alignof(mccsDevWorkElemP2p)))/sizeof(mccsDevWorkElemP2p)) >= 16, "Sanity check: MCCS_MAX_WORK_ELEMENTS_P2P == 16");
#define MCCS_MAX_WORK_ELEMENTS_P2P 16

// Number of named barriers supported by CUDA
#define MCCS_MAX_GROUPS 16

struct mccsDevWork {
  struct mccsDevWorkHeader header;
  union {
    char pad[MCCS_WORK_SIZE - sizeof(struct mccsDevWorkHeader)];
    struct mccsDevWorkElem elems[MCCS_MAX_WORK_ELEMENTS];
    struct mccsDevWorkElemP2p p2pElems[MCCS_MAX_WORK_ELEMENTS_P2P];
  };
};
static_assert(sizeof(struct mccsDevWork) == MCCS_WORK_SIZE, "Sanity check: sizeof(struct mccsDevWork) == MCCS_WORK_SIZE");
static_assert(sizeof(struct mccsDevWork)%16 == 0, "Sanity check: sizeof(struct mccsDevWork)%16 == 0");

struct mccsDevChannelPeer {
  struct mccsDevConnInfo send[MCCS_MAX_CONNS];
  struct mccsDevConnInfo recv[MCCS_MAX_CONNS];
};

struct alignas(16) mccsDevChannel {
  struct mccsDevChannelPeer *peers;
  struct mccsDevRing ring;
  uint32_t* workFifoDone; // Location of done counter, device writes index+1 of last work processed
};

struct mccsDevComm {
  int rank;
  int nRanks;
  int buffSizes[MCCS_NUM_PROTOCOLS];

  // Flag to ask MCCS kernels to abort
  volatile uint32_t* abortFlag;
};

struct alignas(16) mccsDevCommAndChannels {
  struct mccsDevComm comm;
  struct mccsDevChannel channels[MCCS_MAX_NCHANNELS];
};

#endif
