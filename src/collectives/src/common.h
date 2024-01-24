/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MCCS_DEVICE_COMMON_H_
#define MCCS_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"
#include "op128.h"
#include <stdio.h>

#if __CUDA_ARCH__ >= 800
#define COLL_UNROLL 8
#else
#define COLL_UNROLL 4
#endif


typedef void(*mccsDevKern_t)();
extern __device__ mccsDevKern_t mccsDevFuncs[];

struct mccsShmemGroup {
  mccsDevConnInfo *recvConns[1];
  mccsDevConnInfo *sendConns[1];
  void* srcs[2];
  void* dsts[2];
  int totalSendSize[MCCS_MAX_SLICE_PER_CHUNK];
};

struct mccsShmemData {
  struct mccsShmemGroup groups[MCCS_MAX_GROUPS];
  uint64_t redOpArgs[2];
  int channelId;
  alignas(16) struct mccsDevComm comm;
  alignas(16) struct mccsDevChannel channel;
  alignas(16) struct mccsDevWork work;
};
static_assert(offsetof(struct mccsShmemData, work)%16 == 0, "shmem.work needs to be 16B aligned");

// shared memory is shared within each thread block
extern __shared__ mccsShmemData mccsShmem;

__device__ inline bool barrierReduceAny(int bit) {
  uint32_t popc;
  asm ("{"
    ".reg .pred barr_pred;"
    "setp.eq.u32 barr_pred, %1, 1;"
    "bar.red.popc.u32 %0, 2, barr_pred;"
  "}" : "=r"(popc) : "r"(bit));
  return popc != 0;
}

// Copy 16-byte aligned data. You must call with at least `(bytes+15)/16` threads.
inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    uint64_t a=0, b=0;
    asm("ld.v2.u64 {%0,%1},[%2];" : "=l"(a),"=l"(b) : "l"((char const*)src + offset));
    asm volatile("st.v2.u64 [%0],{%1,%2};" :: "l"((char*)dst + offset), "l"(a), "l"(b));
  }
}

template<mccsDevFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkElement {
  __device__ void run(mccsDevWorkElem*) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

template<mccsDevFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWork {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL mccsKernel.
  __device__ __forceinline__ void run(mccsDevWork *w) {
    int wid = threadIdx.x / WARP_SIZE;
    mccsDevWorkElem* we = &w->elems[0];
    int stride = sizeof(mccsDevWorkElem);
    #pragma unroll 1
    while ((char*)we + stride <= (char*)(w+1) && we->isUsed) {
      if (wid < we->nWarps) {
        RunWorkElement<Fn, T, RedOp, Algo, Proto>().run(we);
      }
      we = (mccsDevWorkElem*)((char*)we + stride);
    }
  }
};

template<mccsDevFunc_t Fn, typename T, typename RedOp, int Algo, int Proto, int FnIndex>
__device__ void mccsKernel(
    struct mccsDevComm* comm, uint64_t channelMask, struct mccsDevWork* workHead
  )  {
  int tid = threadIdx.x;

  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  if (tid < WARP_SIZE) {
    // only threads in the first wrap check channels
    int x = tid;
    // thread x within the block check channel x
    if (channelMask & (1ull<<x)) {
      int y = __popcll(channelMask & ((1ull<<x)-1));
      if (blockIdx.x == y) mccsShmem.channelId = x;
    }
    if (32 < MCCS_MAX_NCHANNELS) {
      // each thread also checks higher 32 channels if needed
      x = 32 + tid;
      if (channelMask & (1ull<<x)) {
        int y = __popcll(channelMask & ((1ull<<x)-1));
        if (blockIdx.x == y) mccsShmem.channelId = x;
      }
    }
  }
  __syncthreads(); // publish mccsShmem.channelId
  int channelId = mccsShmem.channelId;

  if (true) {
    void *dst, *src;
    int bytes;
    // Use first 3 warps to load comm, channel, and work into mccsShmem
    switch (tid/WARP_SIZE) {
    case 0:
      dst = &mccsShmem.comm;
      src = comm;
      bytes = sizeof(mccsDevComm);
      static_assert(sizeof(mccsDevComm) <= 16*WARP_SIZE, "mccsDevComm cannot be loaded by a single warp in one insn.");
      break;
    case 1:
      // Get address of channel without incurring indirect load from mccsDevComm::channels
      dst = &mccsShmem.channel;
      src = &((mccsDevCommAndChannels*)comm)->channels[channelId];
      bytes = sizeof(mccsDevChannel);
      static_assert(sizeof(mccsDevChannel) <= 16*WARP_SIZE, "mccsDevChannel cannot be loaded by a single warp in one insn.");
      break;
    case 2:
      dst = &mccsShmem.work;
      src = workHead + blockIdx.x;
      bytes = sizeof(mccsDevWork);
      static_assert(sizeof(mccsDevWork) <= 16*WARP_SIZE, "mccsDevWork cannot be loaded by a single warp in one insn.");
      break;
    default:
      bytes = 0;
      break;
    }
    copyToShmem16(tid%WARP_SIZE, dst, src, bytes);
  }
  __syncthreads(); // publish mccsShmem

  while (true) {
    // Notify host that all fifo reads are complete.
    if (tid == 0 && mccsShmem.work.header.isLast && mccsShmem.work.header.inFifo) {
      *mccsShmem.channel.workFifoDone = mccsShmem.work.header.doneAcks;
    }

    __syncthreads();

    // if (mccsShmem.work.header.funcIndex == FnIndex) {
    //   RunWork<Fn, T, RedOp, Algo, Proto>().run(&mccsShmem.work);
    // } else {
    //   mccsDevFuncs[mccsShmem.work.header.funcIndex]();
    // }

    RunWork<Fn, T, RedOp, Algo, Proto>().run(&mccsShmem.work);
    
    int workIxNext = mccsShmem.work.header.workNext;
    __syncthreads();
    if (mccsShmem.work.header.isLast) break;

    copyToShmem16(tid, &mccsShmem.work, workHead + workIxNext, sizeof(mccsDevWork));

    { // Check whether the last operation was aborted and make sure all threads exit
      int aborted = tid == 0 ? *comm->abortFlag : 0;
      if (barrierReduceAny(aborted)) // publish mccsShmem.work
        break;
    }
  }
}

// Only generate kernels for SUM
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fIndex) \
__global__ void MCCS_KERN_NAME(func, algo, proto, devredop, type)( \
    struct mccsDevComm* comm, uint64_t channelMask, struct mccsDevWork* workHead \
  ) { \
  mccsKernel<mccsFunc##func, type, Func##devredop<type>, MCCS_ALGO_##algo, MCCS_PROTO_##proto, fIndex> \
    (comm, channelMask, workHead); \
}

#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__ void MCCS_FUNC_NAME(func, algo, proto, devredop, type)() { \
  RunWork<mccsFunc##func, type, Func##devredop<type>, MCCS_ALGO_##algo, MCCS_PROTO_##proto>().run(&mccsShmem.work); \
}

// Only generate inline kernels for SIMPLE
#define IMPL_COLL4(func, algo, devredop, type, mccsType) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, devredop, type) \
  IMPL_COLL_KERN(func, algo, SIMPLE, devredop, type, FUNC_INDEX(mccsFunc##func, mccsDev##devredop, mccsType, MCCS_ALGO_##algo, MCCS_PROTO_SIMPLE)) 

#define IMPL_COLL3(func, devredop, type, mccsType) \
  IMPL_COLL4(func, RING,    devredop, type, mccsType) 

#if MCCS_TYPE == 0
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int8_t,   mccsInt8)
#elif MCCS_TYPE == 1
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint8_t,  mccsUint8)
#elif MCCS_TYPE == 2
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int32_t,  mccsInt32)
#elif MCCS_TYPE == 3
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint32_t, mccsUint32)
#elif MCCS_TYPE == 4
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int64_t,  mccsInt64)
#elif MCCS_TYPE == 5
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint64_t, mccsUint64)
#elif MCCS_TYPE == 6
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, half,     mccsFloat16)
#elif MCCS_TYPE == 7
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, float,    mccsFloat32)
#elif MCCS_TYPE == 8
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, double,   mccsFloat64)
#elif MCCS_TYPE == 9 && defined(__CUDA_BF16_TYPES_EXIST__)
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, __nv_bfloat16, mccsBfloat16)
#else
#pragma message("WTF"#MCCS_TYPE)
#endif

// Reduction define all functions
#if MCCS_OP == 0
#define IMPL_COLL_R(func) IMPL_COLL2(func, Sum);
#elif MCCS_OP == 1
#define IMPL_COLL_R(func) IMPL_COLL2(func, Prod);
#elif MCCS_OP == 2
#define IMPL_COLL_R(func) IMPL_COLL2(func, Min);
#elif MCCS_OP == 3
#define IMPL_COLL_R(func) IMPL_COLL2(func, Max);
#elif MCCS_OP == 4
#define IMPL_COLL_R(func) IMPL_COLL2(func, PreMulSum);
#elif MCCS_OP == 5
  #if MCCS_TYPE < 6
    #define IMPL_COLL_R(func) IMPL_COLL2(func, SumPostDiv);
  #else
    #define IMPL_COLL_R(func) // skip SumPostDiv for floating point
  #endif
#endif

#if MCCS_OP == 0 && MCCS_TYPE == 0
// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t, mccsInt8);
#else
#define IMPL_COLL_C(func)
#define IMPL_COLL_P(func)
#endif

#endif