// mCCS collective kernels and functions

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif

typedef enum { mccsInt8       = 0, mccsChar       = 0,
               mccsUint8      = 1,
               mccsInt32      = 2, mccsInt        = 2,
               mccsUint32     = 3,
               mccsInt64      = 4,
               mccsUint64     = 5,
               mccsFloat16    = 6, mccsHalf       = 6,
               mccsFloat32    = 7, mccsFloat      = 7,
               mccsFloat64    = 8, mccsDouble     = 8,
#if defined(__CUDA_BF16_TYPES_EXIST__)
               mccsBfloat16   = 9,
               mccsNumTypes   = 10
#else
               mccsNumTypes   = 9
#endif
} mccsDevDataType_t;

enum mccsDevRedOp_t {
  mccsDevSum, mccsDevProd, mccsDevMax, mccsDevMin,
  mccsDevPreMulSum, mccsDevSumPostDiv,
  mccsNumDevRedOps
};
struct mccsDevRedOpFull {
  mccsDevRedOp_t op;
  uint64_t scalarArg;
};

#define FUNC_INDEX(func, devredop, mccsType, algo, proto) (1+mccsNumTypes+(((((func)*mccsNumDevRedOps + (devredop))*mccsNumTypes) + (mccsType))*MCCS_NUM_ALGORITHMS+(algo))*MCCS_NUM_PROTOCOLS+(proto))

#define MCCS_FUNC_NAME(func, algo, proto, devredop, type) \
  mccsFunction_##func##_##algo##_##proto##_##devredop##_##type

#define MCCS_KERN_NAME(func, algo, proto, devredop, type) \
  mccsKernel_##func##_##algo##_##proto##_##devredop##_##type

/* Declare all collective operations */
#define DECL5(func, algo, proto, devredop, type) \
  extern __device__ void MCCS_FUNC_NAME(func, algo, proto, devredop, type)(); \
  extern __global__ void MCCS_KERN_NAME(func, algo, proto, devredop, type)(struct mccsDevComm* comm, uint64_t channelMask, struct mccsDevWork* workHead); \

#define SINGLE_ARG(...) __VA_ARGS__
#define CONCAT(a,b) a##b
#define MACRO_IF(cond, t, f) CONCAT(MACRO_IF_, cond)(SINGLE_ARG(t), SINGLE_ARG(f))
#define MACRO_IF_0(t, f) f
#define MACRO_IF_1(t, f) t

#define DECL4(func, algo, devredop, type, undef) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIMPLE, devredop, type)) \

#define DECL3(func, devredop, type, undef) \
  DECL4(func, RING,    devredop, type, undef) \

#if defined(__CUDA_BF16_TYPES_EXIST__)
#define DECL2(func, devredop, undefForFloat) \
  DECL3(func, devredop, int8_t, /*undef=*/0) \
  DECL3(func, devredop, uint8_t, /*undef=*/0) \
  DECL3(func, devredop, int32_t, /*undef=*/0) \
  DECL3(func, devredop, uint32_t, /*undef=*/0) \
  DECL3(func, devredop, int64_t, /*undef=*/0) \
  DECL3(func, devredop, uint64_t, /*undef=*/0) \
  DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  DECL3(func, devredop, double, /*undef=*/undefForFloat) \
  DECL3(func, devredop, __nv_bfloat16, /*undef=*/undefForFloat)
#else
#define DECL2(func, devredop, undefForFloat) \
  DECL3(func, devredop, int8_t, /*undef=*/0) \
  DECL3(func, devredop, uint8_t, /*undef=*/0) \
  DECL3(func, devredop, int32_t, /*undef=*/0) \
  DECL3(func, devredop, uint32_t, /*undef=*/0) \
  DECL3(func, devredop, int64_t, /*undef=*/0) \
  DECL3(func, devredop, uint64_t, /*undef=*/0) \
  DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  DECL3(func, devredop, double, /*undef=*/undefForFloat)
#endif

#define DECL(func) \
  DECL2(func, Sum, /*undefForFloat=*/0) \
  DECL2(func, Prod, /*undefForFloat=*/0) \
  DECL2(func, Min, /*undefForFloat=*/0) \
  DECL2(func, Max, /*undefForFloat=*/0) \
  DECL2(func, PreMulSum, /*undefForFloat=*/0) \
  DECL2(func, SumPostDiv, /*undefForFloat=*/1)

DECL2(AllGather, Sum, /*undefForFloat=*/0)
DECL2(AllReduce, Sum, /*undefForFloat=*/0)
DECL2(AllReduce, Prod, /*undefForFloat=*/0)
DECL2(AllReduce, Min, /*undefForFloat=*/0)
DECL2(AllReduce, Max, /*undefForFloat=*/0)
DECL2(AllReduce, PreMulSum, /*undefForFloat=*/0)
DECL2(AllReduce, SumPostDiv, /*undefForFloat=*/0)

#define ALLGATHER_SLICESTEPS (MCCS_BUFFER_SLOTS/4)
#define ALLGATHER_CHUNKSTEPS (MCCS_BUFFER_SLOTS/2)
#define ALLREDUCE_SLICESTEPS (MCCS_BUFFER_SLOTS/4)
#define ALLREDUCE_CHUNKSTEPS (MCCS_BUFFER_SLOTS/2)
#define MCCS_MAX_SLICE_PER_CHUNK 2  

#endif