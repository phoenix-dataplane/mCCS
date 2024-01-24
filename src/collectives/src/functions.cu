#include "devcomm.h"
#include "collectives.h"
#include "common.h"

__shared__ mccsShmemData mccsShmem;

#define MCCS_FUNC5(func, algo, devredop, type, nullify) \
  MACRO_IF(nullify, nullptr, MCCS_FUNC_NAME(func, algo, SIMPLE, devredop, type))

#define MCCS_FUNC4(func, devredop, type, nullify) \
  MCCS_FUNC5(func, RING,    devredop, type, nullify)

#if defined(__CUDA_BF16_TYPES_EXIST__)
// Must be consistent with ncclDataType_t
#define MCCS_FUNCS3A(func, devredop, nullForFloat) \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, uint8_t, 0), \
  MCCS_FUNC4(func, devredop, int32_t, 0), \
  MCCS_FUNC4(func, devredop, uint32_t, 0), \
  MCCS_FUNC4(func, devredop, int64_t, 0), \
  MCCS_FUNC4(func, devredop, uint64_t, 0), \
  MCCS_FUNC4(func, devredop, half, nullForFloat), \
  MCCS_FUNC4(func, devredop, float, nullForFloat), \
  MCCS_FUNC4(func, devredop, double, nullForFloat), \
  MCCS_FUNC4(func, devredop, __nv_bfloat16, nullForFloat)
#define MCCS_FUNCS3B(func, devredop) \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0)
#else
// Must be consistent with mccsDataType_t
#define MCCS_FUNCS3A(func, devredop, nullForFloat) \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, uint8_t, 0), \
  MCCS_FUNC4(func, devredop, int32_t, 0), \
  MCCS_FUNC4(func, devredop, uint32_t, 0), \
  MCCS_FUNC4(func, devredop, int64_t, 0), \
  MCCS_FUNC4(func, devredop, uint64_t, 0), \
  MCCS_FUNC4(func, devredop, half, nullForFloat), \
  MCCS_FUNC4(func, devredop, float, nullForFloat), \
  MCCS_FUNC4(func, devredop, double, nullForFloat)
#define MCCS_FUNCS3B(func, devredop) \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0), \
  MCCS_FUNC4(func, devredop, int8_t, 0)
#endif

// Must be consistent with ncclRedOp_t
#define MCCS_FUNCS2A(func) \
  MCCS_FUNCS3A(func, Sum,        /*nullForFloat=*/0), \
  MCCS_FUNCS3A(func, Prod,       /*nullForFloat=*/0), \
  MCCS_FUNCS3A(func, Max,        /*nullForFloat=*/0), \
  MCCS_FUNCS3A(func, Min,        /*nullForFloat=*/0), \
  MCCS_FUNCS3A(func, PreMulSum,  /*nullForFloat=*/0), \
  MCCS_FUNCS3A(func, SumPostDiv, /*nullForFloat=*/1)

#define MCCS_FUNCS2B(func) \
  MCCS_FUNCS3B(func, Sum), \
  MCCS_FUNCS3B(func, Sum), \
  MCCS_FUNCS3B(func, Sum), \
  MCCS_FUNCS3B(func, Sum), \
  MCCS_FUNCS3B(func, Sum), \
  MCCS_FUNCS3B(func, Sum)

#define MCCS_FUNCS2C(func) \
  MCCS_FUNCS3A(func, Sum,        /*nullForFloat=*/0), \
  MCCS_FUNCS3A(func, Prod,       /*nullForFloat=*/0)

// Must be consistent with the mccsFuncSet enum
__device__ mccsDevKern_t mccsDevFuncs[1+mccsNumTypes+MCCS_NUM_FUNCTIONS*mccsNumDevRedOps*mccsNumTypes*MCCS_NUM_ALGORITHMS*MCCS_NUM_PROTOCOLS] = {
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if __CUDA_ARCH__
  MCCS_FUNCS2B(AllGather),
  MCCS_FUNCS2C(AllReduce),
#endif
};

// Workaround for https://reviews.llvm.org/D55580
__device__ void ncclWorkaroundClangD55580() {}
