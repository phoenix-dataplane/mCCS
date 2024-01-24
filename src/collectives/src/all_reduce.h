#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

namespace {
template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__

    void
    runRing(mccsDevWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps * WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    mccsDevRing *ring = &mccsShmem.channel.ring;
    int ringIx = ring->index;
    const ssize_t chunkSize =
        int(Proto::calcBytePerStep() / sizeof(T) *
            (Proto::Id == MCCS_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const int nranks = mccsShmem.comm.nRanks;
    const ssize_t loopSize = nChannels * nranks * chunkSize;
    const ssize_t size = args->count;

    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff,
        args->redOpArg);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize;
        if (Proto::Id == MCCS_PROTO_SIMPLE) {
            realChunkSize =
                min(chunkSize, divUp(size - gridOffset, nChannels * nranks));
            realChunkSize =
                roundUp(realChunkSize,
                        (nthreads - WARP_SIZE) * sizeof(uint64_t) / sizeof(T));
        }
        realChunkSize = int(realChunkSize);

        auto calcOffset = [&] __device__(int chunk) -> ssize_t {
            return gridOffset + bid * nranks * realChunkSize +
                   chunk * realChunkSize;
        };
        auto modRanks = [&] __device__(int r) -> int {
            return r - (r >= nranks ? nranks : 0);
        };

        ssize_t offset;
        int nelem;
        int chunk;

        // step 0: push data to next GPU
        chunk = modRanks(ringIx + nranks - 1);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size - offset);
        prims.send(offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j = 2; j < nranks; ++j) {
            chunk = modRanks(ringIx + nranks - j);
            offset = calcOffset(chunk);
            nelem = min(realChunkSize, size - offset);
            prims.recvReduceSend(offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final
        // result that we store in this data and push to the next GPU
        chunk = ringIx + 0;
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size - offset);
        prims.directRecvReduceCopySend(offset, offset, offset, nelem,
                                       /*postOp=*/true);

        // k-2 steps: copy to next GPU
        for (int j = 1; j < nranks - 1; ++j) {
            chunk = modRanks(ringIx + nranks - j);
            offset = calcOffset(chunk);
            nelem = min(realChunkSize, size - offset);
            prims.directRecvCopySend(offset, offset, nelem);
        }

        // Make final copy from buffer to dest.
        chunk = modRanks(ringIx + 1);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size - offset);
        prims.directRecv(offset, nelem);
    }
}
}  // namespace

template <typename T, typename RedOp>
struct RunWorkElement<mccsFuncAllReduce, T, RedOp, MCCS_ALGO_RING,
                      MCCS_PROTO_SIMPLE> {
    __device__ __forceinline__ void run(mccsDevWorkElem *args) {
        using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS,
                                  ALLREDUCE_SLICESTEPS>;
        runRing<T, RedOp, Proto>(args);
    }
};
