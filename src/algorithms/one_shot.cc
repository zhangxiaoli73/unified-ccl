#include "algorithm.h"
#include "../device/primitives.hpp"
#include "../include/comm.h"
#include "../include/channel.h"
#include "../misc/debug.h"

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>

namespace uccl {

/* One-Shot AllReduce implementation.
 *
 * Optimized for small messages (< 4KB). Each rank broadcasts its
 * full data to all peers in a single round, then locally reduces.
 *
 * Trade-off: lowest latency (1 communication step) but poor bandwidth
 * efficiency (total traffic = N * dataSize per rank). */

template <typename T, typename RedOp, typename Proto>
void runOneShotAllReduce(
    Primitives<T, RedOp, Proto>& prims,
    const T* sendbuff, T* recvbuff,
    size_t count, int nranks, int rank)
{
    if (nranks == 1) {
        if (sendbuff != recvbuff) {
            prims.copySend(0, 0, static_cast<int>(count));
        }
        return;
    }

    /* Phase 1: Send full data to every other rank */
    for (int peer = 0; peer < nranks; peer++) {
        if (peer == rank) continue;
        prims.send(0, static_cast<int>(count));
    }

    /* Phase 2: Copy local data to output as initial value */
    prims.copySend(0, 0, static_cast<int>(count));

    /* Phase 3: Receive from all peers and reduce into output */
    for (int peer = 0; peer < nranks; peer++) {
        if (peer == rank) continue;
        prims.recvReduceSend(0, static_cast<int>(count));
    }
}

/* Launch One-Shot AllReduce as SYCL kernel. */
template <typename T>
ucclResult_t launchOneShotAllReduce(
    sycl::queue& queue,
    const T* sendbuff,
    T* recvbuff,
    size_t count,
    int nranks,
    int rank,
    ucclChannel& channel,
    int protocol)
{
    size_t workGroupSize = (protocol == UCCL_PROTO_LL128) ? 256 : 512;

    queue.submit([&](sycl::handler& cgh) {
        const T* send = sendbuff;
        T* recv = recvbuff;
        size_t cnt = count;
        int nr = nranks;
        int rk = rank;
        int proto = protocol;

        cgh.parallel_for(
            sycl::nd_range<1>(workGroupSize, workGroupSize),
            [=](sycl::nd_item<1> item) {
                if (proto == UCCL_PROTO_SIMPLE) {
                    Primitives<T, ReduceSum<T>, ProtoSimple> prims(
                        send, recv, item);
                    runOneShotAllReduce(prims, send, recv, cnt, nr, rk);
                } else {
                    Primitives<T, ReduceSum<T>, ProtoLL128> prims(
                        send, recv, item);
                    runOneShotAllReduce(prims, send, recv, cnt, nr, rk);
                }
            });
    });

    return ucclSuccess;
}

/* Explicit template instantiations */
template ucclResult_t launchOneShotAllReduce<sycl::half>(
    sycl::queue&, const sycl::half*, sycl::half*,
    size_t, int, int, ucclChannel&, int);

template ucclResult_t launchOneShotAllReduce<sycl::ext::oneapi::bfloat16>(
    sycl::queue&, const sycl::ext::oneapi::bfloat16*,
    sycl::ext::oneapi::bfloat16*,
    size_t, int, int, ucclChannel&, int);

} /* namespace uccl */
