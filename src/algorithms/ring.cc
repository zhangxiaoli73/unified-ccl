#include "algorithm.h"
#include "../device/primitives.hpp"
#include "../include/comm.h"
#include "../include/channel.h"
#include "../misc/debug.h"

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>

namespace uccl {

/* Ring AllReduce implementation.
 *
 * Mirrors NCCL src/device/sendrecv.h ring allreduce kernel.
 * Two phases:
 *   1. Reduce-Scatter: propagate along ring, each rank accumulates
 *   2. AllGather: broadcast final results along ring
 *
 * The algorithm is protocol-agnostic: it uses Primitives (send/recv/reduce)
 * which are implemented differently by Simple and LL128 protocols. */

template <typename T, typename RedOp, typename Proto>
void runRingAllReduce(
    Primitives<T, RedOp, Proto>& prims,
    const T* sendbuff, T* recvbuff,
    size_t count, int nranks, int rank)
{
    if (nranks == 1) {
        /* Single rank: just copy if needed */
        if (sendbuff != recvbuff) {
            prims.copySend(0, 0, static_cast<int>(count));
        }
        return;
    }

    size_t chunkCount = count / nranks;
    size_t remainder = count % nranks;

    /* Phase 1: Reduce-Scatter
     * Each step: receive data from prev, reduce with local, send to next.
     * After nranks-1 steps, each rank holds the fully reduced chunk. */
    for (int step = 0; step < nranks - 1; step++) {
        /* Compute chunk indices with modular arithmetic */
        int sendChunk = ((rank - step) % nranks + nranks) % nranks;
        int recvChunk = ((rank - step - 1) % nranks + nranks) % nranks;

        intptr_t sendOff = static_cast<intptr_t>(sendChunk * chunkCount);
        intptr_t recvOff = static_cast<intptr_t>(recvChunk * chunkCount);

        /* Handle remainder: last chunk may be larger */
        int sendEltN = static_cast<int>(
            (sendChunk == nranks - 1) ? chunkCount + remainder : chunkCount);
        int recvEltN = static_cast<int>(
            (recvChunk == nranks - 1) ? chunkCount + remainder : chunkCount);

        (void)sendEltN; /* used implicitly via sendOff */

        if (step == 0) {
            /* First step: receive + reduce with input + copy to output + send */
            prims.recvReduceCopySend(sendOff, recvOff, recvEltN);
        } else {
            /* Subsequent steps: receive + reduce + send (in-place on output) */
            prims.recvReduceSend(recvOff, recvEltN);
        }
    }

    /* Phase 2: AllGather
     * Each step: receive fully reduced chunk from prev, copy to output, send to next.
     * After nranks-1 steps, all ranks have the complete result. */
    for (int step = 0; step < nranks - 1; step++) {
        int chunk = ((rank + 1 - step) % nranks + nranks) % nranks;
        intptr_t offset = static_cast<intptr_t>(chunk * chunkCount);
        int eltN = static_cast<int>(
            (chunk == nranks - 1) ? chunkCount + remainder : chunkCount);

        prims.recvCopySend(offset, eltN);
    }
}

/* Launch Ring AllReduce as SYCL kernel.
 *
 * This dispatches to the correct protocol-specialized kernel
 * based on the protocol type configured for the channel. */
template <typename T>
ucclResult_t launchRingAllReduce(
    sycl::queue& queue,
    const T* sendbuff,
    T* recvbuff,
    size_t count,
    int nranks,
    int rank,
    ucclChannel& channel,
    int protocol)
{
    /* Determine work-group size based on protocol */
    size_t workGroupSize = (protocol == UCCL_PROTO_LL128) ? 256 : 512;

    queue.submit([&](sycl::handler& cgh) {
        /* Capture variables for the kernel */
        const T* send = sendbuff;
        T* recv = recvbuff;
        size_t cnt = count;
        int nr = nranks;
        int rk = rank;
        int proto = protocol;

        cgh.parallel_for(
            sycl::nd_range<1>(workGroupSize, workGroupSize),
            [=](sycl::nd_item<1> item) {
                /* Create Primitives instance for this work-item.
                 * In a full implementation, this would be initialized
                 * with channel connection info and buffer pointers. */
                if (proto == UCCL_PROTO_SIMPLE) {
                    Primitives<T, ReduceSum<T>, ProtoSimple> prims(
                        send, recv, item);
                    runRingAllReduce(prims, send, recv, cnt, nr, rk);
                } else {
                    Primitives<T, ReduceSum<T>, ProtoLL128> prims(
                        send, recv, item);
                    runRingAllReduce(prims, send, recv, cnt, nr, rk);
                }
            });
    });

    return ucclSuccess;
}

/* Explicit template instantiations for supported types */
template ucclResult_t launchRingAllReduce<sycl::half>(
    sycl::queue&, const sycl::half*, sycl::half*,
    size_t, int, int, ucclChannel&, int);

template ucclResult_t launchRingAllReduce<sycl::ext::oneapi::bfloat16>(
    sycl::queue&, const sycl::ext::oneapi::bfloat16*,
    sycl::ext::oneapi::bfloat16*,
    size_t, int, int, ucclChannel&, int);


/* ============================================================
 * Net Ring AllReduce — inter-node (via FIFO + bounce buffer + proxy)
 *
 * Persistent kernel: 1 workgroup on 1 Xe Core (0.6% EU on B60).
 * Kernel loops internally over all ring steps.
 * Communication with proxy through ucclConnFifo.
 * ============================================================ */

/* Build NetKernelArgs from channel state (called on host). */
static NetKernelArgs buildNetKernelArgs(ucclChannel& channel) {
    NetKernelArgs args{};

    args.sendFifoHead = &channel.sendFifo->head;
    args.sendFifoTail = &channel.sendFifo->tail;
    args.recvFifoHead = &channel.recvFifo->head;
    args.recvFifoTail = &channel.recvFifo->tail;

    args.sendEntries = channel.sendFifo->entries;
    args.recvEntries = channel.recvFifo->entries;

    for (int i = 0; i < UCCL_STEPS; i++) {
        args.bounceSendBuffs[i] = channel.bounceSend.buffs[i];
        args.bounceRecvBuffs[i] = channel.bounceRecv.buffs[i];
        args.bounceSendMhandles[i] = channel.bounceSend.mhandles[i];
        args.bounceRecvMhandles[i] = channel.bounceRecv.mhandles[i];
    }

    return args;
}

template <typename T>
ucclResult_t launchRingAllReduceNet(
    sycl::queue& queue,
    const T* sendbuff, T* recvbuff,
    size_t count, int nranks, int rank,
    ucclChannel& channel)
{
    if (channel.sendFifo == nullptr || channel.recvFifo == nullptr) {
        UCCL_LOG(ERROR, "Net kernel: FIFOs not allocated for channel %d",
                 channel.id);
        return ucclInternalError;
    }

    /* Build kernel args on host (POD, captured by value) */
    NetKernelArgs args = buildNetKernelArgs(channel);

    /* Persistent kernel: 1 workgroup = 1 Xe Core */
    constexpr size_t workGroupSize = 512;

    queue.submit([&](sycl::handler& cgh) {
        const T* send = sendbuff;
        T* recv = recvbuff;
        size_t cnt = count;
        int nr = nranks;
        int rk = rank;
        NetKernelArgs kargs = args;

        cgh.parallel_for(
            sycl::nd_range<1>(workGroupSize, workGroupSize),
            [=](sycl::nd_item<1> item) {
                NetPrimitives<T, ReduceSum<T>> prims(
                    send, recv, kargs, item);

                if (nr == 1) return;

                size_t chunkCount = cnt / nr;
                size_t remainder = cnt % nr;
                int step = 0;

                /* Phase 1: Reduce-Scatter (nr - 1 steps) */
                for (int s = 0; s < nr - 1; s++, step++) {
                    int sendChunk = ((rk - s) % nr + nr) % nr;
                    int recvChunk = ((rk - s - 1) % nr + nr) % nr;

                    size_t sendEltN = (sendChunk == nr - 1)
                        ? chunkCount + remainder : chunkCount;
                    size_t recvEltN = (recvChunk == nr - 1)
                        ? chunkCount + remainder : chunkCount;
                    size_t sendBytes = sendEltN * sizeof(T);
                    size_t recvBytes = recvEltN * sizeof(T);

                    const T* sendSrc = (s == 0)
                        ? send + sendChunk * chunkCount
                        : recv + sendChunk * chunkCount;

                    prims.sendViaBounce(sendSrc, sendBytes, step);
                    prims.postRecv(recvBytes, step);

                    /* Overlap: reduce previous step's recv data */
                    if (s > 0) {
                        int prevRecvChunk = ((rk - s) % nr + nr) % nr;
                        size_t prevRecvEltN = (prevRecvChunk == nr - 1)
                            ? chunkCount + remainder : chunkCount;
                        prims.waitRecv(step - 1);
                        prims.reduceFromBounceRecv(
                            recv + prevRecvChunk * chunkCount,
                            recv + prevRecvChunk * chunkCount,
                            prevRecvEltN, step - 1);
                    }
                }

                /* Drain last reduce-scatter recv */
                {
                    int lastChunk = ((rk - (nr - 1)) % nr + nr) % nr;
                    size_t lastEltN = (lastChunk == nr - 1)
                        ? chunkCount + remainder : chunkCount;
                    prims.waitRecv(step - 1);
                    prims.reduceFromBounceRecv(
                        recv + lastChunk * chunkCount,
                        recv + lastChunk * chunkCount,
                        lastEltN, step - 1);
                }

                /* Phase 2: AllGather (nr - 1 steps) */
                for (int s = 0; s < nr - 1; s++, step++) {
                    int sendChunk = ((rk + 1 - s) % nr + nr) % nr;
                    int recvChunk = ((rk - s) % nr + nr) % nr;

                    size_t sendEltN = (sendChunk == nr - 1)
                        ? chunkCount + remainder : chunkCount;
                    size_t recvEltN = (recvChunk == nr - 1)
                        ? chunkCount + remainder : chunkCount;
                    size_t sendBytes = sendEltN * sizeof(T);
                    size_t recvBytes = recvEltN * sizeof(T);

                    prims.sendViaBounce(
                        recv + sendChunk * chunkCount,
                        sendBytes, step);
                    prims.postRecv(recvBytes, step);

                    prims.waitRecv(step);
                    prims.copyFromBounceRecv(
                        recv + recvChunk * chunkCount,
                        recvEltN, step);
                }
            });
    });

    return ucclSuccess;
}

template ucclResult_t launchRingAllReduceNet<sycl::half>(
    sycl::queue&, const sycl::half*, sycl::half*,
    size_t, int, int, ucclChannel&);

template ucclResult_t launchRingAllReduceNet<sycl::ext::oneapi::bfloat16>(
    sycl::queue&, const sycl::ext::oneapi::bfloat16*,
    sycl::ext::oneapi::bfloat16*,
    size_t, int, int, ucclChannel&);


/* ============================================================
 * Ring AllGather
 *
 * This is Phase 2 of Ring AllReduce running independently.
 * Each rank contributes sendcount elements; the result is
 * the concatenation of all ranks' data (sendcount * nranks elements).
 * ============================================================ */

template <typename T, typename Proto>
void runRingAllGather(
    Primitives<T, ReduceSum<T>, Proto>& prims,
    const T* sendbuff, T* recvbuff,
    size_t sendcount, int nranks, int rank)
{
    if (nranks == 1) {
        if (sendbuff != recvbuff) {
            prims.copySend(0, 0, static_cast<int>(sendcount));
        }
        return;
    }

    size_t chunkCount = sendcount;

    /* Copy local data to its position in recvbuff */
    prims.copySend(0, static_cast<intptr_t>(rank * chunkCount),
                   static_cast<int>(chunkCount));

    /* nranks - 1 steps: receive a chunk from prev rank, place it, forward */
    for (int step = 0; step < nranks - 1; step++) {
        int chunk = ((rank - step) % nranks + nranks) % nranks;
        intptr_t offset = static_cast<intptr_t>(chunk * chunkCount);

        prims.recvCopySend(offset, static_cast<int>(chunkCount));
    }
}

/* Launch Ring AllGather as SYCL kernel. */
template <typename T>
ucclResult_t launchRingAllGather(
    sycl::queue& queue,
    const T* sendbuff,
    T* recvbuff,
    size_t sendcount,
    int nranks,
    int rank,
    ucclChannel& channel,
    int protocol)
{
    size_t workGroupSize = (protocol == UCCL_PROTO_LL128) ? 256 : 512;

    queue.submit([&](sycl::handler& cgh) {
        const T* send = sendbuff;
        T* recv = recvbuff;
        size_t cnt = sendcount;
        int nr = nranks;
        int rk = rank;
        int proto = protocol;

        cgh.parallel_for(
            sycl::nd_range<1>(workGroupSize, workGroupSize),
            [=](sycl::nd_item<1> item) {
                if (proto == UCCL_PROTO_SIMPLE) {
                    Primitives<T, ReduceSum<T>, ProtoSimple> prims(
                        send, recv, item);
                    runRingAllGather<T, ProtoSimple>(
                        prims, send, recv, cnt, nr, rk);
                } else {
                    Primitives<T, ReduceSum<T>, ProtoLL128> prims(
                        send, recv, item);
                    runRingAllGather<T, ProtoLL128>(
                        prims, send, recv, cnt, nr, rk);
                }
            });
    });

    return ucclSuccess;
}

template ucclResult_t launchRingAllGather<sycl::half>(
    sycl::queue&, const sycl::half*, sycl::half*,
    size_t, int, int, ucclChannel&, int);

template ucclResult_t launchRingAllGather<sycl::ext::oneapi::bfloat16>(
    sycl::queue&, const sycl::ext::oneapi::bfloat16*,
    sycl::ext::oneapi::bfloat16*,
    size_t, int, int, ucclChannel&, int);

/* ============================================================
 * Ring ReduceScatter
 *
 * This is Phase 1 of Ring AllReduce running independently.
 * Input: each rank has recvcount * nranks elements.
 * Output: each rank gets recvcount elements (its reduced chunk).
 * ============================================================ */

template <typename T, typename RedOp, typename Proto>
void runRingReduceScatter(
    Primitives<T, RedOp, Proto>& prims,
    const T* sendbuff, T* recvbuff,
    size_t recvcount, int nranks, int rank)
{
    if (nranks == 1) {
        if (sendbuff != recvbuff) {
            prims.copySend(0, 0, static_cast<int>(recvcount));
        }
        return;
    }

    size_t chunkCount = recvcount;

    /* nranks - 1 steps: propagate along ring, reduce each chunk */
    for (int step = 0; step < nranks - 1; step++) {
        int sendChunk = ((rank - step) % nranks + nranks) % nranks;
        int recvChunk = ((rank - step - 1) % nranks + nranks) % nranks;
        intptr_t sendOff = static_cast<intptr_t>(sendChunk * chunkCount);
        intptr_t recvOff = static_cast<intptr_t>(recvChunk * chunkCount);

        if (step == 0) {
            prims.recvReduceCopySend(sendOff, recvOff,
                                     static_cast<int>(chunkCount));
        } else {
            prims.recvReduceSend(recvOff, static_cast<int>(chunkCount));
        }
    }
}

/* Launch Ring ReduceScatter as SYCL kernel. */
template <typename T>
ucclResult_t launchRingReduceScatter(
    sycl::queue& queue,
    const T* sendbuff,
    T* recvbuff,
    size_t recvcount,
    int nranks,
    int rank,
    ucclChannel& channel,
    int protocol)
{
    size_t workGroupSize = (protocol == UCCL_PROTO_LL128) ? 256 : 512;

    queue.submit([&](sycl::handler& cgh) {
        const T* send = sendbuff;
        T* recv = recvbuff;
        size_t cnt = recvcount;
        int nr = nranks;
        int rk = rank;
        int proto = protocol;

        cgh.parallel_for(
            sycl::nd_range<1>(workGroupSize, workGroupSize),
            [=](sycl::nd_item<1> item) {
                if (proto == UCCL_PROTO_SIMPLE) {
                    Primitives<T, ReduceSum<T>, ProtoSimple> prims(
                        send, recv, item);
                    runRingReduceScatter<T, ReduceSum<T>, ProtoSimple>(
                        prims, send, recv, cnt, nr, rk);
                } else {
                    Primitives<T, ReduceSum<T>, ProtoLL128> prims(
                        send, recv, item);
                    runRingReduceScatter<T, ReduceSum<T>, ProtoLL128>(
                        prims, send, recv, cnt, nr, rk);
                }
            });
    });

    return ucclSuccess;
}

template ucclResult_t launchRingReduceScatter<sycl::half>(
    sycl::queue&, const sycl::half*, sycl::half*,
    size_t, int, int, ucclChannel&, int);

template ucclResult_t launchRingReduceScatter<sycl::ext::oneapi::bfloat16>(
    sycl::queue&, const sycl::ext::oneapi::bfloat16*,
    sycl::ext::oneapi::bfloat16*,
    size_t, int, int, ucclChannel&, int);

} /* namespace uccl */
