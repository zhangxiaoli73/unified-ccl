#include "algorithm.h"
#include "symmetric_context.h"
#include "../device/reduce_kernel.hpp"
#include "../include/comm.h"
#include "../include/channel.h"
#include "../misc/debug.h"

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>

namespace uccl {

/* Symmetric Memory AllReduce implementation.
 *
 * Single-node only. All GPUs have P2P mapped access to each other's
 * device memory via Level Zero IPC handles. Each GPU reads from all
 * peers and reduces locally — no explicit send/recv needed.
 *
 * Advantages: no channel/proxy/FIFO overhead, kernel-only execution.
 * Limitations: intra-node only, P2P read bandwidth limited by PCIe/UPI. */

template <typename T, typename RedOp>
void runSymmetricAllReduce(
    const SymmetricMemoryContext& ctx,
    const T* sendbuff, T* recvbuff,
    size_t count, int nGpus, int myGpu,
    sycl::nd_item<1> item)
{
    int lid = item.get_local_id(0);
    int groupSize = item.get_local_range(0);
    RedOp op;

    /* Each work-item processes a strided portion of elements */
    for (size_t i = lid; i < count; i += groupSize) {
        /* Initialize accumulator from local sendbuff */
        T acc = sendbuff[i];

        /* Read from all other GPUs' symmetric buffers and reduce */
        for (int g = 0; g < nGpus; g++) {
            if (g == myGpu) continue;
            const T* remoteBuf = static_cast<const T*>(ctx.remoteBuffs[g]);
            T remoteVal = remoteBuf[i];  /* Direct P2P read */
            acc = op(acc, remoteVal);
        }

        recvbuff[i] = acc;
    }
}

/* Launch Symmetric Memory AllReduce as SYCL kernel. */
template <typename T>
ucclResult_t launchSymmetricAllReduce(
    sycl::queue& queue,
    const T* sendbuff,
    T* recvbuff,
    size_t count,
    int nGpus,
    int myGpu,
    const SymmetricMemoryContext& ctx)
{
    /* Check precondition: single-node only */
    if (ctx.nGpus < 1) {
        UCCL_LOG(ERROR, "SymmetricAllReduce: invalid nGpus=%d", ctx.nGpus);
        return ucclInvalidArgument;
    }

    if (nGpus == 1) {
        /* Single GPU: just copy */
        if (sendbuff != recvbuff) {
            queue.memcpy(recvbuff, sendbuff, count * sizeof(T)).wait();
        }
        return ucclSuccess;
    }

    size_t workGroupSize = 256;
    size_t globalSize = ((count + workGroupSize - 1) / workGroupSize)
                        * workGroupSize;
    /* Cap global size to avoid excessive over-subscription */
    if (globalSize > 65536) globalSize = 65536;

    SymmetricMemoryContext ctxCopy = ctx;

    queue.submit([&](sycl::handler& cgh) {
        const T* send = sendbuff;
        T* recv = recvbuff;
        size_t cnt = count;
        int ng = nGpus;
        int mg = myGpu;
        SymmetricMemoryContext c = ctxCopy;

        cgh.parallel_for(
            sycl::nd_range<1>(globalSize, workGroupSize),
            [=](sycl::nd_item<1> item) {
                runSymmetricAllReduce<T, ReduceSum<T>>(
                    c, send, recv, cnt, ng, mg, item);
            });
    });

    return ucclSuccess;
}

/* Explicit template instantiations */
template ucclResult_t launchSymmetricAllReduce<sycl::half>(
    sycl::queue&, const sycl::half*, sycl::half*,
    size_t, int, int, const SymmetricMemoryContext&);

template ucclResult_t launchSymmetricAllReduce<sycl::ext::oneapi::bfloat16>(
    sycl::queue&, const sycl::ext::oneapi::bfloat16*,
    sycl::ext::oneapi::bfloat16*,
    size_t, int, int, const SymmetricMemoryContext&);

/* ============================================================
 * Symmetric Memory AllGather
 *
 * Each GPU reads its peers' sendbuffs directly via P2P and writes
 * them into the correct position in its local recvbuff.
 * No explicit send/recv needed — just P2P reads.
 * ============================================================ */

template <typename T>
void runSymmetricAllGather(
    const SymmetricMemoryContext& ctx,
    const T* sendbuff, T* recvbuff,
    size_t sendcount, int nGpus, int myGpu,
    sycl::nd_item<1> item)
{
    int lid = item.get_local_id(0);
    int stride = item.get_local_range(0) * item.get_group_range(0);
    int gid = item.get_global_id(0);

    /* Copy local data to its slot in recvbuff */
    for (size_t i = gid; i < sendcount; i += stride) {
        recvbuff[myGpu * sendcount + i] = sendbuff[i];
    }

    /* Read each peer's data from their symmetric buffer */
    for (int g = 0; g < nGpus; g++) {
        if (g == myGpu) continue;
        const T* remoteBuf = static_cast<const T*>(ctx.remoteBuffs[g]);
        for (size_t i = gid; i < sendcount; i += stride) {
            recvbuff[g * sendcount + i] = remoteBuf[i];
        }
    }
}

template <typename T>
ucclResult_t launchSymmetricAllGather(
    sycl::queue& queue,
    const T* sendbuff,
    T* recvbuff,
    size_t sendcount,
    int nGpus,
    int myGpu,
    const SymmetricMemoryContext& ctx)
{
    if (nGpus == 1) {
        if (sendbuff != recvbuff) {
            queue.memcpy(recvbuff, sendbuff, sendcount * sizeof(T)).wait();
        }
        return ucclSuccess;
    }

    size_t workGroupSize = 256;
    size_t globalSize = ((sendcount + workGroupSize - 1) / workGroupSize)
                        * workGroupSize;
    if (globalSize > 65536) globalSize = 65536;

    SymmetricMemoryContext ctxCopy = ctx;

    queue.submit([&](sycl::handler& cgh) {
        const T* send = sendbuff;
        T* recv = recvbuff;
        size_t cnt = sendcount;
        int ng = nGpus;
        int mg = myGpu;
        SymmetricMemoryContext c = ctxCopy;

        cgh.parallel_for(
            sycl::nd_range<1>(globalSize, workGroupSize),
            [=](sycl::nd_item<1> item) {
                runSymmetricAllGather<T>(c, send, recv, cnt, ng, mg, item);
            });
    });

    return ucclSuccess;
}

template ucclResult_t launchSymmetricAllGather<sycl::half>(
    sycl::queue&, const sycl::half*, sycl::half*,
    size_t, int, int, const SymmetricMemoryContext&);

template ucclResult_t launchSymmetricAllGather<sycl::ext::oneapi::bfloat16>(
    sycl::queue&, const sycl::ext::oneapi::bfloat16*,
    sycl::ext::oneapi::bfloat16*,
    size_t, int, int, const SymmetricMemoryContext&);

/* ============================================================
 * Symmetric Memory ReduceScatter
 *
 * Input: each GPU has recvcount * nGpus elements.
 * Output: each GPU gets recvcount elements that are the sum of
 * the corresponding chunk across all GPUs.
 *
 * Each GPU reads chunk[myGpu] from every peer's buffer via P2P,
 * reduces locally, and writes to recvbuff.
 * ============================================================ */

template <typename T, typename RedOp>
void runSymmetricReduceScatter(
    const SymmetricMemoryContext& ctx,
    const T* sendbuff, T* recvbuff,
    size_t recvcount, int nGpus, int myGpu,
    sycl::nd_item<1> item)
{
    int gid = item.get_global_id(0);
    int stride = item.get_local_range(0) * item.get_group_range(0);
    RedOp op;

    /* My chunk is at offset [myGpu * recvcount .. (myGpu+1) * recvcount) */
    for (size_t i = gid; i < recvcount; i += stride) {
        /* Start with local chunk */
        T acc = sendbuff[myGpu * recvcount + i];

        /* Reduce corresponding chunk from all peers */
        for (int g = 0; g < nGpus; g++) {
            if (g == myGpu) continue;
            const T* remoteBuf = static_cast<const T*>(ctx.remoteBuffs[g]);
            T remoteVal = remoteBuf[myGpu * recvcount + i];
            acc = op(acc, remoteVal);
        }

        recvbuff[i] = acc;
    }
}

template <typename T>
ucclResult_t launchSymmetricReduceScatter(
    sycl::queue& queue,
    const T* sendbuff,
    T* recvbuff,
    size_t recvcount,
    int nGpus,
    int myGpu,
    const SymmetricMemoryContext& ctx)
{
    if (nGpus == 1) {
        if (sendbuff != recvbuff) {
            queue.memcpy(recvbuff, sendbuff, recvcount * sizeof(T)).wait();
        }
        return ucclSuccess;
    }

    size_t workGroupSize = 256;
    size_t globalSize = ((recvcount + workGroupSize - 1) / workGroupSize)
                        * workGroupSize;
    if (globalSize > 65536) globalSize = 65536;

    SymmetricMemoryContext ctxCopy = ctx;

    queue.submit([&](sycl::handler& cgh) {
        const T* send = sendbuff;
        T* recv = recvbuff;
        size_t cnt = recvcount;
        int ng = nGpus;
        int mg = myGpu;
        SymmetricMemoryContext c = ctxCopy;

        cgh.parallel_for(
            sycl::nd_range<1>(globalSize, workGroupSize),
            [=](sycl::nd_item<1> item) {
                runSymmetricReduceScatter<T, ReduceSum<T>>(
                    c, send, recv, cnt, ng, mg, item);
            });
    });

    return ucclSuccess;
}

template ucclResult_t launchSymmetricReduceScatter<sycl::half>(
    sycl::queue&, const sycl::half*, sycl::half*,
    size_t, int, int, const SymmetricMemoryContext&);

template ucclResult_t launchSymmetricReduceScatter<sycl::ext::oneapi::bfloat16>(
    sycl::queue&, const sycl::ext::oneapi::bfloat16*,
    sycl::ext::oneapi::bfloat16*,
    size_t, int, int, const SymmetricMemoryContext&);

} /* namespace uccl */
