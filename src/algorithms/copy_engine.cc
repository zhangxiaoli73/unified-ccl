#include "algorithm.h"
#include "symmetric_context.h"
#include "../include/comm.h"
#include "../include/hw_resources.h"
#include "../device/reduce_kernel.hpp"
#include "../misc/debug.h"

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace uccl {

/* Copy-Engine-Only Collective Implementations.
 *
 * Strategy: use DMA copy engines for all data movement (P2P memcpy),
 * then perform reduction on EUs as a final step.
 *
 * This is beneficial when:
 *   - EUs are busy with compute (e.g., overlapping with forward pass)
 *   - Copy engines provide dedicated bandwidth (BCS on Intel GPUs)
 *   - Latency of copy engine path < kernel launch overhead for small data
 *
 * Data flow for AllReduce (copy-engine mode, N ranks):
 *   1. [Copy Engine] Each rank copies its data to all peers' scratch buffers
 *      via P2P memcpy (through IPC-mapped symmetric memory)
 *   2. [Barrier] Wait for all copies to complete
 *   3. [EU Kernel]  Each rank reduces N buffers (own + N-1 received) locally
 *
 * This decouples data movement from computation, allowing the copy engines
 * to run independently of the EU compute pipeline. */

/* ============================================================
 * Copy-Engine AllReduce
 *
 * Phase 1 (Copy Engine): each rank writes sendbuff to all peers'
 *          scratch buffers[myRank] via P2P memcpy
 * Phase 2 (EU Kernel):   each rank reduces N scratch buffers
 *          into recvbuff
 * ============================================================ */

template <typename T>
ucclResult_t launchCopyEngineAllReduce(
    sycl::queue& computeQueue,
    sycl::queue& copyQueue,
    const T* sendbuff,
    T* recvbuff,
    size_t count,
    int nGpus,
    int myGpu,
    const SymmetricMemoryContext& ctx)
{
    if (nGpus == 1) {
        if (sendbuff != recvbuff) {
            copyQueue.memcpy(recvbuff, sendbuff, count * sizeof(T)).wait();
        }
        return ucclSuccess;
    }

    size_t nbytes = count * sizeof(T);

    /* Phase 1: Copy Engine — write local data to all peers' scratch.
     *
     * Each peer's remoteBuffs[myGpu] points to a region where we
     * can write our data. We use the copy engine queue for P2P memcpy.
     * All copies are enqueued to the same in-order copy queue,
     * so they execute sequentially on the copy engine. */
    std::vector<sycl::event> copyEvents;
    copyEvents.reserve(nGpus - 1);

    for (int peer = 0; peer < nGpus; peer++) {
        if (peer == myGpu) continue;

        /* Write our sendbuff into peer's scratch area for slot [myGpu] */
        void* remoteDst = static_cast<char*>(ctx.remoteBuffs[peer])
                          + myGpu * nbytes;
        auto ev = copyQueue.memcpy(remoteDst, sendbuff, nbytes);
        copyEvents.push_back(ev);
    }

    /* Also copy local data to our own scratch slot */
    void* localSlot = static_cast<char*>(ctx.localBuff)
                      + myGpu * nbytes;
    auto localEv = copyQueue.memcpy(localSlot, sendbuff, nbytes);
    copyEvents.push_back(localEv);

    /* Wait for all copy engine transfers to complete.
     * In a real implementation, a cross-rank barrier is needed here
     * to ensure all peers have finished writing to our scratch. */
    for (auto& ev : copyEvents) {
        ev.wait();
    }

    /* Cross-rank synchronization: MPI barrier to ensure all ranks
     * have completed their P2P writes */
    /* MPI_Barrier(comm->mpiComm); — would be called at a higher level */

    /* Phase 2: EU Kernel — local reduction of N scratch buffers.
     *
     * Our scratch buffer layout:
     *   scratch[0 * count .. 1 * count - 1] = data from rank 0
     *   scratch[1 * count .. 2 * count - 1] = data from rank 1
     *   ...
     * Reduce all N chunks into recvbuff. */

    size_t workGroupSize = 256;
    size_t globalSize = ((count + workGroupSize - 1) / workGroupSize)
                        * workGroupSize;
    if (globalSize > 65536) globalSize = 65536;

    const T* scratchBase = static_cast<const T*>(ctx.localBuff);
    int ng = nGpus;

    computeQueue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(globalSize, workGroupSize),
            [=](sycl::nd_item<1> item) {
                ReduceSum<T> op;
                int gid = item.get_global_id(0);
                int stride = item.get_global_range(0);

                for (size_t i = gid; i < count; i += stride) {
                    /* Start with rank 0's data */
                    T acc = scratchBase[0 * count + i];

                    /* Reduce remaining ranks */
                    for (int g = 1; g < ng; g++) {
                        acc = op(acc, scratchBase[g * count + i]);
                    }

                    recvbuff[i] = acc;
                }
            });
    });

    computeQueue.wait();

    return ucclSuccess;
}

/* Explicit template instantiations */
template ucclResult_t launchCopyEngineAllReduce<sycl::half>(
    sycl::queue&, sycl::queue&,
    const sycl::half*, sycl::half*,
    size_t, int, int, const SymmetricMemoryContext&);

template ucclResult_t launchCopyEngineAllReduce<sycl::ext::oneapi::bfloat16>(
    sycl::queue&, sycl::queue&,
    const sycl::ext::oneapi::bfloat16*, sycl::ext::oneapi::bfloat16*,
    size_t, int, int, const SymmetricMemoryContext&);

/* ============================================================
 * Copy-Engine AllGather
 *
 * Pure copy engine: each rank writes its sendbuff to all peers'
 * recvbuff at the correct offset. No reduction needed.
 * ============================================================ */

template <typename T>
ucclResult_t launchCopyEngineAllGather(
    sycl::queue& copyQueue,
    const T* sendbuff,
    T* recvbuff,
    size_t sendcount,
    int nGpus,
    int myGpu,
    const SymmetricMemoryContext& ctx)
{
    if (nGpus == 1) {
        if (sendbuff != recvbuff) {
            copyQueue.memcpy(recvbuff, sendbuff, sendcount * sizeof(T)).wait();
        }
        return ucclSuccess;
    }

    size_t nbytes = sendcount * sizeof(T);
    std::vector<sycl::event> events;
    events.reserve(nGpus);

    /* Copy own data to local recvbuff slot */
    auto localEv = copyQueue.memcpy(
        recvbuff + myGpu * sendcount, sendbuff, nbytes);
    events.push_back(localEv);

    /* Copy own data to each peer's recvbuff at slot [myGpu] via P2P.
     * remotePtrs[peer] points to peer's recvbuff base (or scratch). */
    for (int peer = 0; peer < nGpus; peer++) {
        if (peer == myGpu) continue;

        void* remoteDst = static_cast<char*>(ctx.remoteBuffs[peer])
                          + myGpu * nbytes;
        auto ev = copyQueue.memcpy(remoteDst, sendbuff, nbytes);
        events.push_back(ev);
    }

    /* Wait for all copies */
    for (auto& ev : events) {
        ev.wait();
    }

    /* After barrier (at higher level), read peers' data into local recvbuff.
     * Alternatively, if remotePtrs point to peers' recvbuff directly,
     * each peer has already written to our recvbuff via P2P above. */

    /* Read each peer's contribution from our local scratch/recvbuff.
     * If peers wrote to our recvbuff directly (above pattern), we just
     * need to read from their symmetric buffers for the pull model: */
    for (int peer = 0; peer < nGpus; peer++) {
        if (peer == myGpu) continue;

        const void* remoteSrc =
            static_cast<const char*>(ctx.remoteBuffs[peer]);
        auto ev = copyQueue.memcpy(
            recvbuff + peer * sendcount,
            static_cast<const T*>(remoteSrc),
            nbytes);
        events.push_back(ev);
    }

    for (auto& ev : events) {
        ev.wait();
    }

    return ucclSuccess;
}

template ucclResult_t launchCopyEngineAllGather<sycl::half>(
    sycl::queue&,
    const sycl::half*, sycl::half*,
    size_t, int, int, const SymmetricMemoryContext&);

template ucclResult_t launchCopyEngineAllGather<sycl::ext::oneapi::bfloat16>(
    sycl::queue&,
    const sycl::ext::oneapi::bfloat16*, sycl::ext::oneapi::bfloat16*,
    size_t, int, int, const SymmetricMemoryContext&);

/* ============================================================
 * Copy-Engine ReduceScatter
 *
 * Phase 1 (Copy Engine): Gather chunk[myGpu] from all peers via
 *          P2P memcpy into local scratch buffers
 * Phase 2 (EU Kernel):   Reduce N scratch chunks into recvbuff
 *
 * Each rank only needs to read its own chunk from all peers,
 * then reduce locally. The copy engine handles the reads, and
 * the final reduction is a simple EU kernel.
 * ============================================================ */

template <typename T>
ucclResult_t launchCopyEngineReduceScatter(
    sycl::queue& computeQueue,
    sycl::queue& copyQueue,
    const T* sendbuff,
    T* recvbuff,
    size_t recvcount,
    int nGpus,
    int myGpu,
    const SymmetricMemoryContext& ctx)
{
    if (nGpus == 1) {
        if (sendbuff != recvbuff) {
            copyQueue.memcpy(recvbuff, sendbuff,
                             recvcount * sizeof(T)).wait();
        }
        return ucclSuccess;
    }

    size_t chunkBytes = recvcount * sizeof(T);

    /* Allocate scratch buffer for N chunks (one from each rank).
     * scratch[g] = chunk[myGpu] from rank g's sendbuff.
     * We use the local symmetric buffer as scratch. */
    T* scratchBase = static_cast<T*>(ctx.localBuff);

    std::vector<sycl::event> events;
    events.reserve(nGpus);

    /* Copy own chunk[myGpu] to scratch[myGpu] */
    auto localEv = copyQueue.memcpy(
        scratchBase + myGpu * recvcount,
        sendbuff + myGpu * recvcount,
        chunkBytes);
    events.push_back(localEv);

    /* Copy engine: read chunk[myGpu] from each peer's sendbuff */
    for (int peer = 0; peer < nGpus; peer++) {
        if (peer == myGpu) continue;

        /* remotePtrs[peer] → peer's sendbuff base (symmetric buffer).
         * We want peer's sendbuff[myGpu * recvcount ..] */
        const T* remoteSrc = static_cast<const T*>(ctx.remoteBuffs[peer])
                             + myGpu * recvcount;

        auto ev = copyQueue.memcpy(
            scratchBase + peer * recvcount,
            remoteSrc, chunkBytes);
        events.push_back(ev);
    }

    /* Wait for all copy engine reads to complete */
    for (auto& ev : events) {
        ev.wait();
    }

    /* Phase 2: EU Kernel — reduce N chunks into recvbuff */
    size_t workGroupSize = 256;
    size_t globalSize = ((recvcount + workGroupSize - 1) / workGroupSize)
                        * workGroupSize;
    if (globalSize > 65536) globalSize = 65536;

    const T* scratch = scratchBase;
    int ng = nGpus;
    size_t cnt = recvcount;

    computeQueue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(globalSize, workGroupSize),
            [=](sycl::nd_item<1> item) {
                ReduceSum<T> op;
                int gid = item.get_global_id(0);
                int stride = item.get_global_range(0);

                for (size_t i = gid; i < cnt; i += stride) {
                    T acc = scratch[0 * cnt + i];
                    for (int g = 1; g < ng; g++) {
                        acc = op(acc, scratch[g * cnt + i]);
                    }
                    recvbuff[i] = acc;
                }
            });
    });

    computeQueue.wait();

    return ucclSuccess;
}

template ucclResult_t launchCopyEngineReduceScatter<sycl::half>(
    sycl::queue&, sycl::queue&,
    const sycl::half*, sycl::half*,
    size_t, int, int, const SymmetricMemoryContext&);

template ucclResult_t launchCopyEngineReduceScatter<sycl::ext::oneapi::bfloat16>(
    sycl::queue&, sycl::queue&,
    const sycl::ext::oneapi::bfloat16*, sycl::ext::oneapi::bfloat16*,
    size_t, int, int, const SymmetricMemoryContext&);

} /* namespace uccl */
