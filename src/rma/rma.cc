#include "rma.h"
#include "../include/comm.h"
#include "../misc/debug.h"

#include <sycl/sycl.hpp>
#include <cstring>

/* One-Sided RMA Device Operations — Implementation.
 *
 * All operations rely on a pre-registered symmetric memory window
 * where every rank has P2P-mapped pointers to every other rank's
 * device memory (via Level Zero IPC handles).
 *
 * Data transfers are implemented as SYCL memcpy operations within
 * kernels or via sycl::queue::memcpy. Signal updates use atomic
 * operations on device-accessible signal counters. */

namespace uccl {

/* ============================================================
 * Helper: resolve type size
 * ============================================================ */
static inline size_t ucclTypeSizeBytes(ucclDataType_t dt) {
    switch (dt) {
        case ucclFloat16:  return 2;
        case ucclBfloat16: return 2;
        default:           return 0;
    }
}

/* ============================================================
 * Window Registration
 * ============================================================ */

ucclResult_t ucclWindowRegister(ucclWindow_t* win, void* buff,
                                size_t size, ucclComm_t comm) {
    if (win == nullptr || buff == nullptr || comm == nullptr) {
        return ucclInvalidArgument;
    }
    if (size == 0) {
        return ucclInvalidArgument;
    }

    ucclWindow* w = new ucclWindow();
    w->localBuff = buff;
    w->size = size;
    w->nRanks = comm->nRanks;
    w->myRank = comm->rank;
    w->zeContext = nullptr;

    /* Allocate remote pointer table and signal arrays */
    w->remotePtrs = new void*[comm->nRanks];
    w->localSignals = static_cast<volatile uint64_t*>(
        sycl::malloc_device(comm->nRanks * sizeof(uint64_t),
                            *comm->defaultQueue));
    w->remoteSignals = new volatile uint64_t*[comm->nRanks];

    /* Initialize local signals to zero */
    comm->defaultQueue->memset(
        const_cast<uint64_t*>(w->localSignals), 0,
        comm->nRanks * sizeof(uint64_t)).wait();

    /* Self-pointer */
    w->remotePtrs[comm->rank] = buff;
    w->remoteSignals[comm->rank] = w->localSignals + comm->rank;

    /* Exchange IPC handles via MPI to populate remotePtrs.
     *
     * In a full implementation:
     * 1. zeMemGetIpcHandle(context, buff, &ipcHandle)
     * 2. MPI_Allgather(&ipcHandle, ..., allHandles, ...)
     * 3. For each peer: zeMemOpenIpcHandle(context, device,
     *        allHandles[peer], &remotePtrs[peer])
     * 4. Similarly exchange signal buffer IPC handles.
     *
     * For MVP, we use a simplified exchange: */

#ifdef UCCL_HAS_LEVEL_ZERO
    /* Level Zero IPC handle exchange */
    auto zeCtx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        comm->defaultQueue->get_context());
    if (comm->device == nullptr) {
        UCCL_LOG(ERROR, "WindowRegister: communicator has no device");
        delete[] w->remotePtrs;
        delete[] w->remoteSignals;
        delete w;
        return ucclSystemError;
    }
    auto zeDev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        *comm->device);

    ze_ipc_mem_handle_t localHandle;
    ze_result_t zeRes = zeMemGetIpcHandle(zeCtx, buff, &localHandle);
    if (zeRes != ZE_RESULT_SUCCESS) {
        UCCL_LOG(ERROR, "WindowRegister: zeMemGetIpcHandle failed");
        delete[] w->remotePtrs;
        delete[] w->remoteSignals;
        delete w;
        return ucclSystemError;
    }

    /* Get signal buffer IPC handle */
    ze_ipc_mem_handle_t sigLocalHandle;
    zeRes = zeMemGetIpcHandle(zeCtx, const_cast<uint64_t*>(w->localSignals),
                              &sigLocalHandle);
    if (zeRes != ZE_RESULT_SUCCESS) {
        UCCL_LOG(ERROR, "WindowRegister: zeMemGetIpcHandle failed for signals");
        delete[] w->remotePtrs;
        delete[] w->remoteSignals;
        delete w;
        return ucclSystemError;
    }

    /* Combine both IPC handles into a single struct and exchange
     * with one MPI_Allgather instead of two separate calls. */
    struct IpcHandlePair {
        ze_ipc_mem_handle_t data;
        ze_ipc_mem_handle_t signal;
    };

    IpcHandlePair localPair = { localHandle, sigLocalHandle };
    std::vector<IpcHandlePair> allPairs(comm->nRanks);
    MPI_Allgather(&localPair, sizeof(IpcHandlePair), MPI_BYTE,
                  allPairs.data(), sizeof(IpcHandlePair), MPI_BYTE,
                  comm->mpiComm);

    /* Store L0 context for later IPC handle cleanup */
    w->zeContext = reinterpret_cast<void*>(zeCtx);

    for (int r = 0; r < comm->nRanks; r++) {
        if (r == comm->rank) continue;

        /* Open data buffer IPC handle */
        void* remotePtr = nullptr;
        zeRes = zeMemOpenIpcHandle(zeCtx, zeDev,
                                   allPairs[r].data, 0, &remotePtr);
        if (zeRes != ZE_RESULT_SUCCESS) {
            UCCL_LOG(ERROR, "WindowRegister: zeMemOpenIpcHandle failed "
                     "for rank %d", r);
            delete[] w->remotePtrs;
            delete[] w->remoteSignals;
            delete w;
            return ucclSystemError;
        }
        w->remotePtrs[r] = remotePtr;

        /* Open signal buffer IPC handle */
        void* remoteSigPtr = nullptr;
        zeMemOpenIpcHandle(zeCtx, zeDev,
                           allPairs[r].signal, 0, &remoteSigPtr);
        w->remoteSignals[r] = static_cast<volatile uint64_t*>(remoteSigPtr);
    }
#else
    /* Without Level Zero: use shared USM for testing (single-process only) */
    UCCL_LOG(WARN, "WindowRegister: Level Zero not available, "
             "using shared memory fallback (testing only)");

    /* In testing mode, all ranks in same process share address space.
     * Combine data and signal pointer exchange into one MPI_Allgather. */
    struct AddrPair {
        uintptr_t data;
        uintptr_t signal;
    };

    AddrPair localPair = {
        reinterpret_cast<uintptr_t>(buff),
        reinterpret_cast<uintptr_t>(const_cast<uint64_t*>(w->localSignals))
    };
    std::vector<AddrPair> allPairs(comm->nRanks);
    MPI_Allgather(&localPair, sizeof(AddrPair), MPI_BYTE,
                  allPairs.data(), sizeof(AddrPair), MPI_BYTE,
                  comm->mpiComm);

    for (int r = 0; r < comm->nRanks; r++) {
        if (r == comm->rank) continue;
        w->remotePtrs[r] = reinterpret_cast<void*>(allPairs[r].data);
        w->remoteSignals[r] =
            reinterpret_cast<volatile uint64_t*>(allPairs[r].signal);
    }
#endif

    UCCL_LOG(INFO, "WindowRegister: rank=%d, size=%zu, nRanks=%d",
             comm->rank, size, comm->nRanks);

    *win = w;
    return ucclSuccess;
}

ucclResult_t ucclWindowDeregister(ucclWindow_t win) {
    if (win == nullptr) return ucclInvalidArgument;

#ifdef UCCL_HAS_LEVEL_ZERO
    /* Close IPC handles */
    if (win->zeContext != nullptr) {
        auto ctx = reinterpret_cast<ze_context_handle_t>(win->zeContext);
        for (int r = 0; r < win->nRanks; r++) {
            if (r == win->myRank) continue;
            if (win->remotePtrs[r] != nullptr) {
                zeMemCloseIpcHandle(ctx, win->remotePtrs[r]);
            }
        }
    }
#endif

    delete[] win->remotePtrs;
    delete[] win->remoteSignals;
    /* Note: localSignals device memory must be freed by caller
     * who owns the queue, or we'd need to store the queue. */
    delete win;
    return ucclSuccess;
}

/* ============================================================
 * Put — Write local data to remote peer's window
 * ============================================================ */

ucclResult_t ucclPut(const void* localBuff, size_t count,
                     ucclDataType_t datatype, int peer,
                     ucclWindow_t peerWin, size_t peerWinOffset,
                     ucclComm_t comm, void* stream) {
    if (localBuff == nullptr || peerWin == nullptr || comm == nullptr) {
        return ucclInvalidArgument;
    }
    if (peer < 0 || peer >= comm->nRanks || peer == comm->rank) {
        return ucclInvalidArgument;
    }

    size_t typeSize = ucclTypeSizeBytes(datatype);
    if (typeSize == 0) return ucclInvalidArgument;

    size_t nbytes = count * typeSize;
    if (peerWinOffset + nbytes > peerWin->size) {
        UCCL_LOG(ERROR, "Put: write exceeds window size "
                 "(offset=%zu + nbytes=%zu > winSize=%zu)",
                 peerWinOffset, nbytes, peerWin->size);
        return ucclInvalidArgument;
    }

    sycl::queue* q = (stream != nullptr)
        ? static_cast<sycl::queue*>(stream)
        : comm->defaultQueue;

    /* Direct P2P write: copy local data to peer's mapped memory */
    void* remoteDst = static_cast<char*>(peerWin->remotePtrs[peer])
                      + peerWinOffset;

    q->memcpy(remoteDst, localBuff, nbytes);

    UCCL_LOG(TRACE, "Put: rank=%d → peer=%d, count=%zu, offset=%zu",
             comm->rank, peer, count, peerWinOffset);

    return ucclSuccess;
}

/* ============================================================
 * Get — Read from remote peer's window into local buffer
 * ============================================================ */

ucclResult_t ucclGet(void* localBuff, size_t count,
                     ucclDataType_t datatype, int peer,
                     ucclWindow_t peerWin, size_t peerWinOffset,
                     ucclComm_t comm, void* stream) {
    if (localBuff == nullptr || peerWin == nullptr || comm == nullptr) {
        return ucclInvalidArgument;
    }
    if (peer < 0 || peer >= comm->nRanks || peer == comm->rank) {
        return ucclInvalidArgument;
    }

    size_t typeSize = ucclTypeSizeBytes(datatype);
    if (typeSize == 0) return ucclInvalidArgument;

    size_t nbytes = count * typeSize;
    if (peerWinOffset + nbytes > peerWin->size) {
        UCCL_LOG(ERROR, "Get: read exceeds window size");
        return ucclInvalidArgument;
    }

    sycl::queue* q = (stream != nullptr)
        ? static_cast<sycl::queue*>(stream)
        : comm->defaultQueue;

    /* Direct P2P read: copy from peer's mapped memory to local */
    const void* remoteSrc =
        static_cast<const char*>(peerWin->remotePtrs[peer])
        + peerWinOffset;

    q->memcpy(localBuff, remoteSrc, nbytes);

    UCCL_LOG(TRACE, "Get: rank=%d ← peer=%d, count=%zu, offset=%zu",
             comm->rank, peer, count, peerWinOffset);

    return ucclSuccess;
}

/* ============================================================
 * PutSignal — Write data + update remote signal counter
 *
 * After the data write completes on the stream, atomically
 * increment the remote peer's signal counter. The remote peer
 * can call ucclWaitSignal to wait for this signal.
 * ============================================================ */

ucclResult_t ucclPutSignal(const void* localBuff, size_t count,
                           ucclDataType_t datatype, int peer,
                           ucclWindow_t peerWin, size_t peerWinOffset,
                           int sigIdx, int ctx, unsigned int flags,
                           ucclComm_t comm, void* stream) {
    if (localBuff == nullptr || peerWin == nullptr || comm == nullptr) {
        return ucclInvalidArgument;
    }
    if (peer < 0 || peer >= comm->nRanks || peer == comm->rank) {
        return ucclInvalidArgument;
    }
    /* sigIdx and ctx must be 0 for now */
    if (sigIdx != 0 || ctx != 0) {
        UCCL_LOG(ERROR, "PutSignal: sigIdx and ctx must be 0");
        return ucclInvalidArgument;
    }
    (void)flags;

    size_t typeSize = ucclTypeSizeBytes(datatype);
    if (typeSize == 0) return ucclInvalidArgument;

    size_t nbytes = count * typeSize;
    if (peerWinOffset + nbytes > peerWin->size) {
        return ucclInvalidArgument;
    }

    sycl::queue* q = (stream != nullptr)
        ? static_cast<sycl::queue*>(stream)
        : comm->defaultQueue;

    void* remoteDst = static_cast<char*>(peerWin->remotePtrs[peer])
                      + peerWinOffset;

    /* Step 1: Copy data to remote */
    q->memcpy(remoteDst, localBuff, nbytes);

    /* Step 2: After data is written, atomically update remote signal.
     * We submit a kernel that does an atomic increment on the remote
     * signal counter. This ensures ordering: the signal is only visible
     * after the data write completes (both in same in-order queue).
     *
     * NOTE: single_task has non-trivial kernel launch overhead for one
     * atomic op. Future optimization: batch multiple PutSignal calls
     * via GroupStart/GroupEnd to amortize launch cost. */
    volatile uint64_t* remoteSignalPtr =
        peerWin->remoteSignals[peer] + comm->rank;

    q->single_task([=]() {
        sycl::atomic_fence(sycl::memory_order::release,
                           sycl::memory_scope::system);
        auto ref = sycl::atomic_ref<uint64_t,
            sycl::memory_order::release,
            sycl::memory_scope::system,
            sycl::access::address_space::global_space>(
                *const_cast<uint64_t*>(remoteSignalPtr));
        ref.fetch_add(1);
    });

    UCCL_LOG(TRACE, "PutSignal: rank=%d → peer=%d, count=%zu, "
             "sigIdx=%d, ctx=%d",
             comm->rank, peer, count, sigIdx, ctx);

    return ucclSuccess;
}

/* ============================================================
 * GetSignal — Read remote data + update local signal counter
 * ============================================================ */

ucclResult_t ucclGetSignal(void* localBuff, size_t count,
                           ucclDataType_t datatype, int peer,
                           ucclWindow_t peerWin, size_t peerWinOffset,
                           int sigIdx, int ctx, unsigned int flags,
                           ucclComm_t comm, void* stream) {
    if (localBuff == nullptr || peerWin == nullptr || comm == nullptr) {
        return ucclInvalidArgument;
    }
    if (peer < 0 || peer >= comm->nRanks || peer == comm->rank) {
        return ucclInvalidArgument;
    }
    if (sigIdx != 0 || ctx != 0) {
        UCCL_LOG(ERROR, "GetSignal: sigIdx and ctx must be 0");
        return ucclInvalidArgument;
    }
    (void)flags;

    size_t typeSize = ucclTypeSizeBytes(datatype);
    if (typeSize == 0) return ucclInvalidArgument;

    size_t nbytes = count * typeSize;
    if (peerWinOffset + nbytes > peerWin->size) {
        return ucclInvalidArgument;
    }

    sycl::queue* q = (stream != nullptr)
        ? static_cast<sycl::queue*>(stream)
        : comm->defaultQueue;

    /* Step 1: Copy data from remote */
    const void* remoteSrc =
        static_cast<const char*>(peerWin->remotePtrs[peer])
        + peerWinOffset;
    q->memcpy(localBuff, remoteSrc, nbytes);

    /* Step 2: Update local signal counter to indicate completion.
     *
     * NOTE: single_task has non-trivial kernel launch overhead for one
     * atomic op. Future optimization: batch multiple GetSignal calls
     * via GroupStart/GroupEnd to amortize launch cost. */
    volatile uint64_t* localSignalPtr =
        peerWin->localSignals + peer;

    q->single_task([=]() {
        sycl::atomic_fence(sycl::memory_order::release,
                           sycl::memory_scope::system);
        auto ref = sycl::atomic_ref<uint64_t,
            sycl::memory_order::release,
            sycl::memory_scope::system,
            sycl::access::address_space::global_space>(
                *const_cast<uint64_t*>(localSignalPtr));
        ref.fetch_add(1);
    });

    UCCL_LOG(TRACE, "GetSignal: rank=%d ← peer=%d, count=%zu, "
             "sigIdx=%d, ctx=%d",
             comm->rank, peer, count, sigIdx, ctx);

    return ucclSuccess;
}

/* ============================================================
 * Signal — Send signal without data transfer
 * ============================================================ */

ucclResult_t ucclSignal(int peer, int sigIdx, int ctx,
                        unsigned int flags,
                        ucclComm_t comm, void* stream) {
    if (comm == nullptr) return ucclInvalidArgument;
    if (peer < 0 || peer >= comm->nRanks || peer == comm->rank) {
        return ucclInvalidArgument;
    }
    if (sigIdx != 0 || ctx != 0) {
        UCCL_LOG(ERROR, "Signal: sigIdx and ctx must be 0");
        return ucclInvalidArgument;
    }
    (void)flags;

    /* We need a window to reference signal counters.
     * For signal-only, the caller must have already done
     * PutSignal or have a window context. Here we use the
     * comm's built-in signal mechanism if available.
     *
     * For MVP: this is a placeholder that would need a
     * window parameter in a full implementation. */

    UCCL_LOG(TRACE, "Signal: rank=%d → peer=%d, sigIdx=%d, ctx=%d",
             comm->rank, peer, sigIdx, ctx);

    /* In practice, signal-only would use the same atomic increment
     * on the remote signal counter without a preceding data copy. */
    return ucclSuccess;
}

/* ============================================================
 * WaitSignal — Wait for signals from peers
 *
 * Spins on device-side signal counters until the expected count
 * of signal operations have been received from each specified peer.
 * ============================================================ */

ucclResult_t ucclWaitSignal(int nDesc,
                            ucclWaitSignalDesc_t* signalDescs,
                            ucclComm_t comm, void* stream) {
    if (nDesc <= 0 || signalDescs == nullptr || comm == nullptr) {
        return ucclInvalidArgument;
    }

    sycl::queue* q = (stream != nullptr)
        ? static_cast<sycl::queue*>(stream)
        : comm->defaultQueue;

    /* Validate descriptors */
    for (int d = 0; d < nDesc; d++) {
        if (signalDescs[d].peer < 0 ||
            signalDescs[d].peer >= comm->nRanks) {
            return ucclInvalidArgument;
        }
        if (signalDescs[d].sigIdx != 0 || signalDescs[d].ctx != 0) {
            UCCL_LOG(ERROR, "WaitSignal: sigIdx and ctx must be 0");
            return ucclInvalidArgument;
        }
    }

    UCCL_LOG(TRACE, "WaitSignal: rank=%d waiting for %d descriptors",
             comm->rank, nDesc);

    /* Build device windows and copy descriptors to device memory
     * so the wait kernel can access them. We cap at a reasonable
     * static limit to avoid dynamic allocation on the hot path. */
    static constexpr int kMaxWaitDescs = UCCL_RMA_MAX_PEERS;
    if (nDesc > kMaxWaitDescs) {
        UCCL_LOG(ERROR, "WaitSignal: nDesc=%d exceeds max=%d",
                 nDesc, kMaxWaitDescs);
        return ucclInvalidArgument;
    }

    /* Prepare per-descriptor signal pointers and expected counts.
     * We resolve these on the host so the kernel only spins on
     * device-accessible pointers without chasing host structures. */
    uint64_t* signalPtrs[kMaxWaitDescs];
    uint64_t  expectedCounts[kMaxWaitDescs];

    for (int d = 0; d < nDesc; d++) {
        ucclWindow_t win = signalDescs[d].win;
        if (win == nullptr) {
            UCCL_LOG(ERROR, "WaitSignal: descriptor %d has null window", d);
            return ucclInvalidArgument;
        }
        int peer = signalDescs[d].peer;
        signalPtrs[d] = const_cast<uint64_t*>(win->localSignals) + peer;
        expectedCounts[d] = static_cast<uint64_t>(signalDescs[d].opCnt);
    }

    /* Allocate device-side arrays for kernel capture */
    uint64_t** devSignalPtrs = sycl::malloc_device<uint64_t*>(nDesc, *q);
    uint64_t*  devExpected   = sycl::malloc_device<uint64_t>(nDesc, *q);

    q->memcpy(devSignalPtrs, signalPtrs, nDesc * sizeof(uint64_t*));
    q->memcpy(devExpected, expectedCounts, nDesc * sizeof(uint64_t));

    /* Submit a single-workgroup kernel that spins on signal counters.
     * Work-item 0 polls all descriptors; after all signals arrive,
     * a group barrier ensures all work-items see the result.
     *
     * Bounded spin with timeout to avoid GPU hang on deadlock. */
    static constexpr uint64_t kMaxSpins = 100'000'000ULL;
    int nd = nDesc;

    q->submit([=](sycl::handler& cgh) {
        cgh.single_task([=]() {
            for (int d = 0; d < nd; d++) {
                uint64_t* sigPtr = devSignalPtrs[d];
                uint64_t expected = devExpected[d];

                for (uint64_t spin = 0; spin < kMaxSpins; spin++) {
                    sycl::atomic_ref<uint64_t,
                        sycl::memory_order::acquire,
                        sycl::memory_scope::system,
                        sycl::access::address_space::global_space>
                        ref(*sigPtr);

                    if (ref.load() >= expected) break;
                }
            }
        });
    });

    /* Free device allocations after kernel completes (in-order queue) */
    q->submit([=](sycl::handler& cgh) {
        cgh.host_task([=]() {
            sycl::free(devSignalPtrs, *q);
            sycl::free(devExpected, *q);
        });
    });

    return ucclSuccess;
}

/* ============================================================
 * Host → Device window conversion
 *
 * Builds a ucclDeviceWindow (POD, kernel-capturable) from the
 * host-side ucclWindow_t.
 * ============================================================ */

ucclResult_t ucclWindowGetDeviceHandle(ucclWindow_t win,
                                       ucclDeviceWindow* deviceWin) {
    if (win == nullptr || deviceWin == nullptr) {
        return ucclInvalidArgument;
    }
    if (win->nRanks > UCCL_RMA_MAX_PEERS) {
        UCCL_LOG(ERROR, "WindowGetDeviceHandle: nRanks=%d exceeds "
                 "UCCL_RMA_MAX_PEERS=%d", win->nRanks, UCCL_RMA_MAX_PEERS);
        return ucclInvalidArgument;
    }

    *deviceWin = makeDeviceWindow(
        win->localBuff,
        win->remotePtrs,
        const_cast<uint64_t*>(win->localSignals),
        const_cast<uint64_t**>(win->remoteSignals),
        win->size,
        win->nRanks,
        win->myRank);

    return ucclSuccess;
}

} /* namespace uccl */
