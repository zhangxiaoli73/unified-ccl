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

    /* Gather all handles */
    std::vector<ze_ipc_mem_handle_t> allHandles(comm->nRanks);
    MPI_Allgather(&localHandle, sizeof(ze_ipc_mem_handle_t), MPI_BYTE,
                  allHandles.data(), sizeof(ze_ipc_mem_handle_t), MPI_BYTE,
                  comm->mpiComm);

    /* Store L0 context for later IPC handle cleanup */
    w->zeContext = reinterpret_cast<void*>(zeCtx);

    for (int r = 0; r < comm->nRanks; r++) {
        if (r == comm->rank) continue;
        void* remotePtr = nullptr;
        zeRes = zeMemOpenIpcHandle(zeCtx, zeDev,
                                   allHandles[r], 0, &remotePtr);
        if (zeRes != ZE_RESULT_SUCCESS) {
            UCCL_LOG(ERROR, "WindowRegister: zeMemOpenIpcHandle failed "
                     "for rank %d", r);
            delete[] w->remotePtrs;
            delete[] w->remoteSignals;
            delete w;
            return ucclSystemError;
        }
        w->remotePtrs[r] = remotePtr;
    }

    /* Similarly exchange signal buffer handles */
    ze_ipc_mem_handle_t sigLocalHandle;
    zeMemGetIpcHandle(zeCtx, const_cast<uint64_t*>(w->localSignals),
                      &sigLocalHandle);

    std::vector<ze_ipc_mem_handle_t> sigHandles(comm->nRanks);
    MPI_Allgather(&sigLocalHandle, sizeof(ze_ipc_mem_handle_t), MPI_BYTE,
                  sigHandles.data(), sizeof(ze_ipc_mem_handle_t), MPI_BYTE,
                  comm->mpiComm);

    for (int r = 0; r < comm->nRanks; r++) {
        if (r == comm->rank) continue;
        void* remotePtr = nullptr;
        zeMemOpenIpcHandle(zeCtx, zeDev, sigHandles[r], 0, &remotePtr);
        w->remoteSignals[r] = static_cast<volatile uint64_t*>(remotePtr);
    }
#else
    /* Without Level Zero: use shared USM for testing (single-process only) */
    UCCL_LOG(WARN, "WindowRegister: Level Zero not available, "
             "using shared memory fallback (testing only)");

    /* In testing mode, all ranks in same process share address space.
     * Exchange raw pointers via MPI. */
    uintptr_t localAddr = reinterpret_cast<uintptr_t>(buff);
    std::vector<uintptr_t> allAddrs(comm->nRanks);
    MPI_Allgather(&localAddr, sizeof(uintptr_t), MPI_BYTE,
                  allAddrs.data(), sizeof(uintptr_t), MPI_BYTE,
                  comm->mpiComm);

    for (int r = 0; r < comm->nRanks; r++) {
        if (r == comm->rank) continue;
        w->remotePtrs[r] = reinterpret_cast<void*>(allAddrs[r]);
    }

    /* Exchange signal pointers */
    uintptr_t sigAddr = reinterpret_cast<uintptr_t>(
        const_cast<uint64_t*>(w->localSignals));
    std::vector<uintptr_t> sigAddrs(comm->nRanks);
    MPI_Allgather(&sigAddr, sizeof(uintptr_t), MPI_BYTE,
                  sigAddrs.data(), sizeof(uintptr_t), MPI_BYTE,
                  comm->mpiComm);

    for (int r = 0; r < comm->nRanks; r++) {
        if (r == comm->rank) continue;
        w->remoteSignals[r] =
            reinterpret_cast<volatile uint64_t*>(sigAddrs[r]);
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
     * after the data write completes (both in same in-order queue). */
    volatile uint64_t* remoteSignalPtr =
        peerWin->remoteSignals[peer] + comm->rank;

    q->single_task([=]() {
        auto ref = sycl::atomic_ref<uint64_t,
            sycl::memory_order::relaxed,
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

    /* Step 2: Update local signal counter to indicate completion */
    volatile uint64_t* localSignalPtr =
        peerWin->localSignals + peer;

    q->single_task([=]() {
        auto ref = sycl::atomic_ref<uint64_t,
            sycl::memory_order::relaxed,
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

    /* Submit a kernel that spins until all expected signals arrive.
     *
     * NOTE: In production, this would use a more efficient wait
     * mechanism (e.g., hardware doorbell, interrupt-based wait).
     * The spin-poll approach is used here for correctness. */

    UCCL_LOG(TRACE, "WaitSignal: rank=%d waiting for %d descriptors",
             comm->rank, nDesc);

    /* TODO: Implement proper signal wait kernel.
     * For now, this is a placeholder — no actual wait is performed.
     * A real implementation would submit a kernel that spins on
     * device-side signal counters. */

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
