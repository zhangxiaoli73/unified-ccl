#pragma once

#include "../include/uccl_common.h"
#include "../include/uccl.h"
#include "rma_device.hpp"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>

/* One-Sided RMA (Remote Memory Access) Device Operations.
 *
 * Modeled after NCCL's one-sided point-to-point operations:
 *   - put       : Write local data to a remote peer's registered window
 *   - get       : Read remote peer's data into a local buffer
 *   - putSignal : Write data + update remote signal (ordering guarantee)
 *   - getSignal : Read remote data + update local signal (ordering guarantee)
 *   - signal    : Send signal without data transfer
 *   - waitSignal: Wait until expected signals have arrived
 *
 * These operations require a registered symmetric memory window
 * (ucclWindow_t) where all participating ranks have pre-registered
 * device memory via Level Zero IPC handle exchange.
 *
 * Usage pattern (similar to NCCL ncclPutSignal/ncclWaitSignal):
 *   Rank 0:  ucclPut(...)          // write data to Rank 1's window
 *            ucclPutSignal(...)     // write data + signal to Rank 1
 *   Rank 1:  ucclWaitSignal(...)   // wait for signal from Rank 0
 *            // data is now visible in local window
 */

namespace uccl {

/* ============================================================
 * Memory Window — symmetric registered memory region
 * ============================================================ */

/* Opaque window handle */
struct ucclWindow {
    void* localBuff;            /* Local device allocation */
    void** remotePtrs;          /* remotePtrs[rank] = IPC-mapped ptr to peer */
    size_t size;                /* Window size in bytes */
    int nRanks;                 /* Number of participating ranks */
    int myRank;                 /* This rank */

    /* Signal state for RMA ordering */
    volatile uint64_t* localSignals;   /* Local signal counters (one per peer) */
    volatile uint64_t** remoteSignals; /* Remote signal counter pointers */

    /* Level Zero context handle for IPC cleanup (stored as void*) */
    void* zeContext;
};

/* ucclWindow_t is typedef'd in uccl.h; do not redefine here */

/* ============================================================
 * Window Registration
 * ============================================================ */

/* Register a memory window for RMA operations.
 * All ranks must call this collectively with their local allocation.
 * Under the hood: zeMemGetIpcHandle + MPI_Allgather + zeMemOpenIpcHandle.
 *
 * @param win       Output window handle
 * @param buff      Local device memory (sycl::malloc_device)
 * @param size      Size of local buffer in bytes
 * @param comm      Communicator
 * @return ucclSuccess on success */
ucclResult_t ucclWindowRegister(ucclWindow_t* win, void* buff,
                                size_t size, ucclComm_t comm);

/* Deregister a memory window and release IPC mappings.
 *
 * @param win       Window to deregister
 * @return ucclSuccess on success */
ucclResult_t ucclWindowDeregister(ucclWindow_t win);

/* ============================================================
 * One-Sided Data Transfer Operations
 * ============================================================ */

/* Put: Write local data to a remote peer's registered window.
 *
 * Copies `count` elements from `localBuff` to `peerWin` at
 * `peerWinOffset` on rank `peer`. No signal is sent.
 *
 * @param localBuff     Source data (local device memory)
 * @param count         Number of elements to transfer
 * @param datatype      Element data type
 * @param peer          Target rank
 * @param peerWin       Target rank's registered window
 * @param peerWinOffset Byte offset within target window
 * @param comm          Communicator
 * @param stream        SYCL queue (void* for C compat) */
ucclResult_t ucclPut(const void* localBuff, size_t count,
                     ucclDataType_t datatype, int peer,
                     ucclWindow_t peerWin, size_t peerWinOffset,
                     ucclComm_t comm, void* stream);

/* Get: Read remote peer's data into a local buffer.
 *
 * Copies `count` elements from `peerWin` at `peerWinOffset`
 * on rank `peer` to `localBuff`.
 *
 * @param localBuff     Destination buffer (local device memory)
 * @param count         Number of elements to transfer
 * @param datatype      Element data type
 * @param peer          Source rank
 * @param peerWin       Source rank's registered window
 * @param peerWinOffset Byte offset within source window
 * @param comm          Communicator
 * @param stream        SYCL queue (void* for C compat) */
ucclResult_t ucclGet(void* localBuff, size_t count,
                     ucclDataType_t datatype, int peer,
                     ucclWindow_t peerWin, size_t peerWinOffset,
                     ucclComm_t comm, void* stream);

/* PutSignal: Write data + update remote signal.
 *
 * Same as ucclPut, but additionally increments a signal counter
 * on the remote peer after the data write completes. The signal
 * guarantees all preceding data from this put is visible.
 *
 * @param localBuff     Source data
 * @param count         Number of elements
 * @param datatype      Element data type
 * @param peer          Target rank
 * @param peerWin       Target rank's registered window
 * @param peerWinOffset Byte offset within target window
 * @param sigIdx        Signal index (must be 0 for now)
 * @param ctx           Context identifier (must be 0 for now)
 * @param flags         Reserved (must be 0)
 * @param comm          Communicator
 * @param stream        SYCL queue */
ucclResult_t ucclPutSignal(const void* localBuff, size_t count,
                           ucclDataType_t datatype, int peer,
                           ucclWindow_t peerWin, size_t peerWinOffset,
                           int sigIdx, int ctx, unsigned int flags,
                           ucclComm_t comm, void* stream);

/* GetSignal: Read remote data + update local signal.
 *
 * Same as ucclGet, but additionally increments a local signal
 * counter after the data read completes. Useful for producer
 * notification when get-based data flow is used.
 *
 * @param localBuff     Destination buffer
 * @param count         Number of elements
 * @param datatype      Element data type
 * @param peer          Source rank
 * @param peerWin       Source rank's registered window
 * @param peerWinOffset Byte offset within source window
 * @param sigIdx        Signal index (must be 0 for now)
 * @param ctx           Context identifier (must be 0 for now)
 * @param flags         Reserved (must be 0)
 * @param comm          Communicator
 * @param stream        SYCL queue */
ucclResult_t ucclGetSignal(void* localBuff, size_t count,
                           ucclDataType_t datatype, int peer,
                           ucclWindow_t peerWin, size_t peerWinOffset,
                           int sigIdx, int ctx, unsigned int flags,
                           ucclComm_t comm, void* stream);

/* ============================================================
 * Signal-Only Operations
 * ============================================================ */

/* Signal: Send a signal without data transfer.
 *
 * Increments the signal counter on rank `peer`. When a signal
 * is updated on the remote peer, all prior ucclPutSignal and
 * ucclSignal operations to the same peer/context have completed.
 *
 * @param peer   Target rank
 * @param sigIdx Signal index (must be 0 for now)
 * @param ctx    Context identifier (must be 0 for now)
 * @param flags  Reserved (must be 0)
 * @param comm   Communicator
 * @param stream SYCL queue */
ucclResult_t ucclSignal(int peer, int sigIdx, int ctx,
                        unsigned int flags,
                        ucclComm_t comm, void* stream);

/* Descriptor for waitSignal: how many signals to wait for from a peer */
struct ucclWaitSignalDesc {
    int opCnt;      /* Number of signal operations to wait for */
    int peer;       /* Peer rank */
    int sigIdx;     /* Signal index (must be 0 for now) */
    int ctx;        /* Context identifier (must be 0 for now) */
};

typedef struct ucclWaitSignalDesc ucclWaitSignalDesc_t;

/* WaitSignal: Wait for signals from peers.
 *
 * Blocks the GPU stream until all specified signal operations
 * have been received. Data from corresponding putSignal operations
 * is guaranteed visible after waitSignal completes.
 *
 * @param nDesc      Number of descriptors
 * @param signalDescs Array of wait descriptors
 * @param comm       Communicator
 * @param stream     SYCL queue */
ucclResult_t ucclWaitSignal(int nDesc,
                            ucclWaitSignalDesc_t* signalDescs,
                            ucclComm_t comm, void* stream);

/* ============================================================
 * Host → Device window conversion
 *
 * Convert a host-side ucclWindow_t into a ucclDeviceWindow that
 * can be captured by value in a SYCL kernel lambda.
 * ============================================================ */

ucclResult_t ucclWindowGetDeviceHandle(ucclWindow_t win,
                                       ucclDeviceWindow* deviceWin);

} /* namespace uccl */
