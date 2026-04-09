#include "transport.h"
#include "../protocols/protocol.h"
#include "../misc/debug.h"

#include <sycl/sycl.hpp>
#include <cstring>

/* Try to include Level Zero for IPC */
#if __has_include(<level_zero/ze_api.h>)
#include <level_zero/ze_api.h>
#define UCCL_HAS_LEVEL_ZERO 1
#else
#define UCCL_HAS_LEVEL_ZERO 0
#endif

/* P2P Transport — Intra-node GPU-to-GPU Communication
 *
 * Provides direct GPU memory access for GPUs within the same node.
 *
 * Implementation paths:
 * 1. SYCL USM: Uses sycl::malloc_device() + queue.memcpy() for
 *    direct P2P transfers. Works when GPUs share the same SYCL context.
 *
 * 2. Level Zero IPC: Uses zeMemGetIpcHandle() / zeMemOpenIpcHandle()
 *    for cross-process GPU memory mapping. Required for multi-process
 *    intra-node communication (e.g., one process per GPU). */

namespace uccl {

/* P2P connection data */
struct P2PConnData {
    void* localBuff;           /* local device buffer */
    void* remoteBuff;          /* mapped remote device buffer */
    size_t buffSize;
    bool useIpc;               /* true if using L0 IPC, false if USM */
    sycl::context ctx;         /* SYCL context for sycl::free */
#if UCCL_HAS_LEVEL_ZERO
    ze_ipc_mem_handle_t ipcHandle;
#endif
};

/* Setup P2P transport between two local GPUs.
 *
 * For same-process GPUs:
 *   Allocate device memory and let SYCL runtime handle P2P.
 * For cross-process GPUs:
 *   Use Level Zero IPC to share memory handles. */
ucclResult_t p2pTransportSetup(sycl::queue& queue,
                               int localRank, int peerLocalRank,
                               ucclTransportConn* conn) {
    if (conn == nullptr) {
        return ucclInvalidArgument;
    }

    conn->type = UCCL_TRANSPORT_P2P;
    conn->peerRank = peerLocalRank;

    /* Allocate P2P connection data */
    auto* p2pData = new P2PConnData();
    p2pData->buffSize = ProtoSimple::DefaultBuffSize;
    p2pData->useIpc = false;

    /* Allocate local device buffer for P2P */
    p2pData->localBuff = sycl::malloc_device(p2pData->buffSize, queue);
    if (p2pData->localBuff == nullptr) {
        UCCL_LOG(ERROR, "P2P: failed to allocate device buffer "
                 "for rank %d -> %d", localRank, peerLocalRank);
        delete p2pData;
        return ucclSystemError;
    }

    /* Store context for later cleanup */
    p2pData->ctx = queue.get_context();

    /* Zero initialize */
    queue.memset(p2pData->localBuff, 0, p2pData->buffSize).wait();

#if UCCL_HAS_LEVEL_ZERO
    /* Get IPC handle for cross-process sharing */
    try {
        auto ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            queue.get_context());
        ze_result_t res = zeMemGetIpcHandle(ctx, p2pData->localBuff,
                                             &p2pData->ipcHandle);
        if (res == ZE_RESULT_SUCCESS) {
            p2pData->useIpc = true;
            UCCL_LOG(INFO, "P2P: IPC handle obtained for rank %d -> %d",
                     localRank, peerLocalRank);
        }
    } catch (...) {
        UCCL_LOG(WARN, "P2P: Level Zero IPC not available, "
                 "using USM fallback");
    }
#endif

    conn->sendComm = p2pData;
    conn->recvComm = p2pData;

    UCCL_LOG(INFO, "P2P transport setup: rank %d -> %d (%s)",
             localRank, peerLocalRank,
             p2pData->useIpc ? "IPC" : "USM");

    return ucclSuccess;
}

/* P2P send: copy data to remote GPU */
ucclResult_t p2pTransportSend(sycl::queue& queue,
                              ucclTransportConn* conn,
                              const void* data, size_t size) {
    if (conn == nullptr || data == nullptr) {
        return ucclInvalidArgument;
    }

    auto* p2pData = static_cast<P2PConnData*>(conn->sendComm);
    if (p2pData == nullptr || p2pData->localBuff == nullptr) {
        return ucclInternalError;
    }

    /* Clamp size to buffer capacity */
    if (size > p2pData->buffSize) {
        size = p2pData->buffSize;
    }

    /* Direct device-to-device copy via SYCL USM */
    queue.memcpy(p2pData->localBuff, data, size).wait();

    return ucclSuccess;
}

/* P2P receive: copy data from remote GPU */
ucclResult_t p2pTransportRecv(sycl::queue& queue,
                              ucclTransportConn* conn,
                              void* data, size_t size) {
    if (conn == nullptr || data == nullptr) {
        return ucclInvalidArgument;
    }

    auto* p2pData = static_cast<P2PConnData*>(conn->recvComm);
    if (p2pData == nullptr || p2pData->localBuff == nullptr) {
        return ucclInternalError;
    }

    if (size > p2pData->buffSize) {
        size = p2pData->buffSize;
    }

    /* Device-to-device copy */
    queue.memcpy(data, p2pData->localBuff, size).wait();

    return ucclSuccess;
}

/* Close P2P transport connection */
ucclResult_t p2pTransportClose(ucclTransportConn* conn) {
    if (conn == nullptr) return ucclSuccess;

    auto* p2pData = static_cast<P2PConnData*>(conn->sendComm);
    if (p2pData != nullptr) {
        if (p2pData->localBuff != nullptr) {
            sycl::free(p2pData->localBuff, p2pData->ctx);
            p2pData->localBuff = nullptr;
        }
        delete p2pData;
        conn->sendComm = nullptr;
        conn->recvComm = nullptr;
    }

    return ucclSuccess;
}

/* Select transport type based on node location */
ucclTransportType selectTransport(int myNode, int peerNode) {
    if (myNode == peerNode) {
        return UCCL_TRANSPORT_P2P;
    }
    return UCCL_TRANSPORT_NET;
}

} /* namespace uccl */
