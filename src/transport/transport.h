#pragma once

#include "../include/uccl_common.h"
#include "../include/uccl_net.h"

#include <sycl/sycl.hpp>

namespace uccl {

/* Transport types */
enum ucclTransportType {
    UCCL_TRANSPORT_P2P = 0,    /* intra-node P2P (PCIe / UPI) */
    UCCL_TRANSPORT_NET = 1,    /* inter-node network (via plugin) */
    UCCL_NUM_TRANSPORTS
};

/* Transport connection handle */
struct ucclTransportConn {
    ucclTransportType type;
    int peerRank;
    void* sendComm;     /* send-side connection */
    void* recvComm;     /* recv-side connection */
    void* mhandle;      /* memory registration handle (for RDMA) */
};

/* ============================================================
 * P2P Transport — Intra-node GPU-to-GPU
 *
 * Two implementation paths:
 * 1. SYCL USM (Unified Shared Memory): sycl::malloc_device + queue.memcpy
 * 2. Level Zero IPC: zeMemGetIpcHandle / zeMemOpenIpcHandle
 * ============================================================ */

ucclResult_t p2pTransportSetup(sycl::queue& queue,
                               int localRank, int peerLocalRank,
                               ucclTransportConn* conn);

ucclResult_t p2pTransportSend(sycl::queue& queue,
                              ucclTransportConn* conn,
                              const void* data, size_t size);

ucclResult_t p2pTransportRecv(sycl::queue& queue,
                              ucclTransportConn* conn,
                              void* data, size_t size);

ucclResult_t p2pTransportClose(ucclTransportConn* conn);

/* ============================================================
 * Net Transport — Inter-node via network plugin
 *
 * Delegates to ucclNet_t plugin interface.
 * Proxy thread calls these asynchronously.
 * ============================================================ */

ucclResult_t netTransportSetup(ucclNet_t* net,
                               void* netHandle,
                               void* connectAddr,
                               ucclTransportConn* conn);

ucclResult_t netTransportSend(ucclNet_t* net,
                              ucclTransportConn* conn,
                              const void* data, size_t size,
                              void** request);

ucclResult_t netTransportRecv(ucclNet_t* net,
                              ucclTransportConn* conn,
                              void* data, size_t size,
                              void** request);

ucclResult_t netTransportTest(ucclNet_t* net,
                              void* request, int* done, int* size);

ucclResult_t netTransportClose(ucclNet_t* net,
                               ucclTransportConn* conn);

/* ============================================================
 * Transport selection based on topology
 * ============================================================ */

ucclTransportType selectTransport(int myNode, int peerNode);

} /* namespace uccl */
