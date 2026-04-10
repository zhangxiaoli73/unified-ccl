#include "transport.h"
#include "../include/comm.h"
#include "../misc/debug.h"

#include <mpi.h>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>

/* Net Transport — Inter-node Communication via Network Plugin
 *
 * Establishes bidirectional connections using:
 * 1. Listen on this rank → share address via MPI_Sendrecv → peer connects
 * 2. Peer listens → receives address via MPI_Sendrecv → this rank connects
 *
 * Supports GPU Direct RDMA: plugin can register GPU memory
 * via regMr to enable zero-copy network transfers. */

namespace uccl {

/* Bootstrap address exchange buffer size */
static constexpr size_t kAddrBufSize = sizeof(struct sockaddr_storage);

/* Setup network transport connection with bidirectional link.
 *
 * Protocol:
 * - Each side listens, exchanges addresses via MPI_Sendrecv,
 *   then connects to the peer's listener and accepts the peer's
 *   incoming connection.
 * - Lower rank sends first to avoid deadlock. */
ucclResult_t netTransportSetup(ucclComm* comm, int peerRank,
                               ucclTransportConn* conn) {
    ucclNet_t* net = comm->net;
    if (net == nullptr || conn == nullptr) {
        return ucclInvalidArgument;
    }

    conn->type = UCCL_TRANSPORT_NET;
    conn->peerRank = peerRank;
    int myRank = comm->rank;

    /* Step 1: Both sides start a listener */
    struct sockaddr_storage myAddr;
    std::memset(&myAddr, 0, sizeof(myAddr));

    /* Use INADDR_ANY with a system-assigned port */
    auto* sin = reinterpret_cast<struct sockaddr_in*>(&myAddr);
    sin->sin_family = AF_INET;
    sin->sin_addr.s_addr = INADDR_ANY;
    sin->sin_port = 0;

    void* listenComm = nullptr;
    ucclResult_t res = net->listen(comm->netContext, &myAddr, &listenComm);
    if (res != ucclSuccess) {
        UCCL_LOG(ERROR, "Net transport: listen failed for rank %d", myRank);
        return res;
    }

    /* Step 2: Exchange listen addresses via MPI */
    struct sockaddr_storage peerAddr;
    std::memset(&peerAddr, 0, sizeof(peerAddr));

    MPI_Sendrecv(&myAddr, kAddrBufSize, MPI_BYTE, peerRank, 0 /* tag */,
                 &peerAddr, kAddrBufSize, MPI_BYTE, peerRank, 0 /* tag */,
                 comm->mpiComm, MPI_STATUS_IGNORE);

    UCCL_LOG(INFO, "Net transport: rank %d exchanged addresses with rank %d",
             myRank, peerRank);

    /* Step 3: Connect to peer's listener (creates sendComm) */
    res = net->connect(comm->netContext, &peerAddr, &conn->sendComm);
    if (res != ucclSuccess) {
        UCCL_LOG(ERROR, "Net transport: connect to rank %d failed", peerRank);
        net->closeListen(listenComm);
        return res;
    }

    /* Step 4: Accept peer's connection (creates recvComm) */
    /* Retry accept with worker progress until the peer's connect arrives */
    int retries = 0;
    static constexpr int kMaxRetries = 10000;
    res = ucclInProgress;
    while (res == ucclInProgress && retries < kMaxRetries) {
        net->progress(comm->netContext);
        res = net->accept(listenComm, &conn->recvComm);
        retries++;
    }
    if (res != ucclSuccess) {
        UCCL_LOG(ERROR, "Net transport: accept from rank %d failed "
                 "after %d retries", peerRank, retries);
        net->closeSend(conn->sendComm);
        conn->sendComm = nullptr;
        net->closeListen(listenComm);
        return res == ucclInProgress ? ucclSystemError : res;
    }

    /* Clean up listener — no longer needed */
    net->closeListen(listenComm);

    /* Step 5: Set deterministic tags for message disambiguation.
     * sendComm tag = (myRank << 32) | peerRank  — matches peer's recvComm
     * recvComm tag = (peerRank << 32) | myRank  — matches peer's sendComm */
    uint64_t sendTag = (static_cast<uint64_t>(myRank) << 32) |
                        static_cast<uint64_t>(peerRank);
    uint64_t recvTag = (static_cast<uint64_t>(peerRank) << 32) |
                        static_cast<uint64_t>(myRank);
    net->setTag(conn->sendComm, sendTag);
    net->setTag(conn->recvComm, recvTag);

    UCCL_LOG(INFO, "Net transport setup: rank %d <-> rank %d connected via %s",
             myRank, peerRank, net->name);
    return ucclSuccess;
}

/* Async send via network plugin */
ucclResult_t netTransportSend(ucclNet_t* net,
                              ucclTransportConn* conn,
                              const void* data, size_t size,
                              void** request) {
    if (net == nullptr || conn == nullptr || data == nullptr) {
        return ucclInvalidArgument;
    }

    return net->isend(conn->sendComm,
                      const_cast<void*>(data), size,
                      conn->mhandle, request);
}

/* Async receive via network plugin */
ucclResult_t netTransportRecv(ucclNet_t* net,
                              ucclTransportConn* conn,
                              void* data, size_t size,
                              void** request) {
    if (net == nullptr || conn == nullptr || data == nullptr) {
        return ucclInvalidArgument;
    }

    return net->irecv(conn->recvComm, data, size,
                      conn->mhandle, request);
}

/* Test completion of async operation */
ucclResult_t netTransportTest(ucclNet_t* net,
                              void* request, int* done, int* size) {
    if (net == nullptr || request == nullptr) {
        return ucclInvalidArgument;
    }

    return net->test(request, done, size);
}

/* Close network transport connection */
ucclResult_t netTransportClose(ucclNet_t* net,
                               ucclTransportConn* conn) {
    if (net == nullptr || conn == nullptr) {
        return ucclSuccess;
    }

    ucclResult_t res = ucclSuccess;

    if (conn->sendComm != nullptr) {
        res = net->closeSend(conn->sendComm);
        conn->sendComm = nullptr;
    }

    if (conn->recvComm != nullptr) {
        ucclResult_t r2 = net->closeRecv(conn->recvComm);
        if (res == ucclSuccess) res = r2;
        conn->recvComm = nullptr;
    }

    return res;
}

} /* namespace uccl */
