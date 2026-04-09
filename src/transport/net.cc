#include "transport.h"
#include "../misc/debug.h"

/* Net Transport — Inter-node Communication via Network Plugin
 *
 * Delegates all operations to the ucclNet_t plugin interface.
 * The proxy thread calls these functions asynchronously to
 * overlap network I/O with GPU computation.
 *
 * Supports GPU Direct RDMA: plugin can register GPU memory
 * via regMr to enable zero-copy network transfers. */

namespace uccl {

/* Setup network transport connection */
ucclResult_t netTransportSetup(ucclNet_t* net,
                               void* netHandle,
                               void* connectAddr,
                               ucclTransportConn* conn) {
    if (net == nullptr || conn == nullptr) {
        return ucclInvalidArgument;
    }

    conn->type = UCCL_TRANSPORT_NET;

    /* Establish send connection */
    ucclResult_t res = net->connect(netHandle, connectAddr, &conn->sendComm);
    if (res != ucclSuccess) {
        UCCL_LOG(ERROR, "Net transport: connect failed for send");
        return res;
    }

    UCCL_LOG(INFO, "Net transport setup: connected via %s", net->name);
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
