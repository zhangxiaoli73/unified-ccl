#include "transport/transport.h"
#include "include/comm.h"
#include "misc/debug.h"

/* Transport layer abstraction — mirrors NCCL transport.cc.
 *
 * Provides a unified interface for P2P (intra-node) and
 * Net (inter-node) transports. The transport type is selected
 * based on the topology relationship between communicating peers. */

namespace uccl {

/* Setup transport connection to a peer rank.
 * Selects P2P or Net transport based on node membership. */
ucclResult_t transportSetup(ucclComm* comm, int peerRank,
                            ucclTransportConn* conn) {
    if (comm == nullptr || conn == nullptr) return ucclInvalidArgument;
    if (peerRank < 0 || peerRank >= comm->nRanks) return ucclInvalidArgument;

    /* Check if peer is on same node */
    int myNode = -1;
    int peerNode = -1;

    if (comm->peerInfo != nullptr) {
        myNode = 0;
        uint64_t myHost = comm->peerInfo[comm->rank].hostHash;
        uint64_t peerHost = comm->peerInfo[peerRank].hostHash;

        if (myHost == peerHost) {
            peerNode = 0; /* same node */
        } else {
            peerNode = 1; /* different node */
        }
    }

    ucclTransportType type = selectTransport(myNode, peerNode);

    if (type == UCCL_TRANSPORT_P2P) {
        UCCL_LOG(INFO, "Transport: P2P for rank %d -> %d",
                 comm->rank, peerRank);

        int peerLocalRank = (comm->peerInfo != nullptr)
            ? comm->peerInfo[peerRank].localRank
            : peerRank;

        return p2pTransportSetup(*comm->defaultQueue,
                                 comm->localRank, peerLocalRank, conn);
    } else {
        UCCL_LOG(INFO, "Transport: Net for rank %d -> %d",
                 comm->rank, peerRank);

        if (comm->net == nullptr) {
            UCCL_LOG(ERROR, "Net transport needed but no plugin loaded");
            return ucclSystemError;
        }

        /* In full implementation: exchange addresses via bootstrap,
         * then setup network connection */
        return netTransportSetup(comm->net, comm->netContext,
                                 nullptr /* addr */, conn);
    }
}

/* Close a transport connection */
ucclResult_t transportClose(ucclComm* comm, ucclTransportConn* conn) {
    if (conn == nullptr) return ucclSuccess;

    if (conn->type == UCCL_TRANSPORT_P2P) {
        return p2pTransportClose(conn);
    } else {
        return netTransportClose(comm->net, conn);
    }
}

} /* namespace uccl */
