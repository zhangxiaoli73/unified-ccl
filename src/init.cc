#include "include/uccl.h"
#include "include/uccl_common.h"
#include "include/comm.h"
#include "include/channel.h"
#include "topo/topo.h"
#include "plugin/net_plugin.h"
#include "misc/debug.h"

#include <sycl/sycl.hpp>
#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include <random>
#include <set>

/* Communicator initialization and lifecycle management.
 * Mirrors NCCL src/init.cc. */

/* Forward declarations for internal functions */
namespace uccl {
    void debugInit();
    ucclResult_t bootstrapInit(ucclComm* comm);
    ucclResult_t bootstrapBcastUniqueId(ucclUniqueId* id, int root);
    ucclResult_t bootstrapAllGather(void* sendbuf, void* recvbuf, size_t size);
    ucclResult_t channelInit(ucclComm* comm);
    ucclResult_t channelDestroy(ucclComm* comm);
    ucclResult_t ucclProxyCreate(ucclComm* comm);
    ucclResult_t ucclProxyDestroy(ucclComm* comm);
    uint64_t getHostHash();
    uint64_t getPidHash();
}

/* ============================================================
 * Version
 * ============================================================ */

ucclResult_t ucclGetVersion(int* version) {
    if (version == nullptr) return ucclInvalidArgument;
    *version = UCCL_VERSION;
    return ucclSuccess;
}

/* ============================================================
 * Unique ID generation
 * ============================================================ */

ucclResult_t ucclGetUniqueId(ucclUniqueId* uniqueId) {
    if (uniqueId == nullptr) return ucclInvalidArgument;

    /* Generate random bytes for unique ID */
    std::random_device rd;
    std::mt19937_64 gen(rd());

    auto* data = reinterpret_cast<uint64_t*>(uniqueId->internal);
    for (size_t i = 0; i < UCCL_UNIQUE_ID_BYTES / sizeof(uint64_t); i++) {
        data[i] = gen();
    }

    UCCL_LOG(INFO, "Generated unique ID");
    return ucclSuccess;
}

/* ============================================================
 * Communicator Init
 * ============================================================ */

ucclResult_t ucclCommInitRank(ucclComm_t* comm, int nranks,
                              ucclUniqueId commId, int rank) {
    if (comm == nullptr) return ucclInvalidArgument;
    if (rank < 0 || rank >= nranks) return ucclInvalidArgument;
    if (nranks < 1) return ucclInvalidArgument;

    uccl::debugInit();
    UCCL_LOG(INFO, "CommInitRank: nranks=%d, rank=%d", nranks, rank);

    /* Allocate communicator */
    auto* c = new ucclComm();
    std::memset(c, 0, sizeof(ucclComm));

    c->rank = rank;
    c->nRanks = nranks;
    c->asyncError = ucclSuccess;
    c->abortFlag = 0;

    /* Copy unique ID (unused after bootstrap, but stored for reference) */
    (void)commId;

    /* Initialize MPI bootstrap */
    ucclResult_t res = uccl::bootstrapInit(c);
    if (res != ucclSuccess) {
        delete c;
        return res;
    }

    /* Enumerate SYCL GPU devices and select device for this rank */
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        UCCL_LOG(ERROR, "No GPU devices found");
        delete c;
        return ucclSystemError;
    }

    /* Select GPU based on local rank */
    int devIdx = c->localRank % static_cast<int>(devices.size());
    c->device = devices[devIdx];
    c->defaultQueue = new sycl::queue(c->device);
    UCCL_LOG(INFO, "Rank %d using GPU %d: %s", rank, devIdx,
             c->device.get_info<sycl::info::device::name>().c_str());

    /* Detect topology */
    c->topo = new uccl::ucclTopology();
    res = uccl::ucclTopoDetect(c->topo);
    if (res != ucclSuccess) {
        UCCL_LOG(WARN, "Topology detection failed, using defaults");
    }

    /* Initialize channels */
    res = uccl::channelInit(c);
    if (res != ucclSuccess) {
        delete c->topo;
        delete c->defaultQueue;
        delete c;
        return res;
    }

    /* Load network plugin */
    res = uccl::autoLoadNetPlugin(&c->net);
    if (res != ucclSuccess) {
        UCCL_LOG(WARN, "Network plugin load failed, "
                 "inter-node communication unavailable");
    } else {
        /* Initialize plugin */
        res = c->net->init(&c->netContext);
        if (res != ucclSuccess) {
            UCCL_LOG(WARN, "Network plugin init failed");
        }
    }

    /* Create proxy threads for network I/O */
    res = uccl::ucclProxyCreate(c);
    if (res != ucclSuccess) {
        UCCL_LOG(WARN, "Proxy creation failed");
    }

    /* Exchange peer info via MPI AllGather */
    c->peerInfo = new uccl::ucclPeerInfo[nranks];
    uccl::ucclPeerInfo myInfo;
    myInfo.rank = rank;
    myInfo.localRank = c->localRank;
    myInfo.node = 0; /* determined by bootstrap */
    myInfo.hostHash = uccl::getHostHash();
    myInfo.pidHash = uccl::getPidHash();

    uccl::bootstrapAllGather(&myInfo, c->peerInfo,
                             sizeof(uccl::ucclPeerInfo));

    /* Determine node count */
    std::set<uint64_t> uniqueHosts;
    for (int i = 0; i < nranks; i++) {
        uniqueHosts.insert(c->peerInfo[i].hostHash);
    }
    c->nNodes = static_cast<int>(uniqueHosts.size());

    UCCL_LOG(INFO, "CommInitRank complete: rank=%d, nRanks=%d, "
             "nNodes=%d, localRank=%d",
             rank, nranks, c->nNodes, c->localRank);

    *comm = c;
    return ucclSuccess;
}

ucclResult_t ucclCommInitAll(ucclComm_t* comm, int ndev,
                             const int* devlist) {
    if (comm == nullptr || ndev < 1) return ucclInvalidArgument;

    /* Generate unique ID once, shared by all communicators */
    ucclUniqueId id;
    ucclGetUniqueId(&id);

    /* Initialize all communicators for single-process multi-GPU */
    for (int i = 0; i < ndev; i++) {
        ucclResult_t res = ucclCommInitRank(&comm[i], ndev, id, i);
        if (res != ucclSuccess) {
            /* Cleanup already initialized communicators */
            for (int j = 0; j < i; j++) {
                ucclCommDestroy(comm[j]);
            }
            return res;
        }
    }

    (void)devlist; /* device selection handled internally */
    return ucclSuccess;
}

/* ============================================================
 * Communicator Finalize / Destroy / Abort
 * ============================================================ */

ucclResult_t ucclCommFinalize(ucclComm_t comm) {
    if (comm == nullptr) return ucclInvalidArgument;

    UCCL_LOG(INFO, "CommFinalize: rank=%d", comm->rank);

    /* Flush all pending operations */
    if (comm->defaultQueue) {
        comm->defaultQueue->wait();
    }

    return ucclSuccess;
}

ucclResult_t ucclCommDestroy(ucclComm_t comm) {
    if (comm == nullptr) return ucclInvalidArgument;

    UCCL_LOG(INFO, "CommDestroy: rank=%d", comm->rank);

    /* Destroy proxy threads */
    uccl::ucclProxyDestroy(comm);

    /* Finalize network plugin */
    if (comm->net && comm->netContext) {
        comm->net->finalize(comm->netContext);
    }

    /* Destroy channels */
    uccl::channelDestroy(comm);

    /* Free topology */
    delete comm->topo;

    /* Free peer info */
    delete[] comm->peerInfo;

    /* Free SYCL queue */
    delete comm->defaultQueue;

    /* Free communicator */
    delete comm;

    return ucclSuccess;
}

ucclResult_t ucclCommAbort(ucclComm_t comm) {
    if (comm == nullptr) return ucclInvalidArgument;

    UCCL_LOG(WARN, "CommAbort: rank=%d", comm->rank);
    comm->abortFlag = 1;

    /* Force destroy */
    return ucclCommDestroy(comm);
}

/* ============================================================
 * Communicator Queries
 * ============================================================ */

ucclResult_t ucclCommCount(const ucclComm_t comm, int* count) {
    if (comm == nullptr || count == nullptr) return ucclInvalidArgument;
    *count = comm->nRanks;
    return ucclSuccess;
}

ucclResult_t ucclCommUserRank(const ucclComm_t comm, int* rank) {
    if (comm == nullptr || rank == nullptr) return ucclInvalidArgument;
    *rank = comm->rank;
    return ucclSuccess;
}

/* ============================================================
 * Error Reporting
 * ============================================================ */

const char* ucclGetErrorString(ucclResult_t result) {
    switch (result) {
        case ucclSuccess:         return "no error";
        case ucclSystemError:     return "system error";
        case ucclInternalError:   return "internal error";
        case ucclInvalidArgument: return "invalid argument";
        case ucclInvalidUsage:    return "invalid usage";
        case ucclRemoteError:     return "remote error";
        case ucclInProgress:      return "in progress";
        default:                  return "unknown error";
    }
}

const char* ucclGetLastError(ucclComm_t comm) {
    if (comm == nullptr) return "communicator is NULL";
    return comm->lastError;
}
