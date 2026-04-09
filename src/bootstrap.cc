#include "include/uccl.h"
#include "include/comm.h"
#include "misc/debug.h"

#include <mpi.h>
#include <cstring>

/* Bootstrap via MPI — rank discovery and metadata exchange.
 *
 * Mirrors NCCL src/bootstrap.cc but uses MPI instead of
 * custom TCP socket implementation.
 *
 * MPI provides:
 * - Process management and rank discovery
 * - Collective operations for metadata exchange
 * - Integration with Intel MPI and GPU toolchain */

namespace uccl {

/* Track MPI initialization state */
static bool mpiInitializedByUs = false;

/* Initialize bootstrap: get rank/size from MPI */
ucclResult_t bootstrapInit(ucclComm* comm) {
    if (comm == nullptr) return ucclInvalidArgument;

    /* Check if MPI is already initialized */
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);

    if (!mpiInitialized) {
        int provided;
        MPI_Init_thread(nullptr, nullptr,
                       MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE) {
            UCCL_LOG(WARN, "MPI_THREAD_MULTIPLE not supported "
                     "(got %d)", provided);
        }
        mpiInitializedByUs = true;
    }

    /* Get rank and size from MPI */
    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    /* Validate against caller-provided values if set */
    if (comm->nRanks > 0 && comm->nRanks != mpiSize) {
        UCCL_LOG(WARN, "Caller nRanks (%d) != MPI size (%d), "
                 "using MPI size", comm->nRanks, mpiSize);
    }
    if (comm->rank >= 0 && comm->rank != mpiRank) {
        UCCL_LOG(WARN, "Caller rank (%d) != MPI rank (%d), "
                 "using MPI rank", comm->rank, mpiRank);
    }

    comm->rank = mpiRank;
    comm->nRanks = mpiSize;
    comm->mpiComm = MPI_COMM_WORLD;

    /* Determine local rank (within same node).
     * We use MPI shared memory communicator for this. */
    MPI_Comm sharedComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &sharedComm);

    MPI_Comm_rank(sharedComm, &comm->localRank);
    MPI_Comm_size(sharedComm, &comm->localRanks);
    MPI_Comm_free(&sharedComm);

    UCCL_LOG(INFO, "Bootstrap: rank=%d/%d, localRank=%d/%d",
             comm->rank, comm->nRanks,
             comm->localRank, comm->localRanks);

    return ucclSuccess;
}

/* Broadcast unique ID from root to all ranks */
ucclResult_t bootstrapBcastUniqueId(ucclUniqueId* id, int root) {
    if (id == nullptr) return ucclInvalidArgument;

    MPI_Bcast(id, sizeof(ucclUniqueId), MPI_BYTE,
              root, MPI_COMM_WORLD);

    return ucclSuccess;
}

/* AllGather operation for exchanging metadata */
ucclResult_t bootstrapAllGather(void* sendbuf, void* recvbuf,
                                size_t size) {
    if (sendbuf == nullptr || recvbuf == nullptr) {
        return ucclInvalidArgument;
    }

    if (size > static_cast<size_t>(INT32_MAX)) {
        UCCL_LOG(ERROR, "bootstrapAllGather: size %zu exceeds INT_MAX",
                 size);
        return ucclInvalidArgument;
    }

    MPI_Allgather(sendbuf, static_cast<int>(size), MPI_BYTE,
                  recvbuf, static_cast<int>(size), MPI_BYTE,
                  MPI_COMM_WORLD);

    return ucclSuccess;
}

/* Barrier synchronization */
ucclResult_t bootstrapBarrier() {
    MPI_Barrier(MPI_COMM_WORLD);
    return ucclSuccess;
}

/* Finalize bootstrap (called at comm destroy) */
ucclResult_t bootstrapFinalize() {
    if (mpiInitializedByUs) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
        mpiInitializedByUs = false;
    }
    return ucclSuccess;
}

} /* namespace uccl */
