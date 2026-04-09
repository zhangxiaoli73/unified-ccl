#pragma once

#include "uccl_common.h"
#include "uccl_net.h"
#include "channel.h"
#include "hw_resources.h"
#include "../protocols/protocol.h"
#include "../topo/topo.h"

#include <sycl/sycl.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <mpi.h>

namespace uccl {

/* Forward declarations */
struct ucclProxyState;
struct SymmetricMemoryContext;

/* Connection FIFO for GPU kernel <-> proxy thread synchronization */
struct ucclConnFifo {
    volatile uint64_t head;      /* proxy updates (received / readable) */
    volatile uint64_t tail;      /* GPU kernel updates (written / sendable) */
    void* buffs[UCCL_STEPS];     /* ring buffer slots */
    size_t sizes[UCCL_STEPS];    /* data size per slot */
};

/* Connection info for a peer */
struct ucclConnInfo {
    void* buff;                  /* Direct buffer pointer (P2P) */
    void* llBuff;                /* LL128 buffer pointer */
    ucclConnFifo* connFifo;      /* FIFO for proxy-based transport */
    size_t buffSize;
    int direct;                  /* 1 if direct P2P, 0 if via proxy/net */
};

/* Peer info exchanged during init */
struct ucclPeerInfo {
    int rank;
    int localRank;
    int node;
    uint64_t hostHash;
    uint64_t pidHash;
};

/* Proxy operation descriptor */
struct ucclProxyOp {
    int channelId;
    int peer;                    /* remote rank */
    size_t nbytes;
    size_t chunkSize;
    int nsteps;                  /* transfer steps */
    int protocol;                /* SIMPLE / LL128 */
    ucclConnInfo* connection;    /* transport connection */
};

/* Proxy thread state */
struct ucclProxyState {
    std::thread progressThread;  /* proxy progress thread */
    volatile int stop;           /* stop flag */
    volatile int* abortFlag;     /* abort flag */

    /* Network plugin reference */
    ucclNet_t* net;
    void* netContext;
};

} /* namespace uccl */

/* Main communicator structure — mirrors ncclComm */
struct ucclComm {
    int rank;
    int nRanks;
    int nNodes;
    int localRank;
    int localRanks;

    /* SYCL device & queue */
    sycl::device device;
    sycl::queue* defaultQueue;

    /* Topology */
    uccl::ucclTopology* topo;

    /* Channels — data parallel lanes */
    int nChannels;
    ucclChannel channels[UCCL_MAX_CHANNELS];

    /* Transport connections */
    uccl::ucclPeerInfo* peerInfo;

    /* Network plugin */
    ucclNet_t* net;
    void* netContext;

    /* Proxy state */
    uccl::ucclProxyState* proxyState;

    /* Symmetric memory context (intra-node P2P, set when UCCL_ALGO=symmetric) */
    uccl::SymmetricMemoryContext* symmetricCtx;

    /* Hardware resources (EU count, copy engines, exec mode) */
    uccl::ucclHwResources hwResources;

    /* Bootstrap (MPI) */
    MPI_Comm mpiComm;

    /* Buffer sizes per protocol */
    size_t buffSizes[UCCL_NUM_PROTOCOLS];

    /* Error state */
    ucclResult_t asyncError;
    volatile int abortFlag;

    /* Last error message */
    char lastError[256];
};
