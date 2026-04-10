#pragma once

#include "uccl_common.h"
#include "fifo.h"
#include "../protocols/protocol.h"

namespace uccl {
struct ucclConnInfo;
}

/* Channel: independent communication lane for bandwidth parallelism.
 * Each channel has its own ring ordering, buffers, and transport connections.
 * Multiple channels run simultaneously, each processing a subset of data. */

/* Peer connection within a channel */
struct ucclChannelPeer {
    uccl::ucclConnInfo* send;    /* send-side connection */
    uccl::ucclConnInfo* recv;    /* recv-side connection */
};

/* Bounce buffer set for a single channel direction (send or recv).
 * UCCL_STEPS slots, each sized to hold one slice of data.
 * Allocated as device memory + registered with NIC via regMr. */
struct ucclBounceBuffers {
    void* buffs[UCCL_STEPS];        /* device memory slots */
    void* mhandles[UCCL_STEPS];     /* NIC regMr handles per slot */
    size_t slotSize;                /* bytes per slot */
};

struct ucclChannel {
    int id;

    /* Ring topology (each channel can have different ring ordering) */
    struct ucclRing {
        int prev;                           /* previous rank in ring */
        int next;                           /* next rank in ring */
        int userRanks[UCCL_MAX_RANKS];      /* rank ordering */
        int index;                          /* this rank's index in userRanks */
    } ring;

    /* Per-peer connection info */
    ucclChannelPeer* peers;
    int nPeers;

    /* Protocol buffers */
    void* buffs[UCCL_NUM_PROTOCOLS];        /* FIFO buffer per protocol */
    size_t buffSizes[UCCL_NUM_PROTOCOLS];

    /* ---- Proxy-based Net transport resources ---- */

    /* Send/recv FIFOs for proxy thread (pinned host memory).
     * Only allocated when transport is Net (nNodes > 1). */
    uccl::ucclConnFifo* sendFifo;
    uccl::ucclConnFifo* recvFifo;

    /* Bounce buffers for collective Net kernel (device memory).
     * Kernel writes to bounce send, proxy DMA's out via UCX.
     * Proxy DMA's into bounce recv, kernel reads and reduces. */
    ucclBounceBuffers bounceSend;
    ucclBounceBuffers bounceRecv;

    /* Net transport connections for this channel's ring peers */
    void* sendNetComm;      /* UCX send comm to ring.next */
    void* recvNetComm;      /* UCX recv comm from ring.prev */
    void* sendMhandle;      /* regMr handle for bounce send */
    void* recvMhandle;      /* regMr handle for bounce recv */
};
