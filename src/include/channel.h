#pragma once

#include "uccl_common.h"
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
};
