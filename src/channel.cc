#include "include/comm.h"
#include "include/channel.h"
#include "protocols/protocol.h"
#include "topo/topo.h"
#include "misc/debug.h"

#include <sycl/sycl.hpp>
#include <cstring>
#include <vector>

/* Channel management — mirrors NCCL channel.cc.
 *
 * Channels are independent communication lanes for bandwidth parallelism.
 * Each channel has its own:
 * - Ring ordering (prev/next pointers)
 * - Protocol buffers (FIFO for Simple, lines for LL128)
 * - Transport connections to peers
 */

namespace uccl {

/* Forward declarations for protocol init */
ucclResult_t simpleProtocolInit(sycl::queue& queue, ucclChannel& channel);
ucclResult_t simpleProtocolDestroy(sycl::queue& queue, ucclChannel& channel);
ucclResult_t ll128ProtocolInit(sycl::queue& queue, ucclChannel& channel);
ucclResult_t ll128ProtocolDestroy(sycl::queue& queue, ucclChannel& channel);

/* Initialize all channels for a communicator */
ucclResult_t channelInit(ucclComm* comm) {
    if (comm == nullptr) return ucclInvalidArgument;

    /* Determine channel count from topology tuning */
    ucclTopoTuning tuning;
    int nChannels = 4; /* default */
    if (comm->topo != nullptr) {
        size_t defaultMsgSize = 1 << 20; /* 1 MB for tuning */
        if (ucclTopoTune(comm->topo, defaultMsgSize, &tuning) == ucclSuccess) {
            nChannels = tuning.nChannels;
        }
    }

    /* Clamp to max channels */
    if (nChannels > UCCL_MAX_CHANNELS) nChannels = UCCL_MAX_CHANNELS;
    if (nChannels < 1) nChannels = 1;
    comm->nChannels = nChannels;

    /* Compute ring ordering from topology */
    std::vector<int> ringOrder;
    if (comm->topo != nullptr && comm->topo->nGpus > 0) {
        ringOrder = computeRingOrder(*comm->topo);
    }

    /* Initialize each channel */
    for (int ch = 0; ch < nChannels; ch++) {
        ucclChannel& channel = comm->channels[ch];
        std::memset(&channel, 0, sizeof(ucclChannel));
        channel.id = ch;

        /* Setup ring: prev and next ranks */
        if (comm->nRanks > 1) {
            channel.ring.prev = (comm->rank - 1 + comm->nRanks) %
                                comm->nRanks;
            channel.ring.next = (comm->rank + 1) % comm->nRanks;
        } else {
            channel.ring.prev = 0;
            channel.ring.next = 0;
        }

        /* Fill userRanks with ring ordering.
         * Different channels may use different orderings. */
        if (!ringOrder.empty()) {
            for (size_t i = 0; i < ringOrder.size() &&
                 i < UCCL_MAX_RANKS; i++) {
                channel.ring.userRanks[i] = ringOrder[i];
                if (ringOrder[i] == comm->rank) {
                    channel.ring.index = static_cast<int>(i);
                }
            }
        } else {
            /* Default: sequential ordering */
            for (int i = 0; i < comm->nRanks && i < UCCL_MAX_RANKS; i++) {
                channel.ring.userRanks[i] = i;
            }
            channel.ring.index = comm->rank;
        }

        /* Allocate peer connections */
        channel.nPeers = comm->nRanks;
        channel.peers = new ucclChannelPeer[comm->nRanks];
        std::memset(channel.peers, 0,
                    sizeof(ucclChannelPeer) * comm->nRanks);

        /* Initialize protocol buffers */
        if (comm->defaultQueue != nullptr) {
            ucclResult_t res;

            /* Simple protocol buffer */
            res = simpleProtocolInit(*comm->defaultQueue, channel);
            if (res != ucclSuccess) {
                UCCL_LOG(WARN, "Simple protocol init failed on channel %d",
                         ch);
            }

            /* LL128 protocol buffer */
            res = ll128ProtocolInit(*comm->defaultQueue, channel);
            if (res != ucclSuccess) {
                UCCL_LOG(WARN, "LL128 protocol init failed on channel %d",
                         ch);
            }
        }

        UCCL_LOG(INFO, "Channel %d init: ring prev=%d, next=%d, "
                 "index=%d",
                 ch, channel.ring.prev, channel.ring.next,
                 channel.ring.index);
    }

    /* Set buffer sizes on comm */
    comm->buffSizes[UCCL_PROTO_SIMPLE] = ProtoSimple::DefaultBuffSize;
    comm->buffSizes[UCCL_PROTO_LL128] = ProtoLL128::DefaultBuffSize;

    UCCL_LOG(INFO, "Channels initialized: %d channels for rank %d",
             nChannels, comm->rank);

    return ucclSuccess;
}

/* Destroy all channels and free resources */
ucclResult_t channelDestroy(ucclComm* comm) {
    if (comm == nullptr) return ucclSuccess;

    for (int ch = 0; ch < comm->nChannels; ch++) {
        ucclChannel& channel = comm->channels[ch];

        /* Free protocol buffers */
        if (comm->defaultQueue != nullptr) {
            simpleProtocolDestroy(*comm->defaultQueue, channel);
            ll128ProtocolDestroy(*comm->defaultQueue, channel);
        }

        /* Free peer connections */
        delete[] channel.peers;
        channel.peers = nullptr;
    }

    UCCL_LOG(INFO, "Channels destroyed for rank %d", comm->rank);
    return ucclSuccess;
}

} /* namespace uccl */
