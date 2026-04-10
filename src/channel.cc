#include "include/comm.h"
#include "include/channel.h"
#include "protocols/protocol.h"
#include "topo/topo.h"
#include "transport/transport.h"
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
 * - Send/recv FIFOs and bounce buffers (for Net transport)
 */

namespace uccl {

/* Forward declarations for protocol init */
ucclResult_t simpleProtocolInit(sycl::queue& queue, ucclChannel& channel);
ucclResult_t simpleProtocolDestroy(sycl::queue& queue, ucclChannel& channel);
ucclResult_t ll128ProtocolInit(sycl::queue& queue, ucclChannel& channel);
ucclResult_t ll128ProtocolDestroy(sycl::queue& queue, ucclChannel& channel);

/* Bounce buffer slot size: 512 KB (matches Simple protocol slice) */
static constexpr size_t kBounceSlotSize = 512 * 1024;

/* Allocate a ucclConnFifo in pinned host memory (GPU-accessible). */
static ucclConnFifo* allocFifo(sycl::queue& queue) {
    auto* fifo = static_cast<ucclConnFifo*>(
        sycl::malloc_host(sizeof(ucclConnFifo), queue));
    if (fifo == nullptr) return nullptr;
    std::memset(fifo, 0, sizeof(ucclConnFifo));
    return fifo;
}

/* Allocate bounce buffers: UCCL_STEPS slots of device memory.
 * Optionally register each slot with the network plugin for RDMA. */
static ucclResult_t allocBounceBuffers(sycl::queue& queue,
                                        ucclBounceBuffers* bb,
                                        ucclNet_t* net,
                                        void* netComm,
                                        size_t slotSize) {
    bb->slotSize = slotSize;
    for (int i = 0; i < UCCL_STEPS; i++) {
        bb->buffs[i] = sycl::malloc_device(slotSize, queue);
        if (bb->buffs[i] == nullptr) {
            UCCL_LOG(ERROR, "Failed to alloc bounce buffer slot %d "
                     "(%zu bytes)", i, slotSize);
            return ucclSystemError;
        }
        queue.memset(bb->buffs[i], 0, slotSize).wait();

        /* Register with NIC for RDMA */
        bb->mhandles[i] = nullptr;
        if (net != nullptr && netComm != nullptr) {
            ucclResult_t res = net->regMr(netComm, bb->buffs[i],
                                          slotSize, UCCL_PTR_DEVICE,
                                          &bb->mhandles[i]);
            if (res != ucclSuccess) {
                UCCL_LOG(WARN, "regMr failed for bounce slot %d", i);
            }
        }
    }
    return ucclSuccess;
}

/* Free bounce buffers and deregister from NIC. */
static void freeBounceBuffers(sycl::queue& queue,
                               ucclBounceBuffers* bb,
                               ucclNet_t* net,
                               void* netComm) {
    for (int i = 0; i < UCCL_STEPS; i++) {
        if (bb->mhandles[i] != nullptr && net != nullptr && netComm != nullptr) {
            net->deregMr(netComm, bb->mhandles[i]);
            bb->mhandles[i] = nullptr;
        }
        if (bb->buffs[i] != nullptr) {
            sycl::free(bb->buffs[i], queue);
            bb->buffs[i] = nullptr;
        }
    }
    bb->slotSize = 0;
}

/* Setup net transport connections for a channel (send to next, recv from prev).
 * Also allocates FIFOs and bounce buffers for the channel. */
static ucclResult_t channelNetSetup(ucclComm* comm, ucclChannel& channel) {
    if (comm->net == nullptr || comm->defaultQueue == nullptr) {
        return ucclSuccess;  /* No net plugin — skip */
    }

    sycl::queue& queue = *comm->defaultQueue;

    /* Allocate send and recv FIFOs (pinned host memory) */
    channel.sendFifo = allocFifo(queue);
    channel.recvFifo = allocFifo(queue);
    if (channel.sendFifo == nullptr || channel.recvFifo == nullptr) {
        UCCL_LOG(ERROR, "Failed to alloc FIFOs for channel %d", channel.id);
        return ucclSystemError;
    }

    /* Setup net transport: send to ring.next, recv from ring.prev */
    ucclTransportConn sendConn{}, recvConn{};
    ucclResult_t res;

    res = netTransportSetup(comm, channel.ring.next, &sendConn);
    if (res != ucclSuccess) {
        UCCL_LOG(ERROR, "Net send setup failed for channel %d -> rank %d",
                 channel.id, channel.ring.next);
        return res;
    }
    channel.sendNetComm = sendConn.sendComm;

    res = netTransportSetup(comm, channel.ring.prev, &recvConn);
    if (res != ucclSuccess) {
        UCCL_LOG(ERROR, "Net recv setup failed for channel %d <- rank %d",
                 channel.id, channel.ring.prev);
        return res;
    }
    channel.recvNetComm = recvConn.recvComm;

    /* Allocate bounce buffers (device memory + regMr) */
    res = allocBounceBuffers(queue, &channel.bounceSend, comm->net,
                             channel.sendNetComm, kBounceSlotSize);
    if (res != ucclSuccess) return res;

    res = allocBounceBuffers(queue, &channel.bounceRecv, comm->net,
                             channel.recvNetComm, kBounceSlotSize);
    if (res != ucclSuccess) return res;

    UCCL_LOG(INFO, "Channel %d net setup: send->rank %d, recv<-rank %d, "
             "bounce %zu bytes/slot",
             channel.id, channel.ring.next, channel.ring.prev,
             kBounceSlotSize);

    return ucclSuccess;
}

/* Teardown net resources for a channel. */
static void channelNetDestroy(ucclComm* comm, ucclChannel& channel) {
    if (comm->defaultQueue == nullptr) return;
    sycl::queue& queue = *comm->defaultQueue;

    /* Free bounce buffers */
    freeBounceBuffers(queue, &channel.bounceSend, comm->net,
                      channel.sendNetComm);
    freeBounceBuffers(queue, &channel.bounceRecv, comm->net,
                      channel.recvNetComm);

    /* Close net connections */
    if (comm->net != nullptr) {
        if (channel.sendNetComm != nullptr) {
            comm->net->closeSend(channel.sendNetComm);
            channel.sendNetComm = nullptr;
        }
        if (channel.recvNetComm != nullptr) {
            comm->net->closeRecv(channel.recvNetComm);
            channel.recvNetComm = nullptr;
        }
    }

    /* Free FIFOs */
    if (channel.sendFifo != nullptr) {
        sycl::free(channel.sendFifo, queue);
        channel.sendFifo = nullptr;
    }
    if (channel.recvFifo != nullptr) {
        sycl::free(channel.recvFifo, queue);
        channel.recvFifo = nullptr;
    }
}

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

        /* Setup net transport resources (FIFOs + bounce buffers)
         * only when multi-node communication is needed */
        if (comm->nNodes > 1) {
            ucclResult_t res = channelNetSetup(comm, channel);
            if (res != ucclSuccess) {
                UCCL_LOG(WARN, "Net setup failed on channel %d", ch);
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

        /* Free net transport resources */
        channelNetDestroy(comm, channel);

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
