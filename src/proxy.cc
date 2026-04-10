#include "include/comm.h"
#include "include/uccl_net.h"
#include "transport/transport.h"
#include "misc/debug.h"

#include <thread>
#include <atomic>
#include <vector>
#include <cstring>

/* Proxy thread implementation — async network I/O.
 *
 * Architecture (per design spec §7):
 * - GPU kernel writes FIFO entries (ucclFifoEntry) and advances tail
 * - Proxy thread polls tail, initiates isend/irecv via ucclNet plugin
 * - When network op completes, proxy sets entry->done = 1 and advances head
 * - GPU kernel polls done flag (overlapped with reduce computation)
 *
 * Ordering: entries are completed in FIFO order (head-of-line).
 * This matches ring AllReduce step ordering. */

namespace uccl {

/* Process one direction of a channel's FIFO.
 * Returns true if any work was done this iteration. */
static bool processFifo(ucclNet_t* net, ucclConnFifo* fifo,
                         void* netComm, bool isSend) {
    if (fifo == nullptr || netComm == nullptr) return false;

    bool worked = false;
    uint64_t pending = fifo->pendingHead;

    while (pending < fifo->tail) {
        int slot = pending % UCCL_STEPS;
        ucclFifoEntry* entry = &fifo->entries[slot];

        if (entry->request == nullptr) {
            /* New entry: initiate network operation */
            ucclResult_t res = ucclSuccess;
            switch (entry->opType) {
            case UCCL_OP_SEND:
            case UCCL_OP_PUT:
                res = net->isend(netComm, entry->buff, entry->size,
                                 entry->mhandle, &entry->request);
                break;
            case UCCL_OP_RECV:
            case UCCL_OP_WAIT:
                res = net->irecv(netComm, entry->buff, entry->size,
                                 entry->mhandle, &entry->request);
                break;
            case UCCL_OP_SIGNAL:
                /* Zero-byte send for signaling */
                res = net->isend(netComm, nullptr, 0,
                                 nullptr, &entry->request);
                break;
            }
            if (res != ucclSuccess) {
                UCCL_LOG(ERROR, "Proxy: failed to post %s op (slot %d)",
                         isSend ? "send" : "recv", slot);
                break;
            }
            worked = true;
            pending++;
        } else {
            /* Already posted: check completion */
            int done = 0;
            net->test(entry->request, &done, nullptr);
            if (done) {
                entry->done = 1;            /* signal GPU kernel */
                entry->request = nullptr;   /* reset for reuse */
                fifo->pendingHead = ++pending;
                fifo->head = pending;       /* free slot for kernel */
                worked = true;
            } else {
                break; /* in-order completion: stop at first incomplete */
            }
        }
    }
    return worked;
}

/* Proxy progress function: runs on dedicated CPU thread.
 * Polls all channels' send/recv FIFOs and drives UCX progress. */
static void proxyProgressFunc(ucclProxyState* state, ucclComm* comm) {
    UCCL_LOG(INFO, "Proxy thread started for rank %d", comm->rank);

    while (!state->stop) {
        /* Check abort flag */
        if (comm->abortFlag) {
            UCCL_LOG(WARN, "Proxy thread: abort flag set");
            break;
        }

        bool anyWork = false;

        /* Poll each channel's send and recv FIFOs */
        for (int ch = 0; ch < comm->nChannels; ch++) {
            ucclChannel& channel = comm->channels[ch];

            anyWork |= processFifo(state->net, channel.sendFifo,
                                   channel.sendNetComm, /*isSend=*/true);
            anyWork |= processFifo(state->net, channel.recvFifo,
                                   channel.recvNetComm, /*isSend=*/false);
        }

        /* Drive UCX worker progress to advance async operations */
        if (state->net != nullptr && state->net->progress != nullptr) {
            state->net->progress(state->netContext);
        }

        if (!anyWork) std::this_thread::yield();
    }

    UCCL_LOG(INFO, "Proxy thread stopped for rank %d", comm->rank);
}

/* Create and start the proxy thread */
ucclResult_t ucclProxyCreate(ucclComm* comm) {
    if (comm == nullptr) return ucclInvalidArgument;

    /* Only create proxy if inter-node communication is needed */
    if (comm->nNodes <= 1 && comm->net == nullptr) {
        UCCL_LOG(INFO, "Single node: proxy thread not needed");
        comm->proxyState = nullptr;
        return ucclSuccess;
    }

    auto* state = new ucclProxyState();
    state->stop = 0;
    state->abortFlag = &comm->abortFlag;
    state->net = comm->net;
    state->netContext = comm->netContext;

    /* Start proxy thread */
    state->progressThread = std::thread(proxyProgressFunc, state, comm);

    comm->proxyState = state;
    UCCL_LOG(INFO, "Proxy thread created for rank %d", comm->rank);

    return ucclSuccess;
}

/* Stop and destroy the proxy thread */
ucclResult_t ucclProxyDestroy(ucclComm* comm) {
    if (comm == nullptr || comm->proxyState == nullptr) {
        return ucclSuccess;
    }

    auto* state = comm->proxyState;

    /* Signal thread to stop */
    state->stop = 1;

    /* Wait for thread to finish */
    if (state->progressThread.joinable()) {
        state->progressThread.join();
    }

    delete state;
    comm->proxyState = nullptr;

    UCCL_LOG(INFO, "Proxy thread destroyed for rank %d", comm->rank);
    return ucclSuccess;
}

} /* namespace uccl */
