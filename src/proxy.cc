#include "include/comm.h"
#include "include/uccl_net.h"
#include "transport/transport.h"
#include "misc/debug.h"

#include <thread>
#include <atomic>
#include <vector>
#include <cstring>

/* Proxy thread implementation — async network I/O.
 * Mirrors NCCL src/proxy.cc.
 *
 * Architecture:
 * - GPU kernel handles intra-node P2P communication
 * - CPU proxy thread handles inter-node network I/O
 * - Synchronization via host-visible FIFO buffer + head/tail pointers
 *
 * The proxy thread polls FIFO buffers and calls ucclNet plugin
 * isend/irecv/test to perform async network transfers.
 *
 * Timeline:
 *   GPU kernel: [reduce chunk0] [reduce chunk1] ...
 *                     |               |
 *                write FIFO      write FIFO
 *                     |               |
 *                     v               v
 *   Proxy:     [send chunk0]   [send chunk1]   ...
 *              [recv chunk0']  [recv chunk1']  ...
 *                     |               |
 *                write FIFO      write FIFO
 *                     v               v
 *   GPU kernel: [reduce chunk0'] [reduce chunk1'] ...
 */

namespace uccl {

/* Proxy progress function: runs on dedicated CPU thread.
 *
 * Continuously polls FIFO buffers for each channel.
 * When new data is available (tail advanced by GPU kernel),
 * initiates network send/recv via the ucclNet plugin. */
static void proxyProgressFunc(ucclProxyState* state, ucclComm* comm) {
    UCCL_LOG(INFO, "Proxy thread started for rank %d", comm->rank);

    while (!state->stop) {
        /* Check abort flag */
        if (comm->abortFlag) {
            UCCL_LOG(WARN, "Proxy thread: abort flag set");
            break;
        }

        /* Poll each channel's FIFO for pending operations */
        for (int ch = 0; ch < comm->nChannels; ch++) {
            ucclChannel& channel = comm->channels[ch];

            /* Check send FIFO: if GPU kernel wrote new data (tail advanced),
             * initiate network send */
            for (int p = 0; p < UCCL_NUM_PROTOCOLS; p++) {
                void* buff = channel.buffs[p];
                if (buff == nullptr) continue;

                /* In full implementation:
                 * 1. Check connFifo->tail for new data
                 * 2. If new data, call net->isend()
                 * 3. Poll existing requests with net->test()
                 * 4. When send completes, update connFifo->head
                 *
                 * Similarly for recv:
                 * 1. Call net->irecv() to post receive
                 * 2. Poll with net->test()
                 * 3. When recv completes, update connFifo->tail
                 *    so GPU kernel can read the data */
            }
        }

        /* Drive UCX worker progress to advance async operations */
        if (state->net != nullptr && state->net->progress != nullptr) {
            state->net->progress(state->netContext);
        }

        /* Yield to avoid busy-waiting when idle.
         * In production, use more sophisticated polling:
         * - Batch multiple operations
         * - Adaptive polling frequency
         * - Event-based wakeup */
        std::this_thread::yield();
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
