#include "protocol.h"
#include "../include/comm.h"
#include "../include/channel.h"
#include "../device/op128.hpp"
#include "../misc/debug.h"

#include <sycl/sycl.hpp>
#include <cstring>

/* Simple Protocol Implementation
 *
 * High bandwidth protocol using direct data copy with head/tail pointer
 * synchronization between producer and consumer.
 *
 * Mirrors NCCL prims_simple.h:
 * - Sender writes data to FIFO buffer, updates tail
 * - Receiver polls head, reads data when available
 * - Pipelined with SlicePerChunk slices per chunk
 */

namespace uccl {

/* Initialize Simple protocol buffers for a channel.
 * Allocates FIFO buffer using sycl::malloc_host (pinned memory)
 * so both GPU kernel and CPU proxy can access it. */
ucclResult_t simpleProtocolInit(sycl::queue& queue,
                                ucclChannel& channel) {
    size_t buffSize = ProtoSimple::DefaultBuffSize;

    /* Allocate pinned host memory for FIFO buffer */
    void* buff = sycl::malloc_host(buffSize, queue);
    if (buff == nullptr) {
        UCCL_LOG(ERROR, "Failed to allocate Simple protocol buffer "
                 "(%zu bytes) for channel %d", buffSize, channel.id);
        return ucclSystemError;
    }

    std::memset(buff, 0, buffSize);
    channel.buffs[UCCL_PROTO_SIMPLE] = buff;
    channel.buffSizes[UCCL_PROTO_SIMPLE] = buffSize;

    UCCL_LOG(INFO, "Simple protocol init: channel %d, buffer %zu bytes",
             channel.id, buffSize);

    return ucclSuccess;
}

/* Destroy Simple protocol resources */
ucclResult_t simpleProtocolDestroy(sycl::queue& queue,
                                   ucclChannel& channel) {
    if (channel.buffs[UCCL_PROTO_SIMPLE] != nullptr) {
        sycl::free(channel.buffs[UCCL_PROTO_SIMPLE], queue);
        channel.buffs[UCCL_PROTO_SIMPLE] = nullptr;
        channel.buffSizes[UCCL_PROTO_SIMPLE] = 0;
    }
    return ucclSuccess;
}

/* Simple protocol data transfer: sender side.
 *
 * Writes data into the FIFO buffer slot and updates the tail pointer
 * to signal the consumer (either peer GPU or proxy thread).
 *
 * FIFO layout:
 *   buffs[tail % UCCL_STEPS] = data
 *   sizes[tail % UCCL_STEPS] = nbytes
 *   tail++
 *
 * Consumer polls tail and reads when tail advances. */
ucclResult_t simpleProtocolSend(ucclConnFifo* fifo,
                                const void* data,
                                size_t nbytes,
                                sycl::queue& queue) {
    if (fifo == nullptr || data == nullptr) {
        return ucclInvalidArgument;
    }

    /* Wait for buffer slot to be available.
     * Slot is available when head has advanced past it. */
    uint64_t tail = fifo->tail;
    uint64_t slot = tail % UCCL_STEPS;

    /* Wait for consumer to free the slot */
    while (fifo->head + UCCL_STEPS <= tail) {
        /* Spin wait — in production, add timeout/abort check */
    }

    /* Copy data to buffer slot */
    queue.memcpy(fifo->buffs[slot], data, nbytes).wait();
    fifo->sizes[slot] = nbytes;

    /* Memory fence to ensure data is visible before tail update */
    fenceSystem();

    /* Advance tail to signal data availability */
    fifo->tail = tail + 1;

    return ucclSuccess;
}

/* Simple protocol data transfer: receiver side.
 *
 * Polls tail pointer, reads data when available,
 * then advances head to free the slot. */
ucclResult_t simpleProtocolRecv(ucclConnFifo* fifo,
                                void* data,
                                size_t* nbytes,
                                sycl::queue& queue) {
    if (fifo == nullptr || data == nullptr) {
        return ucclInvalidArgument;
    }

    uint64_t head = fifo->head;
    uint64_t slot = head % UCCL_STEPS;

    /* Wait for producer to write data */
    while (fifo->tail <= head) {
        /* Spin wait — in production, add timeout/abort check */
    }

    /* Memory fence to ensure we read the latest data */
    fenceSystem();

    /* Read data from buffer slot */
    size_t size = fifo->sizes[slot];
    queue.memcpy(data, fifo->buffs[slot], size).wait();
    if (nbytes) *nbytes = size;

    /* Advance head to free the slot */
    fifo->head = head + 1;

    return ucclSuccess;
}

} /* namespace uccl */
