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
 * With the new proxy design, Simple protocol buffers serve as the
 * legacy P2P protocol path. The Net path uses bounce buffers + FIFO
 * entries (see primitives.hpp NetPrimitives). */

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

} /* namespace uccl */
