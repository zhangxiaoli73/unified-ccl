#include "protocol.h"
#include "../include/comm.h"
#include "../include/channel.h"
#include "../device/op128.hpp"
#include "../misc/debug.h"

#include <sycl/sycl.hpp>
#include <cstring>
#include <cstdint>

/* LL128 Protocol Implementation
 *
 * Low-latency protocol using 128-byte lines with inline flags.
 * Mirrors NCCL prims_ll128.h.
 *
 * Core concept:
 * - Each transfer unit is a 128-byte "line" (16 x uint64_t)
 * - 14 uint64_t carry data, 2 uint64_t carry flags
 * - Receiver detects data arrival by checking flag values
 * - Avoids separate synchronization overhead
 *
 * 128-byte line layout:
 *   u64[0..6]  = data (7 words)
 *   u64[7]     = flag
 *   u64[8..13] = data (6 words)
 *   u64[14]    = data (1 word)
 *   u64[15]    = flag
 *   Total: 14 data + 2 flag = 16 x 8B = 128B
 *
 * Flag convention:
 *   Flag value encodes a sequence number. When the flag matches
 *   the expected sequence, the data in that line is valid.
 */

namespace uccl {

/* LL128 line structure */
struct alignas(128) LL128Line {
    uint64_t data[ProtoLL128::LineElems]; /* 16 x uint64_t = 128 bytes */
};

/* Flag indices within a line */
static constexpr int LL128_FLAG_IDX_0 = 7;   /* first flag at position 7 */
static constexpr int LL128_FLAG_IDX_1 = 15;  /* second flag at position 15 */

/* Data indices: all positions except flags */
static constexpr int LL128_DATA_INDICES[] = {
    0, 1, 2, 3, 4, 5, 6, /* positions 0-6 */
    8, 9, 10, 11, 12, 13, 14 /* positions 8-14 */
};

/* Initialize LL128 protocol buffers for a channel */
ucclResult_t ll128ProtocolInit(sycl::queue& queue,
                               ucclChannel& channel) {
    size_t buffSize = ProtoLL128::DefaultBuffSize;

    /* Allocate pinned host memory for LL128 buffer */
    void* buff = sycl::malloc_host(buffSize, queue);
    if (buff == nullptr) {
        UCCL_LOG(ERROR, "Failed to allocate LL128 protocol buffer "
                 "(%zu bytes) for channel %d", buffSize, channel.id);
        return ucclSystemError;
    }

    /* Zero-initialize: flag=0 means no valid data */
    std::memset(buff, 0, buffSize);
    channel.buffs[UCCL_PROTO_LL128] = buff;
    channel.buffSizes[UCCL_PROTO_LL128] = buffSize;

    UCCL_LOG(INFO, "LL128 protocol init: channel %d, buffer %zu bytes",
             channel.id, buffSize);

    return ucclSuccess;
}

/* Destroy LL128 protocol resources */
ucclResult_t ll128ProtocolDestroy(sycl::queue& queue,
                                  ucclChannel& channel) {
    if (channel.buffs[UCCL_PROTO_LL128] != nullptr) {
        sycl::free(channel.buffs[UCCL_PROTO_LL128], queue);
        channel.buffs[UCCL_PROTO_LL128] = nullptr;
        channel.buffSizes[UCCL_PROTO_LL128] = 0;
    }
    return ucclSuccess;
}

/* Write data into LL128 line with flags.
 *
 * This is the sender-side operation:
 * 1. Pack data into the 14 data positions of the 128B line
 * 2. Write flag value encoding the sequence number
 * 3. The receiver polls the flag to detect arrival */
void ll128PackLine(LL128Line* line,
                   const uint64_t* data,
                   int nDataWords,
                   uint64_t flagVal) {
    /* Write data words to data positions */
    for (int i = 0; i < nDataWords && i < ProtoLL128::DataElems; i++) {
        line->data[LL128_DATA_INDICES[i]] = data[i];
    }

    /* Pad remaining data positions with zero */
    for (int i = nDataWords; i < ProtoLL128::DataElems; i++) {
        line->data[LL128_DATA_INDICES[i]] = 0;
    }

    /* Ensure data is written before flags (release semantics) */
    fenceSystem();

    /* Write flags last — signals data availability */
    storeRelease(&line->data[LL128_FLAG_IDX_0], flagVal);
    storeRelease(&line->data[LL128_FLAG_IDX_1], flagVal);
}

/* Read data from LL128 line, polling flags until valid.
 *
 * This is the receiver-side operation:
 * 1. Poll flag positions until they match expected sequence
 * 2. Read data words from the 14 data positions
 * 3. Return when all data in the line is valid */
bool ll128UnpackLine(const LL128Line* line,
                     uint64_t* data,
                     int nDataWords,
                     uint64_t expectedFlag,
                     uint64_t maxSpins) {
    /* Poll flags until both match expected value */
    for (uint64_t spin = 0; spin < maxSpins; spin++) {
        uint64_t flag0 = loadAcquire(
            const_cast<uint64_t*>(&line->data[LL128_FLAG_IDX_0]));
        uint64_t flag1 = loadAcquire(
            const_cast<uint64_t*>(&line->data[LL128_FLAG_IDX_1]));

        if (flag0 == expectedFlag && flag1 == expectedFlag) {
            /* Flags match: data is valid */
            fenceSystem();

            /* Read data words */
            for (int i = 0; i < nDataWords && i < ProtoLL128::DataElems; i++) {
                data[i] = line->data[LL128_DATA_INDICES[i]];
            }
            return true;
        }
    }

    return false; /* timeout */
}

/* LL128 send: pack user data into LL128 lines and write to buffer.
 *
 * Splits input data into 128B lines, each containing 14 data words
 * (112 bytes of user data) plus 2 flag words. */
ucclResult_t ll128ProtocolSend(void* ll128Buff,
                               size_t buffSize,
                               const void* userData,
                               size_t nbytes,
                               uint64_t seqNum) {
    if (ll128Buff == nullptr || userData == nullptr) {
        return ucclInvalidArgument;
    }

    auto* lines = static_cast<LL128Line*>(ll128Buff);
    size_t maxLines = buffSize / sizeof(LL128Line);
    const auto* src = static_cast<const uint64_t*>(userData);

    /* Number of uint64_t data words */
    size_t totalWords = (nbytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);

    /* Pack data into lines */
    size_t wordOffset = 0;
    for (size_t lineIdx = 0;
         wordOffset < totalWords && lineIdx < maxLines;
         lineIdx++) {
        int wordsInLine = static_cast<int>(
            (totalWords - wordOffset > ProtoLL128::DataElems)
                ? ProtoLL128::DataElems
                : totalWords - wordOffset);

        ll128PackLine(&lines[lineIdx], src + wordOffset,
                      wordsInLine, seqNum);
        wordOffset += wordsInLine;
    }

    return ucclSuccess;
}

/* LL128 receive: poll lines and unpack user data.
 *
 * Polls each line's flags until they match the expected sequence number,
 * then extracts the data words. */
ucclResult_t ll128ProtocolRecv(const void* ll128Buff,
                               size_t buffSize,
                               void* userData,
                               size_t nbytes,
                               uint64_t expectedSeq) {
    if (ll128Buff == nullptr || userData == nullptr) {
        return ucclInvalidArgument;
    }

    const auto* lines = static_cast<const LL128Line*>(ll128Buff);
    size_t maxLines = buffSize / sizeof(LL128Line);
    auto* dst = static_cast<uint64_t*>(userData);

    size_t totalWords = (nbytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    size_t wordOffset = 0;
    static constexpr uint64_t MAX_SPINS = 10000000;

    for (size_t lineIdx = 0;
         wordOffset < totalWords && lineIdx < maxLines;
         lineIdx++) {
        int wordsInLine = static_cast<int>(
            (totalWords - wordOffset > ProtoLL128::DataElems)
                ? ProtoLL128::DataElems
                : totalWords - wordOffset);

        bool ok = ll128UnpackLine(&lines[lineIdx], dst + wordOffset,
                                   wordsInLine, expectedSeq, MAX_SPINS);
        if (!ok) {
            UCCL_LOG(ERROR, "LL128 recv timeout at line %zu", lineIdx);
            return ucclRemoteError;
        }
        wordOffset += wordsInLine;
    }

    return ucclSuccess;
}

} /* namespace uccl */
