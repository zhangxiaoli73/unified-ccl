#pragma once

#include "reduce_kernel.hpp"
#include "op128.hpp"
#include "common_kernel.hpp"
#include "../protocols/protocol.h"
#include "../include/uccl_common.h"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>

/* Communication Primitives — mirrors NCCL primitives.h
 *
 * All protocols implement the same Primitives interface.
 * Algorithm layer uses these to orchestrate data flow
 * without knowing if the underlying protocol is Simple or LL128.
 *
 * Key operations:
 *   send()                 — send from input buffer
 *   sendFromOutput()       — send from output buffer
 *   recv()                 — receive to output buffer
 *   recvReduceSend()       — recv + reduce + send (reduce-scatter core)
 *   recvReduceCopySend()   — recv + reduce + copy to output + send
 *   recvCopySend()         — recv + copy to output + send (allgather core)
 *   copySend()             — copy input to output + send
 */

namespace uccl {

/* ============================================================
 * Primitives template — parameterized by data type, reduction op,
 * and protocol type.
 * ============================================================ */
template <typename T, typename RedOp, typename Proto>
class Primitives {
public:
    Primitives(const T* inputBuff, T* outputBuff,
               sycl::nd_item<1> item)
        : inputBuff_(inputBuff),
          outputBuff_(outputBuff),
          item_(item),
          sg_(item.get_sub_group()),
          redOp_() {}

    /* Send data from input buffer at offset inpIx */
    void send(intptr_t inpIx, int eltN) {
        const T* src = inputBuff_ + inpIx;
        sendData(src, eltN);
    }

    /* Send data from output buffer at offset outIx */
    void sendFromOutput(intptr_t outIx, int eltN) {
        const T* src = outputBuff_ + outIx;
        sendData(src, eltN);
    }

    /* Receive data to output buffer at offset outIx */
    void recv(intptr_t outIx, int eltN, bool postOp = false) {
        T* dst = outputBuff_ + outIx;
        recvData(dst, eltN);
        (void)postOp;
    }

    /* Receive + reduce with input + send.
     * Core primitive for ring reduce-scatter. */
    void recvReduceSend(intptr_t inpIx, int eltN) {
        int lid = item_.get_local_id(0);
        int groupSize = item_.get_local_range(0);

        /* Each work-item processes a portion of elements */
        for (int i = lid; i < eltN; i += groupSize) {
            T recvVal = recvElement(i);
            T localVal = outputBuff_[inpIx + i]; /* from previous step */
            T reduced = redOp_(localVal, recvVal);
            outputBuff_[inpIx + i] = reduced;
            sendElement(i, reduced);
        }
        subGroupBarrier(sg_);
    }

    /* Receive + reduce with input + copy to output + send.
     * First step of reduce-scatter. */
    void recvReduceCopySend(intptr_t inpIx, intptr_t outIx,
                            int eltN, bool postOp = false) {
        int lid = item_.get_local_id(0);
        int groupSize = item_.get_local_range(0);
        (void)postOp;

        for (int i = lid; i < eltN; i += groupSize) {
            T recvVal = recvElement(i);
            T localVal = inputBuff_[inpIx + i];
            T reduced = redOp_(localVal, recvVal);
            outputBuff_[outIx + i] = reduced;
            sendElement(i, reduced);
        }
        subGroupBarrier(sg_);
    }

    /* Receive + copy to output + send.
     * Core primitive for ring allgather. */
    void recvCopySend(intptr_t outIx, int eltN, bool postOp = false) {
        int lid = item_.get_local_id(0);
        int groupSize = item_.get_local_range(0);
        (void)postOp;

        for (int i = lid; i < eltN; i += groupSize) {
            T val = recvElement(i);
            outputBuff_[outIx + i] = val;
            sendElement(i, val);
        }
        subGroupBarrier(sg_);
    }

    /* Copy from input to output + send */
    void copySend(intptr_t inpIx, intptr_t outIx,
                  int eltN, bool postOp = false) {
        int lid = item_.get_local_id(0);
        int groupSize = item_.get_local_range(0);
        (void)postOp;

        for (int i = lid; i < eltN; i += groupSize) {
            T val = inputBuff_[inpIx + i];
            outputBuff_[outIx + i] = val;
            sendElement(i, val);
        }
        subGroupBarrier(sg_);
    }

private:
    const T* inputBuff_;
    T* outputBuff_;
    sycl::nd_item<1> item_;
    sycl::sub_group sg_;
    RedOp redOp_;

    /* Protocol-specific send implementation.
     * Simple: direct copy to send buffer + tail update.
     * LL128: pack into 128B line with flags. */
    void sendData(const T* src, int eltN) {
        if constexpr (Proto::Id == UCCL_PROTO_SIMPLE) {
            sendDataSimple(src, eltN);
        } else {
            sendDataLL128(src, eltN);
        }
    }

    /* Simple protocol send: direct memcpy-style */
    void sendDataSimple(const T* src, int eltN) {
        int lid = item_.get_local_id(0);
        int groupSize = item_.get_local_range(0);

        /* Cooperative copy across work-items */
        for (int i = lid; i < eltN; i += groupSize) {
            /* In full implementation: write to connFifo buffer,
             * update tail pointer for proxy/peer to consume */
            (void)src[i]; /* data would be written to transport buffer */
        }
        /* Ensure all writes are visible */
        fenceDevice();
    }

    /* LL128 protocol send: pack data into 128-byte lines with flags */
    void sendDataLL128(const T* src, int eltN) {
        int lid = item_.get_local_id(0);
        int groupSize = item_.get_local_range(0);

        /* LL128: 14 data elements + 2 flag elements per 128B line.
         * Each line = 16 x uint64_t.
         * Flag threads (lane 7 and 15) write flag values. */
        int laneId = getLaneId(sg_);
        bool isFlagThread = (laneId == 7 || laneId == 15);

        size_t elemsPerLine = ProtoLL128::DataElems;
        int nLines = (eltN * sizeof(T) + ProtoLL128::LineSize - 1) /
                     ProtoLL128::LineSize;

        for (int line = lid / ProtoLL128::LineElems;
             line < nLines; line += groupSize / ProtoLL128::LineElems) {
            int elemInLine = lid % ProtoLL128::LineElems;

            if (isFlagThread) {
                /* Write flag to signal data availability */
                /* In full implementation: write flag word to line */
            } else if (static_cast<size_t>(elemInLine) < elemsPerLine) {
                /* Write data element */
                /* In full implementation: pack data into line */
            }
        }
        subGroupBarrier(sg_);
    }

    /* Receive a single element (protocol-aware) */
    T recvElement(int idx) {
        /* In full implementation: read from recv buffer,
         * with protocol-specific synchronization.
         * Simple: poll head pointer.
         * LL128: poll flag word in 128B line. */
        (void)idx;
        return T(0); /* placeholder */
    }

    /* Send a single element (protocol-aware) */
    void sendElement(int idx, T val) {
        /* In full implementation: write to send buffer */
        (void)idx;
        (void)val;
    }
};

} /* namespace uccl */
