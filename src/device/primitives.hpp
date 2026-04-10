#pragma once

#include "reduce_kernel.hpp"
#include "op128.hpp"
#include "common_kernel.hpp"
#include "../protocols/protocol.h"
#include "../include/uccl_common.h"
#include "../include/fifo.h"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>

/* Communication Primitives for GPU kernel <-> proxy data path.
 *
 * Two separate Primitives classes:
 *
 * 1. Primitives<T, RedOp, Proto> — P2P kernel (intra-node)
 *    Uses direct IPC/USM pointer load/store. No FIFO, no proxy.
 *
 * 2. NetPrimitives<T, RedOp> — Net kernel (inter-node, via proxy)
 *    Uses bounce buffers and ucclConnFifo to communicate with proxy.
 *    Persistent kernel: 1 workgroup loops internally.
 */

namespace uccl {

/* ============================================================
 * P2P Primitives — intra-node, direct GPU-to-GPU
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

    void sendData(const T* src, int eltN) {
        if constexpr (Proto::Id == UCCL_PROTO_SIMPLE) {
            sendDataSimple(src, eltN);
        } else {
            sendDataLL128(src, eltN);
        }
    }

    void sendDataSimple(const T* src, int eltN) {
        int lid = item_.get_local_id(0);
        int groupSize = item_.get_local_range(0);
        for (int i = lid; i < eltN; i += groupSize) {
            (void)src[i]; /* P2P: data written via IPC pointer */
        }
        fenceDevice();
    }

    void sendDataLL128(const T* src, int eltN) {
        int lid = item_.get_local_id(0);
        int groupSize = item_.get_local_range(0);
        int laneId = getLaneId(sg_);
        bool isFlagThread = (laneId == 7 || laneId == 15);
        size_t elemsPerLine = ProtoLL128::DataElems;
        int nLines = (eltN * sizeof(T) + ProtoLL128::LineSize - 1) /
                     ProtoLL128::LineSize;
        for (int line = lid / ProtoLL128::LineElems;
             line < nLines; line += groupSize / ProtoLL128::LineElems) {
            int elemInLine = lid % ProtoLL128::LineElems;
            if (isFlagThread) {
                /* write flag */
            } else if (static_cast<size_t>(elemInLine) < elemsPerLine) {
                /* write data element */
            }
        }
        subGroupBarrier(sg_);
    }

    T recvElement(int idx) {
        (void)idx;
        return T(0); /* P2P: read via IPC pointer — placeholder */
    }

    void sendElement(int idx, T val) {
        (void)idx;
        (void)val; /* P2P: write via IPC pointer — placeholder */
    }
};

/* ============================================================
 * NetPrimitives — inter-node, via FIFO + bounce buffer + proxy
 *
 * Designed for the persistent Net kernel (1 workgroup).
 * The kernel writes data to bounce send buffers, posts FIFO entries,
 * and polls done flags for completion.
 * ============================================================ */

/* Parameters passed to the Net kernel, captured by value. */
struct NetKernelArgs {
    volatile uint64_t* sendFifoHead;    /* &sendFifo->head */
    volatile uint64_t* sendFifoTail;    /* &sendFifo->tail */
    volatile uint64_t* recvFifoHead;    /* &recvFifo->head */
    volatile uint64_t* recvFifoTail;    /* &recvFifo->tail */

    /* Flat arrays of UCCL_STEPS pointers (bounce buffer slots) */
    void* bounceSendBuffs[UCCL_STEPS];
    void* bounceRecvBuffs[UCCL_STEPS];
    void* bounceSendMhandles[UCCL_STEPS];
    void* bounceRecvMhandles[UCCL_STEPS];

    /* Pointers to FIFO entry arrays (pinned host memory) */
    ucclFifoEntry* sendEntries;   /* sendFifo->entries */
    ucclFifoEntry* recvEntries;   /* recvFifo->entries */
};

template <typename T, typename RedOp>
class NetPrimitives {
public:
    NetPrimitives(const T* inputBuff, T* outputBuff,
                  const NetKernelArgs& args,
                  sycl::nd_item<1> item)
        : inputBuff_(inputBuff),
          outputBuff_(outputBuff),
          args_(args),
          item_(item),
          sg_(item.get_sub_group()),
          redOp_(),
          sendTailLocal_(0),
          recvTailLocal_(0) {}

    /* ---- Collective send: copy data to bounce send slot, post FIFO entry ---- */
    void sendViaBounce(const T* src, size_t nbytes, int step) {
        int slot = step % UCCL_STEPS;
        int lid = item_.get_local_id(0);
        int groupSize = item_.get_local_range(0);

        /* Wait for slot to be free (proxy has completed the previous use) */
        if (lid == 0) {
            while (*args_.sendFifoHead + UCCL_STEPS <= sendTailLocal_) {
                /* spin — overlapped with previous reduce */
            }
        }
        sycl::group_barrier(item_.get_group());

        /* Cooperative copy: src -> bounce send slot */
        T* bounceDst = static_cast<T*>(args_.bounceSendBuffs[slot]);
        size_t nElts = nbytes / sizeof(T);
        for (size_t i = lid; i < nElts; i += groupSize) {
            bounceDst[i] = src[i];
        }
        sycl::group_barrier(item_.get_group());

        /* System-scope fence: ensure device stores visible to host */
        sycl::atomic_fence(sycl::memory_order::release,
                           sycl::memory_scope::system);

        /* Post FIFO entry (leader work-item only) */
        if (lid == 0) {
            ucclFifoEntry* entry = &args_.sendEntries[slot];
            entry->opType = UCCL_OP_SEND;
            entry->buff = args_.bounceSendBuffs[slot];
            entry->size = nbytes;
            entry->mhandle = args_.bounceSendMhandles[slot];
            entry->done = 0;
            entry->request = nullptr;

            /* Advance tail to signal proxy */
            sycl::atomic_fence(sycl::memory_order::release,
                               sycl::memory_scope::system);
            sendTailLocal_++;
            *args_.sendFifoTail = sendTailLocal_;
        }
        sycl::group_barrier(item_.get_group());
    }

    /* ---- Collective recv: post FIFO recv entry, poll done ---- */
    void postRecv(size_t nbytes, int step) {
        int slot = step % UCCL_STEPS;
        int lid = item_.get_local_id(0);

        /* Wait for slot to be free */
        if (lid == 0) {
            while (*args_.recvFifoHead + UCCL_STEPS <= recvTailLocal_) {
                /* spin */
            }
        }
        sycl::group_barrier(item_.get_group());

        /* Post FIFO entry (leader only) */
        if (lid == 0) {
            ucclFifoEntry* entry = &args_.recvEntries[slot];
            entry->opType = UCCL_OP_RECV;
            entry->buff = args_.bounceRecvBuffs[slot];
            entry->size = nbytes;
            entry->mhandle = args_.bounceRecvMhandles[slot];
            entry->done = 0;
            entry->request = nullptr;

            sycl::atomic_fence(sycl::memory_order::release,
                               sycl::memory_scope::system);
            recvTailLocal_++;
            *args_.recvFifoTail = recvTailLocal_;
        }
        sycl::group_barrier(item_.get_group());
    }

    /* ---- Wait for recv to complete, then read bounce recv data ---- */
    void waitRecv(int step) {
        int slot = step % UCCL_STEPS;
        int lid = item_.get_local_id(0);

        /* Poll done flag (set by proxy after irecv completes) */
        if (lid == 0) {
            while (args_.recvEntries[slot].done == 0) {
                /* spin — overlapped with reduce on previous step */
            }
        }
        sycl::group_barrier(item_.get_group());

        /* System-scope acquire: ensure we read data written by NIC DMA */
        sycl::atomic_fence(sycl::memory_order::acquire,
                           sycl::memory_scope::system);
    }

    /* ---- Reduce: recv bounce data + local data -> output ---- */
    void reduceFromBounceRecv(T* output, const T* local,
                              size_t nElts, int step) {
        int slot = step % UCCL_STEPS;
        int lid = item_.get_local_id(0);
        int groupSize = item_.get_local_range(0);

        const T* recvData = static_cast<const T*>(args_.bounceRecvBuffs[slot]);
        for (size_t i = lid; i < nElts; i += groupSize) {
            output[i] = redOp_(local[i], recvData[i]);
        }
        sycl::group_barrier(item_.get_group());
    }

    /* ---- Copy: bounce recv data -> output ---- */
    void copyFromBounceRecv(T* output, size_t nElts, int step) {
        int slot = step % UCCL_STEPS;
        int lid = item_.get_local_id(0);
        int groupSize = item_.get_local_range(0);

        const T* recvData = static_cast<const T*>(args_.bounceRecvBuffs[slot]);
        for (size_t i = lid; i < nElts; i += groupSize) {
            output[i] = recvData[i];
        }
        sycl::group_barrier(item_.get_group());
    }

    /* Helper to get bounce send buffer as typed pointer */
    T* bounceSend(int step) {
        return static_cast<T*>(args_.bounceSendBuffs[step % UCCL_STEPS]);
    }

    T* bounceRecv(int step) {
        return static_cast<T*>(args_.bounceRecvBuffs[step % UCCL_STEPS]);
    }

private:
    const T* inputBuff_;
    T* outputBuff_;
    NetKernelArgs args_;
    sycl::nd_item<1> item_;
    sycl::sub_group sg_;
    RedOp redOp_;
    uint64_t sendTailLocal_;
    uint64_t recvTailLocal_;
};

} /* namespace uccl */
