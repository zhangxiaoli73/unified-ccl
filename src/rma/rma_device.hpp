#pragma once

#include "../include/uccl_common.h"
#include "../device/op128.hpp"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>

/* GPU Device-Side RMA (Remote Memory Access) API.
 *
 * These are inline device functions callable from within SYCL kernels.
 * Unlike the host-side RMA API (rma.h / rma.cc) which enqueues work
 * to a SYCL queue, these operate directly on the GPU — work-items
 * perform P2P loads/stores through IPC-mapped pointers.
 *
 * Modeled after NCCL Device API:
 *   https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device.html
 *
 * Prerequisites:
 *   - A ucclDeviceWindow must be set up on the host and passed to the kernel.
 *     It contains pre-resolved IPC-mapped remote pointers and signal counters.
 *   - All remote pointers in the window must be accessible from the current
 *     GPU (via Level Zero IPC or shared USM).
 *
 * Usage pattern (inside a SYCL kernel):
 *
 *   // Single work-item or cooperative sub-group put + signal:
 *   uccl::devicePut(win, localData, count, peer, offset, item);
 *   uccl::devicePutSignal(win, localData, count, peer, offset, item);
 *
 *   // On receiver side:
 *   uccl::deviceWaitSignal(win, peer, expectedCount, item);
 *   // data is now visible in win.localBuff
 */

namespace uccl {

/* ============================================================
 * Device-side window descriptor — passed into SYCL kernels.
 *
 * This is a POD struct (no pointers-to-pointers) suitable for
 * capture by value in SYCL lambdas. Max 8 peers for static sizing;
 * for dynamic sizing use the pointer-based variant below.
 * ============================================================ */

static constexpr int UCCL_RMA_MAX_PEERS = 8;

/* Static-sized device window (suitable for kernel capture by value) */
struct ucclDeviceWindow {
    void* localBuff;                            /* This GPU's buffer */
    void* remotePtrs[UCCL_RMA_MAX_PEERS];       /* IPC-mapped ptrs to peers */
    uint64_t* localSignals;                     /* Local signal counters [nRanks] */
    uint64_t* remoteSignals[UCCL_RMA_MAX_PEERS]; /* Remote signal counters */
    size_t size;                                /* Window size in bytes */
    int nRanks;
    int myRank;
};

/* Dynamic-sized device window (pointers allocated in device/shared USM) */
struct ucclDeviceWindowDyn {
    void* localBuff;
    void** remotePtrs;          /* Device-accessible array of remote ptrs */
    uint64_t* localSignals;     /* Device-accessible signal counters */
    uint64_t** remoteSignals;   /* Device-accessible array of remote signal ptrs */
    size_t size;
    int nRanks;
    int myRank;
};

/* ============================================================
 * Device Put — single work-item writes data to remote peer
 *
 * Each work-item in the calling range cooperatively copies a
 * portion of the data. The full work-group/sub-group covers
 * the entire transfer.
 * ============================================================ */

template <typename T>
inline void devicePut(
    const ucclDeviceWindow& win,
    const T* localData,
    size_t count,
    int peer,
    size_t peerByteOffset,
    sycl::nd_item<1> item)
{
    T* remoteDst = reinterpret_cast<T*>(
        static_cast<char*>(win.remotePtrs[peer]) + peerByteOffset);

    int gid = item.get_global_id(0);
    int stride = item.get_global_range(0);

    for (size_t i = gid; i < count; i += stride) {
        remoteDst[i] = localData[i];
    }
}

/* Overload: offset in elements instead of bytes */
template <typename T>
inline void devicePutElements(
    const ucclDeviceWindow& win,
    const T* localData,
    size_t count,
    int peer,
    size_t peerElemOffset,
    sycl::nd_item<1> item)
{
    devicePut(win, localData, count, peer,
              peerElemOffset * sizeof(T), item);
}

/* ============================================================
 * Device Get — single work-item reads data from remote peer
 * ============================================================ */

template <typename T>
inline void deviceGet(
    const ucclDeviceWindow& win,
    T* localDst,
    size_t count,
    int peer,
    size_t peerByteOffset,
    sycl::nd_item<1> item)
{
    const T* remoteSrc = reinterpret_cast<const T*>(
        static_cast<const char*>(win.remotePtrs[peer]) + peerByteOffset);

    int gid = item.get_global_id(0);
    int stride = item.get_global_range(0);

    for (size_t i = gid; i < count; i += stride) {
        localDst[i] = remoteSrc[i];
    }
}

template <typename T>
inline void deviceGetElements(
    const ucclDeviceWindow& win,
    T* localDst,
    size_t count,
    int peer,
    size_t peerElemOffset,
    sycl::nd_item<1> item)
{
    deviceGet(win, localDst, count, peer,
              peerElemOffset * sizeof(T), item);
}

/* ============================================================
 * Device PutSignal — put data + atomically increment remote signal
 *
 * Must be called by the entire work-group/sub-group cooperatively.
 * After all work-items complete the put, work-item 0 issues a
 * system-scope fence and atomically increments the remote peer's
 * signal counter indexed by this rank (myRank).
 * ============================================================ */

template <typename T>
inline void devicePutSignal(
    const ucclDeviceWindow& win,
    const T* localData,
    size_t count,
    int peer,
    size_t peerByteOffset,
    sycl::nd_item<1> item)
{
    /* Step 1: cooperative put */
    T* remoteDst = reinterpret_cast<T*>(
        static_cast<char*>(win.remotePtrs[peer]) + peerByteOffset);

    int gid = item.get_global_id(0);
    int stride = item.get_global_range(0);

    for (size_t i = gid; i < count; i += stride) {
        remoteDst[i] = localData[i];
    }

    /* Step 2: barrier to ensure all stores are visible */
    sycl::group_barrier(item.get_group());

    /* Step 3: work-item 0 issues fence + atomic signal increment */
    if (item.get_local_id(0) == 0) {
        fenceSystem();

        uint64_t* sigPtr = win.remoteSignals[peer] + win.myRank;
        sycl::atomic_ref<uint64_t,
            sycl::memory_order::acq_rel,
            sycl::memory_scope::system,
            sycl::access::address_space::global_space>
            ref(*sigPtr);
        ref.fetch_add(1);
    }
}

/* ============================================================
 * Device GetSignal — get data + atomically increment local signal
 *
 * After cooperative get completes, work-item 0 increments the
 * local signal counter for `peer` to notify that data is available.
 * ============================================================ */

template <typename T>
inline void deviceGetSignal(
    const ucclDeviceWindow& win,
    T* localDst,
    size_t count,
    int peer,
    size_t peerByteOffset,
    sycl::nd_item<1> item)
{
    /* Step 1: cooperative get */
    const T* remoteSrc = reinterpret_cast<const T*>(
        static_cast<const char*>(win.remotePtrs[peer]) + peerByteOffset);

    int gid = item.get_global_id(0);
    int stride = item.get_global_range(0);

    for (size_t i = gid; i < count; i += stride) {
        localDst[i] = remoteSrc[i];
    }

    /* Step 2: barrier */
    sycl::group_barrier(item.get_group());

    /* Step 3: work-item 0 signals completion */
    if (item.get_local_id(0) == 0) {
        fenceSystem();

        uint64_t* sigPtr = const_cast<uint64_t*>(win.localSignals) + peer;
        sycl::atomic_ref<uint64_t,
            sycl::memory_order::acq_rel,
            sycl::memory_scope::system,
            sycl::access::address_space::global_space>
            ref(*sigPtr);
        ref.fetch_add(1);
    }
}

/* ============================================================
 * Device Signal — send signal to remote peer without data
 *
 * Only work-item 0 should call this (or guard with local_id==0).
 * Guarantees all prior devicePutSignal() stores to the same
 * peer are visible before this signal is observed.
 * ============================================================ */

inline void deviceSignal(
    const ucclDeviceWindow& win,
    int peer)
{
    fenceSystem();

    uint64_t* sigPtr = win.remoteSignals[peer] + win.myRank;
    sycl::atomic_ref<uint64_t,
        sycl::memory_order::acq_rel,
        sycl::memory_scope::system,
        sycl::access::address_space::global_space>
        ref(*sigPtr);
    ref.fetch_add(1);
}

/* ============================================================
 * Device WaitSignal — spin-wait until signal counter >= expected
 *
 * Only work-item 0 should call this (or guard with local_id==0),
 * then broadcast readiness to the group via barrier.
 *
 * Spins on the local signal counter for `peer` until it reaches
 * `expectedCount`. After return, data from the corresponding
 * devicePutSignal() is guaranteed visible in local memory.
 * ============================================================ */

inline bool deviceWaitSignal(
    const ucclDeviceWindow& win,
    int peer,
    uint64_t expectedCount,
    uint64_t maxSpins = 100'000'000ULL)
{
    uint64_t* sigPtr = const_cast<uint64_t*>(win.localSignals) + peer;

    /* Bounded spin-wait with acquire semantics.
     * Returns true if signal arrived, false on timeout. */
    for (uint64_t spin = 0; spin < maxSpins; spin++) {
        sycl::atomic_ref<uint64_t,
            sycl::memory_order::acquire,
            sycl::memory_scope::system,
            sycl::access::address_space::global_space>
            ref(*sigPtr);

        if (ref.load() >= expectedCount) return true;
    }
    return false; /* timeout */
}

/* Cooperative version: work-item 0 waits, then broadcasts via barrier */
inline bool deviceWaitSignalCooperative(
    const ucclDeviceWindow& win,
    int peer,
    uint64_t expectedCount,
    sycl::nd_item<1> item,
    uint64_t maxSpins = 100'000'000ULL)
{
    bool ok = true;
    if (item.get_local_id(0) == 0) {
        ok = deviceWaitSignal(win, peer, expectedCount, maxSpins);
    }
    sycl::group_barrier(item.get_group());
    return ok;
}

/* Wait for signals from multiple peers */
inline bool deviceWaitSignalMulti(
    const ucclDeviceWindow& win,
    const int* peers,
    const uint64_t* expectedCounts,
    int nPeers,
    uint64_t maxSpins = 100'000'000ULL)
{
    for (int p = 0; p < nPeers; p++) {
        if (!deviceWaitSignal(win, peers[p], expectedCounts[p], maxSpins))
            return false;
    }
    return true;
}

/* ============================================================
 * Device PutReduce — Read remote, reduce with local, write back
 *
 * A common pattern in collective algorithms: read peer's data,
 * reduce with local data, write result to peer's buffer.
 * ============================================================ */

template <typename T, typename RedOp>
inline void deviceGetReduce(
    const ucclDeviceWindow& win,
    const T* localData,
    T* localResult,
    size_t count,
    int peer,
    size_t peerByteOffset,
    sycl::nd_item<1> item)
{
    const T* remoteSrc = reinterpret_cast<const T*>(
        static_cast<const char*>(win.remotePtrs[peer]) + peerByteOffset);

    RedOp op;
    int gid = item.get_global_id(0);
    int stride = item.get_global_range(0);

    for (size_t i = gid; i < count; i += stride) {
        T remoteVal = remoteSrc[i];
        T localVal = localData[i];
        localResult[i] = op(localVal, remoteVal);
    }
}

/* ============================================================
 * Host helper: populate ucclDeviceWindow from host-side ucclWindow
 *
 * Call this on the host to create a device-capturable window
 * descriptor from a registered host-side window.
 * ============================================================ */

inline ucclDeviceWindow makeDeviceWindow(
    void* localBuff,
    void** remotePtrs,
    uint64_t* localSignals,
    uint64_t** remoteSignals,
    size_t size,
    int nRanks,
    int myRank)
{
    ucclDeviceWindow dw;
    dw.localBuff = localBuff;
    dw.size = size;
    dw.nRanks = nRanks;
    dw.myRank = myRank;
    dw.localSignals = localSignals;

    for (int r = 0; r < nRanks && r < UCCL_RMA_MAX_PEERS; r++) {
        dw.remotePtrs[r] = remotePtrs[r];
        dw.remoteSignals[r] = remoteSignals[r];
    }

    return dw;
}

} /* namespace uccl */
