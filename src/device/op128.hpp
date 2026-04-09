#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

/* 128-bit load/store primitives for SYCL.
 *
 * Mirrors NCCL op128.h — provides 128-bit atomic-like operations
 * needed by LL128 protocol for flag-based synchronization.
 *
 * On Intel GPUs, we use sycl::vec<uint64_t, 2> for 128-bit operations
 * and sycl::atomic_ref for memory ordering guarantees. */

namespace uccl {

/* Pack two uint64_t into a 128-bit value */
struct alignas(16) uint128_t {
    uint64_t lo;
    uint64_t hi;
};

/* 128-bit volatile load from global memory */
inline uint128_t load128(const volatile uint128_t* ptr) {
    /* Use sycl::vec for 128-bit load */
    const auto* vptr = reinterpret_cast<
        const volatile sycl::vec<uint64_t, 2>*>(ptr);
    sycl::vec<uint64_t, 2> val = *vptr;
    uint128_t result;
    result.lo = val[0];
    result.hi = val[1];
    return result;
}

/* 128-bit volatile store to global memory */
inline void store128(volatile uint128_t* ptr, uint128_t val) {
    auto* vptr = reinterpret_cast<
        volatile sycl::vec<uint64_t, 2>*>(ptr);
    sycl::vec<uint64_t, 2> v;
    v[0] = val.lo;
    v[1] = val.hi;
    *vptr = v;
}

/* Relaxed atomic load (64-bit) */
inline uint64_t loadRelaxed(const uint64_t* ptr) {
    sycl::atomic_ref<uint64_t,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(*const_cast<uint64_t*>(ptr));
    return ref.load();
}

/* Relaxed atomic store (64-bit) */
inline void storeRelaxed(uint64_t* ptr, uint64_t val) {
    sycl::atomic_ref<uint64_t,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(*ptr);
    ref.store(val);
}

/* Acquire load (64-bit) — for reading flags */
inline uint64_t loadAcquire(const uint64_t* ptr) {
    sycl::atomic_ref<uint64_t,
                     sycl::memory_order::acquire,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        ref(*const_cast<uint64_t*>(ptr));
    return ref.load();
}

/* Release store (64-bit) — for writing flags */
inline void storeRelease(uint64_t* ptr, uint64_t val) {
    sycl::atomic_ref<uint64_t,
                     sycl::memory_order::release,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        ref(*ptr);
    ref.store(val);
}

/* GPU memory fence (device scope) */
inline void fenceDevice() {
    sycl::atomic_fence(sycl::memory_order::acq_rel,
                       sycl::memory_scope::device);
}

/* System memory fence (visible to host and all devices) */
inline void fenceSystem() {
    sycl::atomic_fence(sycl::memory_order::acq_rel,
                       sycl::memory_scope::system);
}

} /* namespace uccl */
