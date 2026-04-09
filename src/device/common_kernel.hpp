#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>

/* Common kernel utilities for SYCL device code. */

namespace uccl {

/* Get sub-group (warp) size at runtime.
 * Intel GPUs typically have sub-group size of 16 or 32. */
inline int getSubGroupSize(sycl::sub_group sg) {
    return static_cast<int>(sg.get_local_range()[0]);
}

/* Get lane ID within sub-group */
inline int getLaneId(sycl::sub_group sg) {
    return static_cast<int>(sg.get_local_id()[0]);
}

/* Get sub-group ID within work-group */
inline int getSubGroupId(sycl::sub_group sg) {
    return static_cast<int>(sg.get_group_id()[0]);
}

/* Get number of sub-groups in work-group */
inline int getSubGroupCount(sycl::sub_group sg) {
    return static_cast<int>(sg.get_group_range()[0]);
}

/* Work-group barrier (equivalent to __syncthreads) */
inline void workGroupBarrier(sycl::nd_item<1> item) {
    sycl::group_barrier(item.get_group());
}

/* Sub-group barrier (equivalent to __syncwarp) */
inline void subGroupBarrier(sycl::sub_group sg) {
    sycl::group_barrier(sg);
}

/* Sub-group any (equivalent to __any_sync) */
inline bool subGroupAny(sycl::sub_group sg, bool pred) {
    return sycl::any_of_group(sg, pred);
}

/* Sub-group shuffle (equivalent to __shfl_sync) */
template <typename T>
inline T subGroupShuffle(sycl::sub_group sg, T val, int srcLane) {
    return sycl::select_from_group(sg, val,
                                   static_cast<size_t>(srcLane));
}

/* Funnel shift right: (hi:lo) >> shift
 * Equivalent to CUDA __funnelshift_r(lo, hi, shift) */
inline uint32_t funnelShiftRight(uint32_t lo, uint32_t hi,
                                  uint32_t shift) {
    shift &= 31;
    return (lo >> shift) | (hi << (32 - shift));
}

/* Busy-wait spin loop with a counter for timeout detection.
 * Used for polling head/tail pointers in producer-consumer sync. */
inline bool spinWait(const volatile uint64_t* ptr,
                     uint64_t expected,
                     uint64_t maxSpins = 1000000) {
    for (uint64_t spin = 0; spin < maxSpins; spin++) {
        if (*ptr >= expected) return true;
    }
    return false; /* timeout */
}

} /* namespace uccl */
