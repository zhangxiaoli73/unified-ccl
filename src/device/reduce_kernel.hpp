#pragma once

#include <sycl/sycl.hpp>

/* SYCL Reduce Kernel — bf16/fp16 sum reduction.
 *
 * Mirrors NCCL reduce_kernel.h / applyReduce / applyPreOp / applyPostOp.
 *
 * Supported types:
 *   sycl::half               — fp16
 *   sycl::ext::oneapi::bfloat16 — bf16
 */

namespace uccl {

/* Sum reduction functor */
template <typename T>
struct ReduceSum {
    T operator()(T a, T b) const { return a + b; }
};

/* Apply reduction operation */
template <typename T, typename RedOp>
inline T applyReduce(RedOp op, T a, T b) {
    return op(a, b);
}

/* Vectorized reduce: apply reduction element-wise on arrays */
template <typename T, typename RedOp>
inline void reduceArray(RedOp op, T* dst, const T* src, int count) {
    for (int i = 0; i < count; i++) {
        dst[i] = op(dst[i], src[i]);
    }
}

/* Sub-group (warp) level reduce */
template <typename T, typename RedOp>
inline T subGroupReduce(sycl::sub_group sg, T val, RedOp op) {
    for (int offset = sg.get_local_range()[0] / 2; offset > 0;
         offset /= 2) {
        T other = sycl::shift_group_left(sg, val, offset);
        val = op(val, other);
    }
    return val;
}

/* Work-group level reduce using sub-groups.
 * Reduces 'val' across all work-items in the work-group. */
template <typename T, typename RedOp>
inline T workGroupReduce(sycl::nd_item<1> item,
                         sycl::local_accessor<T, 1> scratch,
                         T val, RedOp op) {
    auto sg = item.get_sub_group();
    auto wg = item.get_group();

    /* Phase 1: sub-group reduce */
    T sgResult = subGroupReduce(sg, val, op);

    /* Phase 2: store sub-group results to shared memory */
    int sgId = sg.get_group_id()[0];
    int sgCount = sg.get_group_range()[0];
    int laneId = sg.get_local_id()[0];

    if (laneId == 0) {
        scratch[sgId] = sgResult;
    }
    sycl::group_barrier(wg);

    /* Phase 3: first sub-group reduces all sub-group results */
    if (sgId == 0) {
        T v = (laneId < sgCount) ? scratch[laneId] : T(0);
        v = subGroupReduce(sg, v, op);
        if (laneId == 0) {
            scratch[0] = v;
        }
    }
    sycl::group_barrier(wg);

    return scratch[0];
}

} /* namespace uccl */
