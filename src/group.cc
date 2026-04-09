#include "include/uccl.h"
#include "misc/debug.h"

#include <atomic>

/* Group semantics — mirrors NCCL group API.
 *
 * Allows batching multiple collective operations:
 *   ucclGroupStart();
 *   ucclAllReduce(..., comm1, ...);
 *   ucclAllReduce(..., comm2, ...);
 *   ucclGroupEnd();
 *
 * Operations between GroupStart/GroupEnd are batched and
 * launched together for better overlap and efficiency. */

namespace uccl {

/* Thread-local group nesting depth */
static thread_local int groupDepth = 0;

/* Whether we are currently inside a group */
static thread_local bool inGroup = false;

} /* namespace uccl */

ucclResult_t ucclGroupStart(void) {
    uccl::groupDepth++;
    uccl::inGroup = true;
    UCCL_LOG(TRACE, "GroupStart: depth=%d", uccl::groupDepth);
    return ucclSuccess;
}

ucclResult_t ucclGroupEnd(void) {
    if (uccl::groupDepth <= 0) {
        UCCL_LOG(ERROR, "GroupEnd called without matching GroupStart");
        return ucclInvalidUsage;
    }

    uccl::groupDepth--;
    if (uccl::groupDepth == 0) {
        uccl::inGroup = false;
        /* Flush all batched operations.
         * In full implementation, this would launch all enqueued
         * operations that were deferred during the group. */
        UCCL_LOG(TRACE, "GroupEnd: flushing batched operations");
    }

    return ucclSuccess;
}
