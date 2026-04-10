#pragma once

#include "uccl_common.h"
#include <cstdint>
#include <cstddef>

/* FIFO data structures for GPU kernel <-> proxy thread synchronization.
 *
 * These are in a separate header so the kernel code (primitives.hpp)
 * can include them without pulling in heavyweight host headers
 * (MPI, threads, etc.) from comm.h.
 *
 * Allocated via sycl::malloc_host (pinned host memory): both GPU
 * kernel and CPU proxy thread can directly load/store. */

namespace uccl {

/* FIFO entry operation types */
enum ucclFifoOpType : uint32_t {
    UCCL_OP_SEND   = 0,   /* collective: send bounce buffer data */
    UCCL_OP_RECV   = 1,   /* collective: recv data to bounce buffer */
    UCCL_OP_PUT    = 2,   /* device API: zero-copy RDMA user buffer */
    UCCL_OP_SIGNAL = 3,   /* device API: zero-byte notify remote */
    UCCL_OP_WAIT   = 4,   /* device API: wait for remote signal */
};

/* Single FIFO entry: describes one async network operation */
struct ucclFifoEntry {
    ucclFifoOpType opType;
    void* buff;                  /* collective: bounce buf; device API: user buf */
    size_t size;
    int peer;
    void* remoteBuff;            /* PUT: remote virtual address */
    void* mhandle;               /* NIC registration handle */
    volatile int done;           /* proxy -> kernel completion flag */
    void* request;               /* proxy internal: async request handle */
};

/* Connection FIFO: ring buffer of FIFO entries.
 * GPU kernel writes entries and advances tail.
 * Proxy thread reads entries, performs network I/O, sets done. */
struct ucclConnFifo {
    volatile uint64_t head;      /* proxy updates (kernel reads to check slot free) */
    volatile uint64_t tail;      /* kernel updates (proxy reads to find new entries) */
    uint64_t pendingHead;        /* proxy internal: tracks posted but incomplete ops */
    ucclFifoEntry entries[UCCL_STEPS];
};

} /* namespace uccl */
