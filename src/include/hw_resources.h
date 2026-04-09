#pragma once

#include "uccl_common.h"

#include <sycl/sycl.hpp>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#if defined(UCCL_HAS_LEVEL_ZERO) && __has_include(<level_zero/ze_api.h>)
#include <level_zero/ze_api.h>
#define UCCL_HWRES_HAS_LEVEL_ZERO 1
#else
#define UCCL_HWRES_HAS_LEVEL_ZERO 0
#endif

/* Hardware Resource Descriptor.
 *
 * Describes the GPU hardware resources available for collective operations.
 * Used to control EU (Execution Unit) allocation and copy engine usage.
 *
 * Intel GPU hardware resources:
 *   - EU (Execution Units): Compute cores that run SYCL kernels.
 *     The number of EUs can be queried via SYCL device info and
 *     optionally restricted to a subset for collective operations.
 *   - Copy Engines (BCS / Blitter Copy Service): DMA engines that
 *     perform memory copies independently of EUs. Intel Data Center
 *     GPU Max (Ponte Vecchio) has multiple link copy engines.
 *
 * Execution modes:
 *   - UCCL_EXEC_EU_ONLY:       All work on EUs (compute kernels)
 *   - UCCL_EXEC_COPY_ENGINE:   Data movement via copy engines, reduction on EUs
 *   - UCCL_EXEC_HYBRID:        Overlap copy engine transfers with EU compute
 */

namespace uccl {

/* Execution mode for collective operations */
enum ucclExecMode_t {
    UCCL_EXEC_EU_ONLY = 0,       /* Default: all work on EUs */
    UCCL_EXEC_COPY_ENGINE = 1,   /* Copy engine for data movement, EU for reduce */
    UCCL_EXEC_HYBRID = 2,        /* Overlap copy engine + EU (future) */
    UCCL_NUM_EXEC_MODES
};

/* Copy engine descriptor */
struct ucclCopyEngine {
    int nEngines;               /* Number of copy engines available */
    int engineIndex;            /* Which engine to use (-1 = auto) */
    sycl::queue* copyQueue;     /* Dedicated queue for copy engine submissions */
    bool available;             /* Whether copy engines are detected */
};

/* Hardware resource descriptor */
struct ucclHwResources {
    /* EU (Execution Unit) configuration */
    int euCount;                /* Number of EUs to use (0 = all available) */
    int euCountMax;             /* Max EUs on this device (from SYCL query) */
    int threadsPerEU;           /* HW threads per EU */
    int slmSizePerSS;           /* SLM (Shared Local Memory) per sub-slice, bytes */

    /* Sub-slice / Xe-core information */
    int numSubSlices;           /* Number of sub-slices (Xe-cores) */
    int numSlices;              /* Number of slices */

    /* Copy engine configuration */
    ucclCopyEngine copyEngine;

    /* Execution mode */
    ucclExecMode_t execMode;

    /* Compute queue (distinct from copy queue) */
    sycl::queue* computeQueue;
};

/* Query hardware resources from a SYCL device.
 * Populates the ucclHwResources struct with device capabilities.
 *
 * @param hw       Output hardware resource descriptor
 * @param device   SYCL device to query
 * @param queue    Default SYCL queue for this device
 * @return ucclSuccess on success */
inline ucclResult_t ucclQueryHwResources(ucclHwResources* hw,
                                         const sycl::device& device,
                                         sycl::queue* queue) {
    if (hw == nullptr || queue == nullptr) return ucclInvalidArgument;

    /* Query EU count from SYCL device */
    hw->euCountMax = static_cast<int>(
        device.get_info<sycl::info::device::max_compute_units>());
    hw->euCount = hw->euCountMax;  /* Default: use all EUs */

    /* Threads per EU — Intel GPUs typically have 7 or 8 threads per EU */
    hw->threadsPerEU = 7;  /* Default for PVC; could be queried via L0 */

    /* Sub-slice / slice counts — approximate from EU count.
     * PVC: 8 EUs per sub-slice, 8 sub-slices per slice.
     * For accurate counts, Level Zero API would be used. */
    hw->numSubSlices = (hw->euCountMax + 7) / 8;
    hw->numSlices = (hw->numSubSlices + 7) / 8;

    /* SLM per sub-slice: 128KB on PVC */
    hw->slmSizePerSS = 128 * 1024;

    /* Copy engine detection.
     * Intel GPUs expose copy engines as separate queue families.
     * We detect and create a dedicated copy queue if available. */
    hw->copyEngine.nEngines = 0;
    hw->copyEngine.engineIndex = -1;
    hw->copyEngine.copyQueue = nullptr;
    hw->copyEngine.available = false;

#if UCCL_HWRES_HAS_LEVEL_ZERO
    /* Query copy engine count via Level Zero */
    auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
    uint32_t numQueueGroups = 0;
    zeDeviceGetCommandQueueGroupProperties(zeDevice, &numQueueGroups, nullptr);

    std::vector<ze_command_queue_group_properties_t> queueGroupProps(numQueueGroups);
    for (auto& p : queueGroupProps) {
        p.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        p.pNext = nullptr;
    }
    zeDeviceGetCommandQueueGroupProperties(zeDevice, &numQueueGroups,
                                            queueGroupProps.data());

    for (uint32_t i = 0; i < numQueueGroups; i++) {
        if ((queueGroupProps[i].flags &
             ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
            !(queueGroupProps[i].flags &
              ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) {
            hw->copyEngine.nEngines = queueGroupProps[i].numQueues;
            hw->copyEngine.engineIndex = static_cast<int>(i);
            hw->copyEngine.available = true;
            break;
        }
    }

    if (hw->copyEngine.available) {
        /* Create a dedicated in-order queue for copy engine */
        hw->copyEngine.copyQueue = new sycl::queue(
            device,
            sycl::property_list{sycl::property::queue::in_order()});
    }
#else
    /* Without Level Zero, assume copy engine available via default queue
     * (SYCL runtime handles engine selection) */
    hw->copyEngine.nEngines = 1;
    hw->copyEngine.available = true;
    hw->copyEngine.copyQueue = queue;  /* Fallback: use same queue */
#endif

    /* Default execution mode */
    hw->execMode = UCCL_EXEC_EU_ONLY;

    /* Check environment variable for execution mode override */
    const char* execEnv = std::getenv("UCCL_EXEC_MODE");
    if (execEnv != nullptr) {
        if (std::strcmp(execEnv, "copy_engine") == 0 ||
            std::strcmp(execEnv, "ce") == 0) {
            hw->execMode = UCCL_EXEC_COPY_ENGINE;
        } else if (std::strcmp(execEnv, "hybrid") == 0) {
            hw->execMode = UCCL_EXEC_HYBRID;
        }
    }

    /* Check EU count override */
    const char* euEnv = std::getenv("UCCL_EU_COUNT");
    if (euEnv != nullptr) {
        int requested = std::atoi(euEnv);
        if (requested > 0 && requested <= hw->euCountMax) {
            hw->euCount = requested;
        }
    }

    hw->computeQueue = queue;

    return ucclSuccess;
}

/* Release hardware resources (free allocated queues, etc.) */
inline void ucclFreeHwResources(ucclHwResources* hw) {
    if (hw == nullptr) return;
#if UCCL_HWRES_HAS_LEVEL_ZERO
    if (hw->copyEngine.copyQueue != nullptr &&
        hw->copyEngine.copyQueue != hw->computeQueue) {
        delete hw->copyEngine.copyQueue;
        hw->copyEngine.copyQueue = nullptr;
    }
#endif
}

} /* namespace uccl */
