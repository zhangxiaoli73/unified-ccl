#include "topo.h"
#include "../misc/debug.h"

#include <algorithm>
#include <map>
#include <vector>

namespace uccl {

/* Compute optimal ring ordering to minimize UPI crossings.
 *
 * Strategy (mirrors NCCL ncclTopoCompute ring ordering):
 * 1. Group GPUs by socketId
 * 2. Sort GPUs within each socket by PCIe bus address
 * 3. Chain sockets sequentially to minimize UPI crossings to exactly 2
 *
 * Example:
 *   Unoptimized:  G0 -> G4 -> G1 -> G5 -> G2 -> G6 -> G3 -> G7  (5 UPI)
 *   Optimized:    G0 -> G1 -> G2 -> G3 -> G4 -> G5 -> G6 -> G7  (2 UPI)
 */
std::vector<int> computeRingOrder(const ucclTopology& topo) {
    if (topo.nGpus <= 1) {
        std::vector<int> order;
        for (int i = 0; i < topo.nGpus; i++) {
            order.push_back(i);
        }
        return order;
    }

    /* Step 1: Group GPUs by socket */
    std::map<int, std::vector<int>> socketGroups;
    for (int i = 0; i < topo.nGpus; i++) {
        int socketId = topo.gpus[i].socketId;
        socketGroups[socketId].push_back(i);
    }

    /* Step 2: Sort GPUs within each socket by PCIe bus address.
     * This ensures GPUs under the same PCIe switch are adjacent. */
    for (auto& [socketId, gpuList] : socketGroups) {
        std::sort(gpuList.begin(), gpuList.end(),
                  [&topo](int a, int b) {
                      const ucclGpuInfo& ga = topo.gpus[a];
                      const ucclGpuInfo& gb = topo.gpus[b];
                      /* Sort by domain, then bus, then device, then function */
                      if (ga.pciDomain != gb.pciDomain)
                          return ga.pciDomain < gb.pciDomain;
                      if (ga.pciBus != gb.pciBus)
                          return ga.pciBus < gb.pciBus;
                      if (ga.pciDevice != gb.pciDevice)
                          return ga.pciDevice < gb.pciDevice;
                      return ga.pciFunction < gb.pciFunction;
                  });
    }

    /* Step 3: Chain sockets sequentially.
     * The ring visits all GPUs in socket 0, then socket 1, etc.
     * This produces exactly 2 UPI crossings for a 2-socket system
     * (one from last GPU of socket 0 to first GPU of socket 1,
     *  and one wrapping from last GPU of socket 1 back to first
     *  GPU of socket 0). */
    std::vector<int> order;
    order.reserve(topo.nGpus);

    /* Sort socket IDs for deterministic ordering */
    std::vector<int> sortedSockets;
    for (const auto& [socketId, _] : socketGroups) {
        sortedSockets.push_back(socketId);
    }
    std::sort(sortedSockets.begin(), sortedSockets.end());

    for (int socketId : sortedSockets) {
        const auto& gpuList = socketGroups[socketId];
        for (int gpuIdx : gpuList) {
            order.push_back(gpuIdx);
        }
    }

    /* Log the computed ring order */
    std::string orderStr;
    for (size_t i = 0; i < order.size(); i++) {
        if (i > 0) orderStr += " -> ";
        orderStr += "G" + std::to_string(order[i]);
    }
    UCCL_LOG(INFO, "Ring order: %s", orderStr.c_str());

    /* Count UPI crossings for logging */
    int upiCrossings = 0;
    for (size_t i = 0; i < order.size(); i++) {
        int curr = order[i];
        int next = order[(i + 1) % order.size()];
        if (topo.gpus[curr].socketId != topo.gpus[next].socketId) {
            upiCrossings++;
        }
    }
    UCCL_LOG(INFO, "Ring UPI crossings: %d", upiCrossings);

    return order;
}

/* Get link type between two GPUs from topology */
ucclLinkType ucclTopology::getLinkType(int gpu1, int gpu2) const {
    for (const auto& link : links) {
        if ((link.gpu1 == gpu1 && link.gpu2 == gpu2) ||
            (link.gpu1 == gpu2 && link.gpu2 == gpu1)) {
            return link.type;
        }
    }
    /* Default: cross-switch PCIe if same socket, UPI if not */
    if (gpu1 >= 0 && gpu1 < nGpus && gpu2 >= 0 && gpu2 < nGpus) {
        if (gpus[gpu1].socketId == gpus[gpu2].socketId) {
            return UCCL_LINK_PCIE_CROSS_SWITCH;
        }
        return UCCL_LINK_UPI;
    }
    return UCCL_LINK_NET;
}

/* Full topology detection: PCIe + UPI */
ucclResult_t ucclTopoDetect(ucclTopology* topo) {
    ucclResult_t res;

    /* Phase 1: PCIe topology */
    res = ucclTopoPciDetect(topo);
    if (res != ucclSuccess) return res;

    /* Phase 2: UPI / NUMA topology */
    res = ucclTopoUpiDetect(topo);
    if (res != ucclSuccess) return res;

    return ucclSuccess;
}

/* Tune algorithm/protocol selection based on topology and message size.
 *
 * Strategy (mirrors NCCL ncclTopoTuneModel):
 *   < 4KB:          LL128 (low latency, flag-based sync)
 *   4KB - 512KB:    LL128 or Simple (depends on topology)
 *   > 512KB:        Simple (high bandwidth, pipelined)
 */
ucclResult_t ucclTopoTune(const ucclTopology* topo,
                          size_t messageSize,
                          ucclTopoTuning* tuning) {
    if (topo == nullptr || tuning == nullptr) {
        return ucclInvalidArgument;
    }

    /* Default algorithm: Ring (MVP) */
    tuning->algorithm = UCCL_ALGO_RING;

    /* Protocol selection based on message size */
    if (messageSize < 4096) {
        /* Small messages: LL128 for low latency */
        tuning->protocol = UCCL_PROTO_LL128;
        tuning->chunkSize = 4096;
        tuning->nThreads = 256;
    } else if (messageSize <= 524288) {
        /* Medium messages: topology-dependent */
        /* If all GPUs on same socket, LL128 might still be good */
        bool allSameSocket = true;
        for (int i = 1; i < topo->nGpus; i++) {
            if (topo->gpus[i].socketId != topo->gpus[0].socketId) {
                allSameSocket = false;
                break;
            }
        }
        tuning->protocol = allSameSocket ?
            UCCL_PROTO_LL128 : UCCL_PROTO_SIMPLE;
        tuning->chunkSize = 65536;
        tuning->nThreads = 512;
    } else {
        /* Large messages: Simple for maximum bandwidth */
        tuning->protocol = UCCL_PROTO_SIMPLE;
        tuning->chunkSize = 524288;
        tuning->nThreads = 512;
    }

    /* Channel count based on GPU count and topology */
    if (topo->nGpus <= 2) {
        tuning->nChannels = 2;
    } else if (topo->nGpus <= 4) {
        tuning->nChannels = 4;
    } else {
        tuning->nChannels = 8;
    }

    /* Clamp to max channels */
    if (tuning->nChannels > UCCL_MAX_CHANNELS) {
        tuning->nChannels = UCCL_MAX_CHANNELS;
    }

    /* Estimate bandwidth and latency */
    tuning->bandwidth = 0.0f;
    tuning->latency = 0.0f;
    if (!topo->links.empty()) {
        /* Find minimum bandwidth and maximum latency */
        float minBw = topo->links[0].bandwidth;
        float maxLat = topo->links[0].latency;
        for (const auto& link : topo->links) {
            if (link.bandwidth < minBw) minBw = link.bandwidth;
            if (link.latency > maxLat) maxLat = link.latency;
        }
        tuning->bandwidth = minBw;
        tuning->latency = maxLat;
    }

    return ucclSuccess;
}

} /* namespace uccl */
