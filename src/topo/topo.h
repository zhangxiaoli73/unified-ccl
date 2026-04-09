#pragma once

#include "../include/uccl_common.h"
#include "../algorithms/algorithm.h"
#include "../protocols/protocol.h"

#include <vector>
#include <string>

namespace uccl {

/* Link types between GPUs */
enum ucclLinkType {
    UCCL_LINK_PCIE_SAME_SWITCH = 0,  /* same PCIe switch */
    UCCL_LINK_PCIE_CROSS_SWITCH = 1, /* same socket, cross PCIe switch */
    UCCL_LINK_UPI = 2,               /* cross socket (UPI) */
    UCCL_LINK_NET = 3,               /* cross node (network) */
    UCCL_NUM_LINK_TYPES
};

/* GPU device info */
struct ucclGpuInfo {
    int devIndex;           /* SYCL device index */
    int pciDomain;
    int pciBus;
    int pciDevice;
    int pciFunction;
    int numaNode;           /* NUMA node -> corresponds to socket */
    int socketId;           /* CPU socket ID */
};

/* Link between two GPUs */
struct ucclTopoLink {
    int gpu1;
    int gpu2;
    ucclLinkType type;
    float bandwidth;        /* GB/s */
    float latency;          /* us */
};

/* Topology structure */
struct ucclTopology {
    int nGpus;
    std::vector<ucclGpuInfo> gpus;
    std::vector<ucclTopoLink> links;

    /* Get link type between two GPUs */
    ucclLinkType getLinkType(int gpu1, int gpu2) const;
};

/* Tuning parameters derived from topology */
struct ucclTopoTuning {
    int nChannels;              /* number of parallel channels */
    int nThreads;               /* threads per kernel */
    size_t chunkSize;           /* chunk size per transfer */
    ucclAlgorithm_t algorithm;  /* selected algorithm */
    ucclProtocol_t protocol;    /* selected protocol */
    float bandwidth;            /* estimated bandwidth */
    float latency;              /* estimated latency */
};

/* ============================================================
 * Topology detection functions
 * ============================================================ */

/* Detect local topology */
ucclResult_t ucclTopoDetect(ucclTopology* topo);

/* PCIe topology detection */
ucclResult_t ucclTopoPciDetect(ucclTopology* topo);

/* UPI / NUMA / cross-socket detection */
ucclResult_t ucclTopoUpiDetect(ucclTopology* topo);

/* Compute optimal ring ordering to minimize UPI crossings */
std::vector<int> computeRingOrder(const ucclTopology& topo);

/* Tune algorithm/protocol based on topology and message size */
ucclResult_t ucclTopoTune(const ucclTopology* topo,
                          size_t messageSize,
                          ucclTopoTuning* tuning);

} /* namespace uccl */
