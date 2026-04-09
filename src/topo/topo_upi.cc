#include "topo.h"
#include "../misc/debug.h"

#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <map>

namespace uccl {

/* Read NUMA node for a PCI device from sysfs */
static int readNumaNode(const ucclGpuInfo& gpu) {
#ifdef __linux__
    char path[512];
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%04x:%02x:%02x.%x/numa_node",
             gpu.pciDomain, gpu.pciBus, gpu.pciDevice, gpu.pciFunction);

    std::ifstream f(path);
    int node = -1;
    if (f.is_open()) {
        f >> node;
        /* -1 means NUMA node is not available / single socket */
    }
    return node;
#else
    (void)gpu;
    return -1;
#endif
}

/* Map NUMA node to socket ID.
 * On most Linux systems, NUMA node == socket for simple topologies.
 * For more complex topologies (e.g., sub-NUMA clustering),
 * we read /sys/devices/system/node/ to build the mapping. */
static std::map<int, int> buildNumaToSocketMap() {
    std::map<int, int> numaToSocket;

#ifdef __linux__
    /* Read /sys/devices/system/node/nodeN/cpulist to group NUMA nodes
     * by physical package (socket). For simplicity, assume 1:1 mapping. */
    for (int node = 0; node < 16; node++) {
        char path[256];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/node/node%d", node);

        std::ifstream check(std::string(path) + "/cpulist");
        if (!check.is_open()) break;

        /* Read the physical package id of the first CPU in this NUMA node */
        std::string cpulist;
        std::getline(check, cpulist);
        if (cpulist.empty()) continue;

        /* Parse first CPU number from cpulist (e.g., "0-15" -> 0) */
        int firstCpu = 0;
        sscanf(cpulist.c_str(), "%d", &firstCpu);

        /* Read physical package id */
        char pkgPath[256];
        snprintf(pkgPath, sizeof(pkgPath),
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id",
                 firstCpu);
        std::ifstream pkgFile(pkgPath);
        int socketId = node; /* default: NUMA node == socket */
        if (pkgFile.is_open()) {
            pkgFile >> socketId;
        }

        numaToSocket[node] = socketId;
    }
#endif

    return numaToSocket;
}

/* Detect UPI / NUMA topology: assign socketId to each GPU */
ucclResult_t ucclTopoUpiDetect(ucclTopology* topo) {
    if (topo == nullptr) {
        return ucclInvalidArgument;
    }

    auto numaToSocket = buildNumaToSocketMap();

    for (auto& gpu : topo->gpus) {
        /* Read NUMA node from sysfs */
        int numaNode = readNumaNode(gpu);
        gpu.numaNode = numaNode;

        /* Map NUMA node to socket */
        if (numaNode >= 0) {
            auto it = numaToSocket.find(numaNode);
            if (it != numaToSocket.end()) {
                gpu.socketId = it->second;
            } else {
                gpu.socketId = numaNode; /* fallback: NUMA == socket */
            }
        } else {
            gpu.socketId = 0; /* single socket system */
        }

        UCCL_LOG(INFO, "GPU %d: NUMA node=%d, socket=%d",
                 gpu.devIndex, gpu.numaNode, gpu.socketId);
    }

    /* Update link types based on socket information.
     * Key scenario: intra-node multi-GPU with PCIe links,
     * where cross-socket communication goes through UPI. */
    for (auto& link : topo->links) {
        const ucclGpuInfo& g1 = topo->gpus[link.gpu1];
        const ucclGpuInfo& g2 = topo->gpus[link.gpu2];

        if (g1.socketId != g2.socketId) {
            /* Cross-socket: UPI link */
            link.type = UCCL_LINK_UPI;
            link.bandwidth = 40.0f;  /* ~40-80 GB/s UPI */
            link.latency = 1.0f;
        }
    }

    return ucclSuccess;
}

} /* namespace uccl */
