#include "topo.h"
#include "../misc/debug.h"

#include <sycl/sycl.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <algorithm>

/* Try to include Level Zero for PCI property queries */
#if __has_include(<level_zero/ze_api.h>)
#include <level_zero/ze_api.h>
#define UCCL_HAS_LEVEL_ZERO 1
#else
#define UCCL_HAS_LEVEL_ZERO 0
#endif

namespace uccl {

/* Parse PCI BDF from sysfs path string like "0000:3a:00.0" */
static bool parsePciBdf(const std::string& bdf,
                        int& domain, int& bus, int& device, int& function) {
    if (sscanf(bdf.c_str(), "%x:%x:%x.%x",
               &domain, &bus, &device, &function) == 4) {
        return true;
    }
    return false;
}

/* Detect PCIe topology by reading sysfs and/or Level Zero */
ucclResult_t ucclTopoPciDetect(ucclTopology* topo) {
    if (topo == nullptr) {
        return ucclInvalidArgument;
    }

    /* Enumerate SYCL GPU devices */
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    topo->nGpus = static_cast<int>(devices.size());

    if (topo->nGpus == 0) {
        UCCL_LOG(WARN, "No GPU devices found");
        return ucclSuccess;
    }

    topo->gpus.resize(topo->nGpus);

    for (int i = 0; i < topo->nGpus; i++) {
        ucclGpuInfo& gpu = topo->gpus[i];
        gpu.devIndex = i;
        gpu.pciDomain = 0;
        gpu.pciBus = 0;
        gpu.pciDevice = 0;
        gpu.pciFunction = 0;
        gpu.numaNode = -1;
        gpu.socketId = -1;

#if UCCL_HAS_LEVEL_ZERO
        /* Try to get PCI properties via Level Zero */
        try {
            auto l0Device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[i]);
            ze_pci_ext_properties_t pciProps = {};
            pciProps.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
            if (zeDevicePciGetPropertiesExt(l0Device, &pciProps) == ZE_RESULT_SUCCESS) {
                gpu.pciDomain = static_cast<int>(pciProps.address.domain);
                gpu.pciBus = static_cast<int>(pciProps.address.bus);
                gpu.pciDevice = static_cast<int>(pciProps.address.device);
                gpu.pciFunction = static_cast<int>(pciProps.address.function);

                UCCL_LOG(INFO, "GPU %d PCI BDF: %04x:%02x:%02x.%x",
                         i, gpu.pciDomain, gpu.pciBus,
                         gpu.pciDevice, gpu.pciFunction);
            }
        } catch (...) {
            UCCL_LOG(WARN, "GPU %d: Level Zero PCI query failed, using defaults", i);
        }
#endif

        /* Fallback: try to read from sysfs on Linux */
#ifdef __linux__
        if (gpu.pciBus == 0 && gpu.pciDevice == 0) {
            /* Try to find GPU in /sys/class/drm/ */
            char sysfsPath[512];
            snprintf(sysfsPath, sizeof(sysfsPath),
                     "/sys/class/drm/card%d/device", i);

            /* Read PCI address from device symlink */
            char linkTarget[512];
            ssize_t len = readlink(sysfsPath, linkTarget, sizeof(linkTarget) - 1);
            if (len > 0) {
                linkTarget[len] = '\0';
                std::string target(linkTarget);
                /* Extract BDF from path like "../../../0000:3a:00.0" */
                size_t lastSlash = target.rfind('/');
                if (lastSlash != std::string::npos) {
                    std::string bdf = target.substr(lastSlash + 1);
                    parsePciBdf(bdf, gpu.pciDomain, gpu.pciBus,
                                gpu.pciDevice, gpu.pciFunction);
                }
            }
        }
#endif
    }

    /* Determine link types between GPU pairs */
    for (int i = 0; i < topo->nGpus; i++) {
        for (int j = i + 1; j < topo->nGpus; j++) {
            ucclTopoLink link;
            link.gpu1 = i;
            link.gpu2 = j;

            const ucclGpuInfo& g1 = topo->gpus[i];
            const ucclGpuInfo& g2 = topo->gpus[j];

            if (g1.socketId != -1 && g2.socketId != -1 &&
                g1.socketId != g2.socketId) {
                /* Cross-socket: UPI link */
                link.type = UCCL_LINK_UPI;
                link.bandwidth = 40.0f;  /* ~40-80 GB/s via UPI */
                link.latency = 1.0f;     /* ~1 us */
            } else if (g1.pciBus == g2.pciBus &&
                       g1.pciDomain == g2.pciDomain) {
                /* Same PCIe bus -> same switch */
                link.type = UCCL_LINK_PCIE_SAME_SWITCH;
                link.bandwidth = 64.0f;  /* PCIe Gen5 x16: ~64 GB/s */
                link.latency = 0.1f;     /* ~100 ns */
            } else if (g1.socketId == g2.socketId ||
                       (g1.socketId == -1 || g2.socketId == -1)) {
                /* Same socket, different switch or unknown */
                link.type = UCCL_LINK_PCIE_CROSS_SWITCH;
                link.bandwidth = 32.0f;  /* Limited by root complex */
                link.latency = 0.3f;     /* ~300 ns */
            } else {
                /* Default: treat as cross-switch */
                link.type = UCCL_LINK_PCIE_CROSS_SWITCH;
                link.bandwidth = 32.0f;
                link.latency = 0.3f;
            }

            topo->links.push_back(link);
        }
    }

    UCCL_LOG(INFO, "PCIe topology: detected %d GPUs, %zu links",
             topo->nGpus, topo->links.size());

    return ucclSuccess;
}

} /* namespace uccl */
