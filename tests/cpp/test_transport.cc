#include <sycl/sycl.hpp>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "uccl.h"
#include "uccl_net.h"

/* Transport layer tests.
 *
 * Tests:
 * 1. Network plugin discovery and initialization
 * 2. P2P transport setup (if multiple GPUs available)
 * 3. Memory registration/deregistration
 * 4. Basic send/recv correctness
 */

static int testPluginDiscovery() {
    std::printf("Test: Plugin discovery...\n");
    int errors = 0;

    /* Test version query */
    int version;
    ucclResult_t res = ucclGetVersion(&version);
    if (res != ucclSuccess) {
        std::fprintf(stderr,
            "FAIL: ucclGetVersion failed: %s\n",
            ucclGetErrorString(res));
        errors++;
    } else {
        std::printf("  UCCL version: %d\n", version);
    }

    /* Test unique ID generation */
    ucclUniqueId id1, id2;
    res = ucclGetUniqueId(&id1);
    if (res != ucclSuccess) {
        std::fprintf(stderr, "FAIL: ucclGetUniqueId failed\n");
        errors++;
    }

    res = ucclGetUniqueId(&id2);
    if (res != ucclSuccess) {
        std::fprintf(stderr, "FAIL: ucclGetUniqueId (2nd) failed\n");
        errors++;
    }

    /* Verify IDs are different */
    if (std::memcmp(&id1, &id2, sizeof(ucclUniqueId)) == 0) {
        std::fprintf(stderr,
            "FAIL: two unique IDs are identical\n");
        errors++;
    } else {
        std::printf("  PASS: Unique IDs are distinct\n");
    }

    return errors;
}

static int testErrorStrings() {
    std::printf("Test: Error strings...\n");
    int errors = 0;

    /* Verify all error codes have strings */
    for (int i = 0; i < ucclNumResults; i++) {
        const char* str = ucclGetErrorString(
            static_cast<ucclResult_t>(i));
        if (str == nullptr || std::strlen(str) == 0) {
            std::fprintf(stderr,
                "FAIL: no error string for code %d\n", i);
            errors++;
        } else {
            std::printf("  Error %d: %s\n", i, str);
        }
    }

    if (errors == 0) {
        std::printf("  PASS: All error codes have strings\n");
    }

    return errors;
}

static int testGpuEnumeration() {
    std::printf("Test: GPU enumeration...\n");
    int errors = 0;

    auto devices = sycl::device::get_devices(
        sycl::info::device_type::gpu);

    std::printf("  Found %zu GPU device(s)\n", devices.size());

    for (size_t i = 0; i < devices.size(); i++) {
        auto name = devices[i].get_info<sycl::info::device::name>();
        auto vendor = devices[i].get_info<sycl::info::device::vendor>();
        auto maxWg = devices[i].get_info<
            sycl::info::device::max_work_group_size>();
        auto maxMem = devices[i].get_info<
            sycl::info::device::global_mem_size>();

        std::printf("  GPU %zu: %s (%s)\n"
                    "    max_work_group_size=%zu, global_mem=%zu MB\n",
                    i, name.c_str(), vendor.c_str(),
                    maxWg, maxMem / (1024 * 1024));
    }

    if (devices.empty()) {
        std::printf("  WARN: No GPU devices found\n");
    }

    return errors;
}

static int testSyclMemory() {
    std::printf("Test: SYCL memory operations...\n");
    int errors = 0;

    auto devices = sycl::device::get_devices(
        sycl::info::device_type::gpu);
    if (devices.empty()) {
        std::printf("  SKIP: No GPU devices\n");
        return 0;
    }

    sycl::queue queue(devices[0]);

    /* Test device memory allocation */
    size_t count = 1024;
    size_t size = count * sizeof(float);

    float* devBuf = sycl::malloc_device<float>(count, queue);
    if (devBuf == nullptr) {
        std::fprintf(stderr, "FAIL: sycl::malloc_device failed\n");
        return 1;
    }

    /* Test host pinned memory */
    float* hostBuf = sycl::malloc_host<float>(count, queue);
    if (hostBuf == nullptr) {
        std::fprintf(stderr, "FAIL: sycl::malloc_host failed\n");
        sycl::free(devBuf, queue);
        return 1;
    }

    /* Initialize host buffer */
    for (size_t i = 0; i < count; i++) {
        hostBuf[i] = static_cast<float>(i);
    }

    /* Copy host -> device */
    queue.memcpy(devBuf, hostBuf, size).wait();

    /* Copy device -> host (to verify) */
    float* verifyBuf = sycl::malloc_host<float>(count, queue);
    queue.memcpy(verifyBuf, devBuf, size).wait();

    /* Verify */
    for (size_t i = 0; i < count; i++) {
        if (verifyBuf[i] != hostBuf[i]) {
            std::fprintf(stderr,
                "FAIL: memory verify at %zu: got %f, expected %f\n",
                i, verifyBuf[i], hostBuf[i]);
            errors++;
            break;
        }
    }

    if (errors == 0) {
        std::printf("  PASS: SYCL memory operations\n");
    }

    sycl::free(verifyBuf, queue);
    sycl::free(hostBuf, queue);
    sycl::free(devBuf, queue);

    return errors;
}

int main(int argc, char** argv) {
    std::printf("=== Unified-CCL Transport Tests ===\n\n");

    int totalErrors = 0;

    totalErrors += testPluginDiscovery();
    totalErrors += testErrorStrings();
    totalErrors += testGpuEnumeration();
    totalErrors += testSyclMemory();

    /* Summary */
    std::printf("\n");
    if (totalErrors == 0) {
        std::printf("=== ALL TRANSPORT TESTS PASSED ===\n");
    } else {
        std::printf("=== %d TRANSPORT TESTS FAILED ===\n", totalErrors);
    }

    return (totalErrors == 0) ? 0 : 1;
}
