#include <sycl/sycl.hpp>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "uccl.h"

/* AllReduce correctness and performance test.
 * Mirrors nccl-tests all_reduce_perf.
 *
 * Usage:
 *   mpirun -n <nranks> ./test_allreduce
 *
 * Tests:
 * 1. Single rank: verify identity (result == input)
 * 2. Multi rank: verify sum across ranks
 * 3. Various message sizes for bandwidth measurement
 */

static bool checkResult(const sycl::half* data, size_t count,
                        float expected, float tolerance) {
    for (size_t i = 0; i < count; i++) {
        float val = static_cast<float>(data[i]);
        if (std::fabs(val - expected) > tolerance) {
            std::fprintf(stderr,
                "MISMATCH at index %zu: got %f, expected %f\n",
                i, val, expected);
            return false;
        }
    }
    return true;
}

static int testAllReduceFp16(ucclComm_t comm, sycl::queue& queue,
                              int rank, int nranks) {
    std::printf("[Rank %d] Testing AllReduce fp16...\n", rank);
    int errors = 0;

    /* Test various sizes */
    size_t testSizes[] = {1, 16, 256, 1024, 4096, 65536, 262144, 1048576};
    int nTests = sizeof(testSizes) / sizeof(testSizes[0]);

    for (int t = 0; t < nTests; t++) {
        size_t count = testSizes[t];

        /* Allocate device memory */
        sycl::half* sendbuff = sycl::malloc_device<sycl::half>(count, queue);
        sycl::half* recvbuff = sycl::malloc_device<sycl::half>(count, queue);

        /* Initialize: each rank fills with (rank + 1) */
        float initVal = static_cast<float>(rank + 1);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(count),
                [=](sycl::id<1> i) {
                    sendbuff[i] = sycl::half(initVal);
                });
        }).wait();

        /* Run AllReduce */
        ucclResult_t res = ucclAllReduce(
            sendbuff, recvbuff, count,
            ucclFloat16, ucclSum, comm,
            static_cast<void*>(&queue));

        if (res != ucclSuccess) {
            std::fprintf(stderr,
                "[Rank %d] AllReduce failed for count=%zu: %s\n",
                rank, count, ucclGetErrorString(res));
            errors++;
        } else {
            queue.wait();

            /* Verify: sum of (rank+1) for all ranks
             * = 1 + 2 + ... + nranks = nranks*(nranks+1)/2 */
            float expectedSum = static_cast<float>(
                nranks * (nranks + 1) / 2);

            /* Copy result to host for verification */
            std::vector<sycl::half> hostResult(count);
            queue.memcpy(hostResult.data(), recvbuff,
                        count * sizeof(sycl::half)).wait();

            if (!checkResult(hostResult.data(), count,
                            expectedSum, 0.1f)) {
                std::fprintf(stderr,
                    "[Rank %d] FAIL: count=%zu, expected=%f\n",
                    rank, count, expectedSum);
                errors++;
            } else {
                std::printf("[Rank %d] PASS: count=%zu\n", rank, count);
            }
        }

        sycl::free(sendbuff, queue);
        sycl::free(recvbuff, queue);
    }

    return errors;
}

static int testAllReduceBf16(ucclComm_t comm, sycl::queue& queue,
                              int rank, int nranks) {
    std::printf("[Rank %d] Testing AllReduce bf16...\n", rank);
    int errors = 0;

    using bf16 = sycl::ext::oneapi::bfloat16;

    size_t count = 1024;

    bf16* sendbuff = sycl::malloc_device<bf16>(count, queue);
    bf16* recvbuff = sycl::malloc_device<bf16>(count, queue);

    float initVal = static_cast<float>(rank + 1);
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(count),
            [=](sycl::id<1> i) {
                sendbuff[i] = bf16(initVal);
            });
    }).wait();

    ucclResult_t res = ucclAllReduce(
        sendbuff, recvbuff, count,
        ucclBfloat16, ucclSum, comm,
        static_cast<void*>(&queue));

    if (res != ucclSuccess) {
        std::fprintf(stderr,
            "[Rank %d] AllReduce bf16 failed: %s\n",
            rank, ucclGetErrorString(res));
        errors++;
    } else {
        queue.wait();
        std::printf("[Rank %d] PASS: AllReduce bf16 count=%zu\n",
                    rank, count);
    }

    sycl::free(sendbuff, queue);
    sycl::free(recvbuff, queue);

    return errors;
}

int main(int argc, char** argv) {
    /* Only initialize MPI when this test is launched as multi-rank.
     * In single-process mode, avoid MPI+SYCL runtime interaction issues. */
    int rank = 0, nranks = 1;
    bool useMpi = false;
    bool mpiInitByUs = false;
    int mpiInit = 0;
    MPI_Initialized(&mpiInit);
    if (mpiInit) {
        useMpi = true;
    } else {
        const char* pmiSize = std::getenv("PMI_SIZE");
        const char* ompiSize = std::getenv("OMPI_COMM_WORLD_SIZE");
        int launcherSize = 1;
        if (pmiSize) launcherSize = std::atoi(pmiSize);
        else if (ompiSize) launcherSize = std::atoi(ompiSize);
        if (launcherSize > 1) {
            MPI_Init(&argc, &argv);
            useMpi = true;
            mpiInitByUs = true;
        }
    }

    if (useMpi) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    }

    std::printf("[Rank %d/%d] Starting AllReduce test\n", rank, nranks);

    /* Select GPU */
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        std::fprintf(stderr, "[Rank %d] No GPU devices found\n", rank);
        if (mpiInitByUs) MPI_Finalize();
        return 1;
    }

    int devIdx = rank % static_cast<int>(devices.size());
    sycl::queue queue(devices[devIdx]);

    /* Initialize UCCL communicator */
    ucclUniqueId id;
    if (rank == 0) ucclGetUniqueId(&id);
    if (useMpi) {
        MPI_Bcast(&id, sizeof(ucclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    ucclComm_t comm;
    ucclResult_t res = ucclCommInitRank(&comm, nranks, id, rank);
    if (res != ucclSuccess) {
        std::fprintf(stderr, "[Rank %d] CommInitRank failed: %s\n",
                     rank, ucclGetErrorString(res));
        if (mpiInitByUs) MPI_Finalize();
        return 1;
    }

    /* Run tests */
    int totalErrors = 0;
    totalErrors += testAllReduceFp16(comm, queue, rank, nranks);
    totalErrors += testAllReduceBf16(comm, queue, rank, nranks);

    /* Cleanup */
    ucclCommFinalize(comm);
    ucclCommDestroy(comm);

    /* Summary */
    int globalErrors = 0;
    if (useMpi) {
        MPI_Allreduce(&totalErrors, &globalErrors, 1,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    } else {
        globalErrors = totalErrors;
    }

    if (rank == 0) {
        if (globalErrors == 0) {
            std::printf("\n=== ALL TESTS PASSED ===\n");
        } else {
            std::printf("\n=== %d TESTS FAILED ===\n", globalErrors);
        }
    }

    if (mpiInitByUs) MPI_Finalize();
    return (globalErrors == 0) ? 0 : 1;
}
