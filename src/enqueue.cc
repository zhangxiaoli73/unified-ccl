#include "include/uccl.h"
#include "include/comm.h"
#include "include/channel.h"
#include "include/hw_resources.h"
#include "algorithms/algorithm.h"
#include "protocols/protocol.h"
#include "topo/topo.h"
#include "device/primitives.hpp"
#include "device/reduce_kernel.hpp"
#include "misc/debug.h"

#include <sycl/sycl.hpp>
#include <cstdlib>
#include <cstring>

/* Task enqueue and algorithm/protocol selection.
 * Mirrors NCCL src/enqueue.cc.
 *
 * Responsibilities:
 * 1. Select algorithm (Ring) based on collective type
 * 2. Select protocol (Simple/LL128) based on message size and topology
 * 3. Split data across channels
 * 4. Launch SYCL kernels for each channel */

namespace uccl {

/* Forward declarations — AllReduce algorithms */
template <typename T>
ucclResult_t launchRingAllReduce(sycl::queue& queue,
                                 const T* sendbuff, T* recvbuff,
                                 size_t count, int nranks, int rank,
                                 ucclChannel& channel, int protocol);

template <typename T>
ucclResult_t launchOneShotAllReduce(sycl::queue& queue,
                                    const T* sendbuff, T* recvbuff,
                                    size_t count, int nranks, int rank,
                                    ucclChannel& channel, int protocol);

/* Forward declarations — AllGather / ReduceScatter */
template <typename T>
ucclResult_t launchRingAllGather(sycl::queue& queue,
                                 const T* sendbuff, T* recvbuff,
                                 size_t sendcount, int nranks, int rank,
                                 ucclChannel& channel, int protocol);

template <typename T>
ucclResult_t launchRingReduceScatter(sycl::queue& queue,
                                      const T* sendbuff, T* recvbuff,
                                      size_t recvcount, int nranks, int rank,
                                      ucclChannel& channel, int protocol);

/* Forward declaration — Symmetric Memory context and launchers */
struct SymmetricMemoryContext;

template <typename T>
ucclResult_t launchSymmetricAllReduce(sycl::queue& queue,
                                      const T* sendbuff, T* recvbuff,
                                      size_t count, int nGpus, int myGpu,
                                      const SymmetricMemoryContext& ctx);

template <typename T>
ucclResult_t launchSymmetricAllGather(sycl::queue& queue,
                                      const T* sendbuff, T* recvbuff,
                                      size_t sendcount, int nGpus, int myGpu,
                                      const SymmetricMemoryContext& ctx);

template <typename T>
ucclResult_t launchSymmetricReduceScatter(sycl::queue& queue,
                                           const T* sendbuff, T* recvbuff,
                                           size_t recvcount, int nGpus, int myGpu,
                                           const SymmetricMemoryContext& ctx);

/* Forward declarations — Copy Engine launchers */
template <typename T>
ucclResult_t launchCopyEngineAllReduce(sycl::queue& computeQueue,
                                       sycl::queue& copyQueue,
                                       const T* sendbuff, T* recvbuff,
                                       size_t count, int nGpus, int myGpu,
                                       const SymmetricMemoryContext& ctx);

template <typename T>
ucclResult_t launchCopyEngineAllGather(sycl::queue& copyQueue,
                                       const T* sendbuff, T* recvbuff,
                                       size_t sendcount, int nGpus, int myGpu,
                                       const SymmetricMemoryContext& ctx);

template <typename T>
ucclResult_t launchCopyEngineReduceScatter(sycl::queue& computeQueue,
                                            sycl::queue& copyQueue,
                                            const T* sendbuff, T* recvbuff,
                                            size_t recvcount, int nGpus, int myGpu,
                                            const SymmetricMemoryContext& ctx);

/* Check UCCL_ALGO environment variable for algorithm override.
 * Supported values: "ring", "one_shot", "symmetric", "copy_engine"
 * Returns -1 if not set or unrecognized. */
static int getAlgoEnvOverride() {
    static int cached = []() {
        const char* env = std::getenv("UCCL_ALGO");
        if (env == nullptr) return -1;
        if (std::strcmp(env, "ring") == 0) return (int)UCCL_ALGO_RING;
        if (std::strcmp(env, "one_shot") == 0) return (int)UCCL_ALGO_ONE_SHOT;
        if (std::strcmp(env, "symmetric") == 0) return (int)UCCL_ALGO_SYMMETRIC_MEM;
        if (std::strcmp(env, "copy_engine") == 0 ||
            std::strcmp(env, "ce") == 0) return (int)UCCL_ALGO_COPY_ENGINE;
        return -1;
    }();
    return cached;
}

/* Select algorithm and protocol for a given collective.
 * Uses topology-based tuning if available. */
static ucclResult_t selectAlgoProto(ucclComm* comm,
                                    size_t messageSize,
                                    ucclAlgorithm_t* algo,
                                    ucclProtocol_t* proto,
                                    int* nChannels) {
    /* Try topology-based tuning */
    if (comm->topo != nullptr) {
        ucclTopoTuning tuning;
        ucclResult_t res = ucclTopoTune(comm->topo, messageSize, &tuning);
        if (res == ucclSuccess) {
            *algo = tuning.algorithm;
            *proto = tuning.protocol;
            *nChannels = tuning.nChannels;
            return ucclSuccess;
        }
    }

    /* Fallback: default selection based on message size */
    if (messageSize < 4096) {
        *algo = UCCL_ALGO_ONE_SHOT;   /* One-Shot for small messages */
        *proto = UCCL_PROTO_LL128;
    } else if (messageSize > 524288) {
        *algo = UCCL_ALGO_RING;
        *proto = UCCL_PROTO_SIMPLE;
    } else {
        *algo = UCCL_ALGO_RING;
        *proto = UCCL_PROTO_SIMPLE;
    }

    *nChannels = (comm->nChannels > 0) ? comm->nChannels : 1;

    /* Check environment variable override */
    int envAlgo = getAlgoEnvOverride();
    if (envAlgo >= 0) {
        *algo = static_cast<ucclAlgorithm_t>(envAlgo);
        UCCL_LOG(INFO, "Algorithm overridden by UCCL_ALGO env: algo=%d", envAlgo);
    }

    /* Symmetric memory is only valid for single-node (intra-node) */
    if (*algo == UCCL_ALGO_SYMMETRIC_MEM && comm->nNodes > 1) {
        UCCL_LOG(WARN, "Symmetric memory algo requested but multi-node "
                 "detected (%d nodes), falling back to Ring", comm->nNodes);
        *algo = UCCL_ALGO_RING;
    }

    /* Copy engine mode also requires symmetric memory (intra-node P2P) */
    if (*algo == UCCL_ALGO_COPY_ENGINE && comm->nNodes > 1) {
        UCCL_LOG(WARN, "Copy engine algo requested but multi-node "
                 "detected, falling back to Ring");
        *algo = UCCL_ALGO_RING;
    }

    /* If hwResources exec mode is COPY_ENGINE, auto-select copy engine algo */
    if (comm->hwResources.execMode == UCCL_EXEC_COPY_ENGINE &&
        *algo != UCCL_ALGO_COPY_ENGINE && comm->nNodes == 1) {
        *algo = UCCL_ALGO_COPY_ENGINE;
        UCCL_LOG(INFO, "Execution mode COPY_ENGINE: switching to copy engine algo");
    }

    return ucclSuccess;
}

/* Enqueue AllReduce operation.
 *
 * This is the main dispatch function that:
 * 1. Determines message size to pick algorithm/protocol
 * 2. Splits data across channels
 * 3. Launches kernels for each channel */
ucclResult_t enqueueAllReduce(const void* sendbuff, void* recvbuff,
                               size_t count, ucclDataType_t datatype,
                               ucclRedOp_t op, ucclComm* comm,
                               sycl::queue* stream) {
    /* Calculate message size in bytes */
    size_t typeSize = (datatype == ucclFloat16) ? 2 : 2; /* bf16/fp16 = 2B */
    size_t messageSize = count * typeSize;

    /* Select algorithm, protocol, and channel count */
    ucclAlgorithm_t algo;
    ucclProtocol_t proto;
    int nChannels;
    selectAlgoProto(comm, messageSize, &algo, &proto, &nChannels);

    UCCL_LOG(INFO, "AllReduce enqueue: rank=%d, count=%zu, msgSize=%zu, "
             "algo=%d, proto=%d, nChannels=%d",
             comm->rank, count, messageSize, algo, proto, nChannels);

    /* For single rank: just copy sendbuff to recvbuff */
    if (comm->nRanks == 1) {
        if (sendbuff != recvbuff) {
            stream->memcpy(recvbuff, sendbuff, messageSize).wait();
        }
        return ucclSuccess;
    }

    /* Split data across channels */
    size_t countPerChannel = count / nChannels;
    size_t remainder = count % nChannels;

    /* Clamp to valid channel range */
    if (nChannels > comm->nChannels) nChannels = comm->nChannels;
    if (nChannels > UCCL_MAX_CHANNELS) nChannels = UCCL_MAX_CHANNELS;

    for (int ch = 0; ch < nChannels; ch++) {
        size_t chOffset = ch * countPerChannel;
        size_t chCount = countPerChannel;
        if (ch == nChannels - 1) {
            chCount += remainder; /* last channel gets remainder */
        }

        ucclChannel& channel = comm->channels[ch % UCCL_MAX_CHANNELS];

        /* Dispatch to type-specific kernel */
        ucclResult_t res;

        /* Copy engine mode: data movement via copy engine, reduce on EU.
         * Requires symmetricCtx for P2P mapped pointers and a copy queue. */
        if (algo == UCCL_ALGO_COPY_ENGINE && comm->symmetricCtx != nullptr &&
            comm->hwResources.copyEngine.copyQueue != nullptr) {
            sycl::queue& copyQ = *comm->hwResources.copyEngine.copyQueue;
            sycl::queue& compQ = *stream;
            switch (datatype) {
                case ucclFloat16: {
                    auto* send = static_cast<const sycl::half*>(sendbuff) + chOffset;
                    auto* recv = static_cast<sycl::half*>(recvbuff) + chOffset;
                    res = launchCopyEngineAllReduce<sycl::half>(
                        compQ, copyQ, send, recv, chCount,
                        comm->localRanks, comm->localRank,
                        *comm->symmetricCtx);
                    break;
                }
                case ucclBfloat16: {
                    using bf16 = sycl::ext::oneapi::bfloat16;
                    auto* send = static_cast<const bf16*>(sendbuff) + chOffset;
                    auto* recv = static_cast<bf16*>(recvbuff) + chOffset;
                    res = launchCopyEngineAllReduce<bf16>(
                        compQ, copyQ, send, recv, chCount,
                        comm->localRanks, comm->localRank,
                        *comm->symmetricCtx);
                    break;
                }
                default:
                    return ucclInvalidArgument;
            }
        }
        /* Symmetric memory: bypass channel split, use P2P read-reduce.
         * Requires comm->symmetricCtx to be initialized. */
        else if (algo == UCCL_ALGO_SYMMETRIC_MEM && comm->symmetricCtx != nullptr) {
            switch (datatype) {
                case ucclFloat16: {
                    auto* send = static_cast<const sycl::half*>(sendbuff) + chOffset;
                    auto* recv = static_cast<sycl::half*>(recvbuff) + chOffset;
                    res = launchSymmetricAllReduce<sycl::half>(
                        *stream, send, recv, chCount,
                        comm->localRanks, comm->localRank,
                        *comm->symmetricCtx);
                    break;
                }
                case ucclBfloat16: {
                    using bf16 = sycl::ext::oneapi::bfloat16;
                    auto* send = static_cast<const bf16*>(sendbuff) + chOffset;
                    auto* recv = static_cast<bf16*>(recvbuff) + chOffset;
                    res = launchSymmetricAllReduce<bf16>(
                        *stream, send, recv, chCount,
                        comm->localRanks, comm->localRank,
                        *comm->symmetricCtx);
                    break;
                }
                default:
                    return ucclInvalidArgument;
            }
        } else {

        switch (datatype) {
            case ucclFloat16: {
                auto* send = static_cast<const sycl::half*>(sendbuff) + chOffset;
                auto* recv = static_cast<sycl::half*>(recvbuff) + chOffset;
                if (algo == UCCL_ALGO_ONE_SHOT) {
                    res = launchOneShotAllReduce<sycl::half>(
                        *stream, send, recv, chCount,
                        comm->nRanks, comm->rank, channel, proto);
                } else {
                    res = launchRingAllReduce<sycl::half>(
                        *stream, send, recv, chCount,
                        comm->nRanks, comm->rank, channel, proto);
                }
                break;
            }
            case ucclBfloat16: {
                using bf16 = sycl::ext::oneapi::bfloat16;
                auto* send = static_cast<const bf16*>(sendbuff) + chOffset;
                auto* recv = static_cast<bf16*>(recvbuff) + chOffset;
                if (algo == UCCL_ALGO_ONE_SHOT) {
                    res = launchOneShotAllReduce<bf16>(
                        *stream, send, recv, chCount,
                        comm->nRanks, comm->rank, channel, proto);
                } else {
                    res = launchRingAllReduce<bf16>(
                        *stream, send, recv, chCount,
                        comm->nRanks, comm->rank, channel, proto);
                }
                break;
            }
            default:
                return ucclInvalidArgument;
        }

        } /* end else (non-symmetric) */

        if (res != ucclSuccess) {
            UCCL_LOG(ERROR, "AllReduce kernel launch failed on channel %d", ch);
            return res;
        }
    }

    (void)op; /* MVP: sum only */
    return ucclSuccess;
}

/* Enqueue AllGather operation.
 * Each rank sends sendcount elements; output has sendcount * nranks elements. */
ucclResult_t enqueueAllGather(const void* sendbuff, void* recvbuff,
                               size_t sendcount, ucclDataType_t datatype,
                               ucclComm* comm, sycl::queue* stream) {
    size_t typeSize = (datatype == ucclFloat16) ? 2 : 2;
    size_t messageSize = sendcount * typeSize;

    ucclAlgorithm_t algo;
    ucclProtocol_t proto;
    int nChannels;
    selectAlgoProto(comm, messageSize, &algo, &proto, &nChannels);

    UCCL_LOG(INFO, "AllGather enqueue: rank=%d, sendcount=%zu, algo=%d, proto=%d",
             comm->rank, sendcount, algo, proto);

    if (comm->nRanks == 1) {
        if (sendbuff != recvbuff) {
            stream->memcpy(recvbuff, sendbuff, messageSize).wait();
        }
        return ucclSuccess;
    }

    /* Copy engine path */
    if (algo == UCCL_ALGO_COPY_ENGINE && comm->symmetricCtx != nullptr &&
        comm->hwResources.copyEngine.copyQueue != nullptr) {
        sycl::queue& copyQ = *comm->hwResources.copyEngine.copyQueue;
        ucclResult_t res;
        switch (datatype) {
            case ucclFloat16:
                res = launchCopyEngineAllGather<sycl::half>(
                    copyQ,
                    static_cast<const sycl::half*>(sendbuff),
                    static_cast<sycl::half*>(recvbuff),
                    sendcount, comm->localRanks, comm->localRank,
                    *comm->symmetricCtx);
                break;
            case ucclBfloat16: {
                using bf16 = sycl::ext::oneapi::bfloat16;
                res = launchCopyEngineAllGather<bf16>(
                    copyQ,
                    static_cast<const bf16*>(sendbuff),
                    static_cast<bf16*>(recvbuff),
                    sendcount, comm->localRanks, comm->localRank,
                    *comm->symmetricCtx);
                break;
            }
            default:
                return ucclInvalidArgument;
        }
        return res;
    }

    /* Symmetric memory path */
    if (algo == UCCL_ALGO_SYMMETRIC_MEM && comm->symmetricCtx != nullptr) {
        ucclResult_t res;
        switch (datatype) {
            case ucclFloat16:
                res = launchSymmetricAllGather<sycl::half>(
                    *stream,
                    static_cast<const sycl::half*>(sendbuff),
                    static_cast<sycl::half*>(recvbuff),
                    sendcount, comm->localRanks, comm->localRank,
                    *comm->symmetricCtx);
                break;
            case ucclBfloat16: {
                using bf16 = sycl::ext::oneapi::bfloat16;
                res = launchSymmetricAllGather<bf16>(
                    *stream,
                    static_cast<const bf16*>(sendbuff),
                    static_cast<bf16*>(recvbuff),
                    sendcount, comm->localRanks, comm->localRank,
                    *comm->symmetricCtx);
                break;
            }
            default:
                return ucclInvalidArgument;
        }
        return res;
    }

    /* Ring path (default for AllGather) */
    ucclChannel& channel = comm->channels[0];

    ucclResult_t res;
    switch (datatype) {
        case ucclFloat16:
            res = launchRingAllGather<sycl::half>(
                *stream,
                static_cast<const sycl::half*>(sendbuff),
                static_cast<sycl::half*>(recvbuff),
                sendcount, comm->nRanks, comm->rank, channel, proto);
            break;
        case ucclBfloat16: {
            using bf16 = sycl::ext::oneapi::bfloat16;
            res = launchRingAllGather<bf16>(
                *stream,
                static_cast<const bf16*>(sendbuff),
                static_cast<bf16*>(recvbuff),
                sendcount, comm->nRanks, comm->rank, channel, proto);
            break;
        }
        default:
            return ucclInvalidArgument;
    }

    return res;
}

/* Enqueue ReduceScatter operation.
 * Input has recvcount * nranks elements; each rank gets recvcount elements. */
ucclResult_t enqueueReduceScatter(const void* sendbuff, void* recvbuff,
                                   size_t recvcount, ucclDataType_t datatype,
                                   ucclRedOp_t op, ucclComm* comm,
                                   sycl::queue* stream) {
    size_t typeSize = (datatype == ucclFloat16) ? 2 : 2;
    size_t messageSize = recvcount * comm->nRanks * typeSize;

    ucclAlgorithm_t algo;
    ucclProtocol_t proto;
    int nChannels;
    selectAlgoProto(comm, messageSize, &algo, &proto, &nChannels);

    UCCL_LOG(INFO, "ReduceScatter enqueue: rank=%d, recvcount=%zu, algo=%d, proto=%d",
             comm->rank, recvcount, algo, proto);

    if (comm->nRanks == 1) {
        if (sendbuff != recvbuff) {
            stream->memcpy(recvbuff, sendbuff, recvcount * typeSize).wait();
        }
        return ucclSuccess;
    }

    /* Copy engine path */
    if (algo == UCCL_ALGO_COPY_ENGINE && comm->symmetricCtx != nullptr &&
        comm->hwResources.copyEngine.copyQueue != nullptr) {
        sycl::queue& copyQ = *comm->hwResources.copyEngine.copyQueue;
        sycl::queue& compQ = *stream;
        ucclResult_t res;
        switch (datatype) {
            case ucclFloat16:
                res = launchCopyEngineReduceScatter<sycl::half>(
                    compQ, copyQ,
                    static_cast<const sycl::half*>(sendbuff),
                    static_cast<sycl::half*>(recvbuff),
                    recvcount, comm->localRanks, comm->localRank,
                    *comm->symmetricCtx);
                break;
            case ucclBfloat16: {
                using bf16 = sycl::ext::oneapi::bfloat16;
                res = launchCopyEngineReduceScatter<bf16>(
                    compQ, copyQ,
                    static_cast<const bf16*>(sendbuff),
                    static_cast<bf16*>(recvbuff),
                    recvcount, comm->localRanks, comm->localRank,
                    *comm->symmetricCtx);
                break;
            }
            default:
                return ucclInvalidArgument;
        }
        (void)op;
        return res;
    }

    /* Symmetric memory path */
    if (algo == UCCL_ALGO_SYMMETRIC_MEM && comm->symmetricCtx != nullptr) {
        ucclResult_t res;
        switch (datatype) {
            case ucclFloat16:
                res = launchSymmetricReduceScatter<sycl::half>(
                    *stream,
                    static_cast<const sycl::half*>(sendbuff),
                    static_cast<sycl::half*>(recvbuff),
                    recvcount, comm->localRanks, comm->localRank,
                    *comm->symmetricCtx);
                break;
            case ucclBfloat16: {
                using bf16 = sycl::ext::oneapi::bfloat16;
                res = launchSymmetricReduceScatter<bf16>(
                    *stream,
                    static_cast<const bf16*>(sendbuff),
                    static_cast<bf16*>(recvbuff),
                    recvcount, comm->localRanks, comm->localRank,
                    *comm->symmetricCtx);
                break;
            }
            default:
                return ucclInvalidArgument;
        }
        (void)op;
        return res;
    }

    /* Ring path (default for ReduceScatter) */
    ucclChannel& channel = comm->channels[0];

    ucclResult_t res;
    switch (datatype) {
        case ucclFloat16:
            res = launchRingReduceScatter<sycl::half>(
                *stream,
                static_cast<const sycl::half*>(sendbuff),
                static_cast<sycl::half*>(recvbuff),
                recvcount, comm->nRanks, comm->rank, channel, proto);
            break;
        case ucclBfloat16: {
            using bf16 = sycl::ext::oneapi::bfloat16;
            res = launchRingReduceScatter<bf16>(
                *stream,
                static_cast<const bf16*>(sendbuff),
                static_cast<bf16*>(recvbuff),
                recvcount, comm->nRanks, comm->rank, channel, proto);
            break;
        }
        default:
            return ucclInvalidArgument;
    }

    (void)op; /* MVP: sum only */
    return res;
}

} /* namespace uccl */
