#include "include/uccl.h"
#include "include/comm.h"
#include "misc/debug.h"

#include <sycl/sycl.hpp>

/* Collective operations dispatch.
 * Mirrors NCCL src/collectives.cc.
 *
 * Entry point for all collective operations.
 * Validates arguments and dispatches to enqueue.cc. */

/* Forward declarations */
namespace uccl {
    ucclResult_t enqueueAllReduce(const void* sendbuff, void* recvbuff,
                                  size_t count, ucclDataType_t datatype,
                                  ucclRedOp_t op, ucclComm* comm,
                                  sycl::queue* stream);
    ucclResult_t enqueueAllGather(const void* sendbuff, void* recvbuff,
                                  size_t sendcount, ucclDataType_t datatype,
                                  ucclComm* comm, sycl::queue* stream);
    ucclResult_t enqueueReduceScatter(const void* sendbuff, void* recvbuff,
                                      size_t recvcount, ucclDataType_t datatype,
                                      ucclRedOp_t op, ucclComm* comm,
                                      sycl::queue* stream);
}

/* Shared validation for all collectives */
static ucclResult_t validateComm(ucclComm_t comm) {
    if (comm == nullptr) return ucclInvalidArgument;
    if (comm->asyncError != ucclSuccess) return comm->asyncError;
    if (comm->abortFlag) return ucclInternalError;
    return ucclSuccess;
}

static sycl::queue* resolveQueue(ucclComm_t comm, void* stream) {
    return (stream != nullptr)
        ? static_cast<sycl::queue*>(stream)
        : comm->defaultQueue;
}

ucclResult_t ucclAllReduce(const void* sendbuff, void* recvbuff,
                           size_t count, ucclDataType_t datatype,
                           ucclRedOp_t op, ucclComm_t comm,
                           void* stream) {
    ucclResult_t valid = validateComm(comm);
    if (valid != ucclSuccess) return valid;
    if (count == 0) return ucclSuccess;
    if (sendbuff == nullptr || recvbuff == nullptr) return ucclInvalidArgument;
    if (datatype >= ucclNumTypes) {
        UCCL_LOG(ERROR, "AllReduce: unsupported datatype %d", datatype);
        return ucclInvalidArgument;
    }
    if (op >= ucclNumOps) {
        UCCL_LOG(ERROR, "AllReduce: unsupported reduction op %d", op);
        return ucclInvalidArgument;
    }

    sycl::queue* q = resolveQueue(comm, stream);
    if (q == nullptr) {
        UCCL_LOG(ERROR, "AllReduce: no SYCL queue available");
        return ucclInternalError;
    }

    UCCL_LOG(TRACE, "AllReduce: rank=%d, count=%zu, dtype=%d, op=%d",
             comm->rank, count, datatype, op);

    return uccl::enqueueAllReduce(sendbuff, recvbuff, count,
                                  datatype, op, comm, q);
}

ucclResult_t ucclAllGather(const void* sendbuff, void* recvbuff,
                           size_t sendcount, ucclDataType_t datatype,
                           ucclComm_t comm, void* stream) {
    ucclResult_t valid = validateComm(comm);
    if (valid != ucclSuccess) return valid;
    if (sendcount == 0) return ucclSuccess;
    if (sendbuff == nullptr || recvbuff == nullptr) return ucclInvalidArgument;
    if (datatype >= ucclNumTypes) {
        UCCL_LOG(ERROR, "AllGather: unsupported datatype %d", datatype);
        return ucclInvalidArgument;
    }

    sycl::queue* q = resolveQueue(comm, stream);
    if (q == nullptr) {
        UCCL_LOG(ERROR, "AllGather: no SYCL queue available");
        return ucclInternalError;
    }

    UCCL_LOG(TRACE, "AllGather: rank=%d, sendcount=%zu, dtype=%d",
             comm->rank, sendcount, datatype);

    return uccl::enqueueAllGather(sendbuff, recvbuff, sendcount,
                                  datatype, comm, q);
}

ucclResult_t ucclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, ucclDataType_t datatype,
                               ucclRedOp_t op, ucclComm_t comm,
                               void* stream) {
    ucclResult_t valid = validateComm(comm);
    if (valid != ucclSuccess) return valid;
    if (recvcount == 0) return ucclSuccess;
    if (sendbuff == nullptr || recvbuff == nullptr) return ucclInvalidArgument;
    if (datatype >= ucclNumTypes) {
        UCCL_LOG(ERROR, "ReduceScatter: unsupported datatype %d", datatype);
        return ucclInvalidArgument;
    }
    if (op >= ucclNumOps) {
        UCCL_LOG(ERROR, "ReduceScatter: unsupported reduction op %d", op);
        return ucclInvalidArgument;
    }

    sycl::queue* q = resolveQueue(comm, stream);
    if (q == nullptr) {
        UCCL_LOG(ERROR, "ReduceScatter: no SYCL queue available");
        return ucclInternalError;
    }

    UCCL_LOG(TRACE, "ReduceScatter: rank=%d, recvcount=%zu, dtype=%d, op=%d",
             comm->rank, recvcount, datatype, op);

    return uccl::enqueueReduceScatter(sendbuff, recvbuff, recvcount,
                                      datatype, op, comm, q);
}
