#ifndef UCCL_H_
#define UCCL_H_

#include "uccl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Types
 * ============================================================ */

/* Opaque communicator handle */
typedef struct ucclComm* ucclComm_t;
#define UCCL_COMM_NULL NULL

/* Unique ID for communicator bootstrap */
typedef struct { char internal[UCCL_UNIQUE_ID_BYTES]; } ucclUniqueId;

/* ============================================================
 * Communicator Lifecycle
 * ============================================================ */

ucclResult_t ucclGetVersion(int* version);
ucclResult_t ucclGetUniqueId(ucclUniqueId* uniqueId);

ucclResult_t ucclCommInitRank(ucclComm_t* comm, int nranks,
                              ucclUniqueId commId, int rank);
ucclResult_t ucclCommInitAll(ucclComm_t* comm, int ndev,
                             const int* devlist);
ucclResult_t ucclCommFinalize(ucclComm_t comm);
ucclResult_t ucclCommDestroy(ucclComm_t comm);
ucclResult_t ucclCommAbort(ucclComm_t comm);
ucclResult_t ucclCommCount(const ucclComm_t comm, int* count);
ucclResult_t ucclCommUserRank(const ucclComm_t comm, int* rank);

/* ============================================================
 * Collective Operations
 * ============================================================ */

ucclResult_t ucclAllReduce(const void* sendbuff, void* recvbuff,
                           size_t count, ucclDataType_t datatype,
                           ucclRedOp_t op, ucclComm_t comm,
                           void* stream);
/* stream: sycl::queue*, use void* for C compatibility */

ucclResult_t ucclAllGather(const void* sendbuff, void* recvbuff,
                           size_t sendcount, ucclDataType_t datatype,
                           ucclComm_t comm, void* stream);
/* recvbuff must hold sendcount * nranks elements */

ucclResult_t ucclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, ucclDataType_t datatype,
                               ucclRedOp_t op, ucclComm_t comm,
                               void* stream);
/* sendbuff must hold recvcount * nranks elements */

/* ============================================================
 * Group Semantics
 * ============================================================ */

ucclResult_t ucclGroupStart(void);
ucclResult_t ucclGroupEnd(void);

/* ============================================================
 * Memory Window Registration (for RMA operations)
 * ============================================================ */

typedef struct ucclWindow* ucclWindow_t;

ucclResult_t ucclCommWindowRegister(ucclWindow_t* win, void* buff,
                                    size_t size, ucclComm_t comm);
ucclResult_t ucclCommWindowDeregister(ucclWindow_t win);

/* ============================================================
 * One-Sided Point-to-Point Operations (RMA)
 * ============================================================ */

ucclResult_t ucclPut(const void* localBuff, size_t count,
                     ucclDataType_t datatype, int peer,
                     ucclWindow_t peerWin, size_t peerWinOffset,
                     ucclComm_t comm, void* stream);

ucclResult_t ucclGet(void* localBuff, size_t count,
                     ucclDataType_t datatype, int peer,
                     ucclWindow_t peerWin, size_t peerWinOffset,
                     ucclComm_t comm, void* stream);

ucclResult_t ucclPutSignal(const void* localBuff, size_t count,
                           ucclDataType_t datatype, int peer,
                           ucclWindow_t peerWin, size_t peerWinOffset,
                           int sigIdx, int ctx, unsigned int flags,
                           ucclComm_t comm, void* stream);

ucclResult_t ucclGetSignal(void* localBuff, size_t count,
                           ucclDataType_t datatype, int peer,
                           ucclWindow_t peerWin, size_t peerWinOffset,
                           int sigIdx, int ctx, unsigned int flags,
                           ucclComm_t comm, void* stream);

ucclResult_t ucclSignal(int peer, int sigIdx, int ctx,
                        unsigned int flags,
                        ucclComm_t comm, void* stream);

typedef struct {
    int opCnt;
    int peer;
    int sigIdx;
    int ctx;
} ucclWaitSignalDesc_t;

ucclResult_t ucclWaitSignal(int nDesc,
                            ucclWaitSignalDesc_t* signalDescs,
                            ucclComm_t comm, void* stream);

/* ============================================================
 * Device-Side RMA API (GPU kernel callable)
 *
 * For GPU device functions (devicePut, deviceGet, devicePutSignal,
 * deviceGetSignal, deviceSignal, deviceWaitSignal), include:
 *   #include "rma/rma_device.hpp"
 *
 * Host-side: convert ucclWindow_t → kernel-capturable handle:
 * ============================================================ */

/* Convert host window handle to device-capturable POD descriptor.
 * The returned ucclDeviceWindow can be captured by value in SYCL
 * kernel lambdas and used with device-side RMA functions. */
/* ucclResult_t ucclWindowGetDeviceHandle(ucclWindow_t win,
 *                                       ucclDeviceWindow* deviceWin);
 * (declared in rma/rma.h with full type info) */

/* ============================================================
 * Error Reporting
 * ============================================================ */

const char* ucclGetErrorString(ucclResult_t result);
const char* ucclGetLastError(ucclComm_t comm);

#ifdef __cplusplus
}
#endif

#endif /* UCCL_H_ */
