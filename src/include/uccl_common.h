#ifndef UCCL_COMMON_H_
#define UCCL_COMMON_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Version
 * ============================================================ */
#define UCCL_MAJOR 0
#define UCCL_MINOR 1
#define UCCL_PATCH 0
#define UCCL_VERSION (UCCL_MAJOR * 10000 + UCCL_MINOR * 100 + UCCL_PATCH)

/* ============================================================
 * Error codes
 * ============================================================ */
typedef enum {
    ucclSuccess             = 0,
    ucclSystemError          = 1,
    ucclInternalError        = 2,
    ucclInvalidArgument      = 3,
    ucclInvalidUsage         = 4,
    ucclRemoteError          = 5,
    ucclInProgress           = 6,
    ucclNumResults           = 7
} ucclResult_t;

/* ============================================================
 * Data types — MVP: fp16, bf16
 * ============================================================ */
typedef enum {
    ucclFloat16    = 0,
    ucclBfloat16   = 1,
    ucclNumTypes   = 2
} ucclDataType_t;

/* ============================================================
 * Reduction operations — MVP: sum only
 * ============================================================ */
typedef enum {
    ucclSum   = 0,
    ucclNumOps = 1
} ucclRedOp_t;

/* ============================================================
 * Constants
 * ============================================================ */
#define UCCL_UNIQUE_ID_BYTES 128
#define UCCL_MAX_CHANNELS    16
#define UCCL_MAX_RANKS       128
#define UCCL_STEPS           8

/* ============================================================
 * Pointer support flags (for network plugin)
 * ============================================================ */
#define UCCL_PTR_HOST   0x1
#define UCCL_PTR_DEVICE 0x2  /* GPU Direct RDMA support */

#ifdef __cplusplus
}
#endif

#endif /* UCCL_COMMON_H_ */
