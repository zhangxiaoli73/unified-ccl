#pragma once

#include "../include/uccl_common.h"

/* Protocol types */
enum ucclProtocol_t {
    UCCL_PROTO_SIMPLE = 0,
    UCCL_PROTO_LL128  = 1,
    /* UCCL_PROTO_LL  = 2, */  /* future extension */
    UCCL_NUM_PROTOCOLS
};

/* Protocol configuration constants — mirrors NCCL */

struct ProtoSimple {
    static constexpr int Id = UCCL_PROTO_SIMPLE;
    static constexpr int SlicePerChunk = 2;
    static constexpr int StepPerSlice = 4;
    /* Buffer size: large enough for pipelined transfers */
    static constexpr size_t DefaultBuffSize = 1 << 22; /* 4 MB */
};

struct ProtoLL128 {
    static constexpr int Id = UCCL_PROTO_LL128;
    /* 128 bytes = 16 x uint64_t per line
     * 14 data words + 2 flag words */
    static constexpr int LineElems = 16;   /* NCCL_LL128_LINEELEMS */
    static constexpr int DataElems = 14;   /* NCCL_LL128_DATAELEMS */
    static constexpr int FlagElems = 2;    /* 2 flag words per line */
    static constexpr int LineSize = 128;   /* bytes per line */
    /* Buffer size: smaller than Simple, optimized for low latency */
    static constexpr size_t DefaultBuffSize = 1 << 20; /* 1 MB */
};
