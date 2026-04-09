#pragma once

#include "../include/uccl_common.h"

/* Algorithm types */
enum ucclAlgorithm_t {
    UCCL_ALGO_RING = 0,
    UCCL_ALGO_ONE_SHOT = 1,
    UCCL_ALGO_SYMMETRIC_MEM = 2,
    UCCL_ALGO_COPY_ENGINE = 3,     /* Copy engine for data movement, EU for reduce */
    /* UCCL_ALGO_TREE = 4, */      /* future extension */
    /* UCCL_ALGO_COLLNET = 5, */   /* future extension */
    UCCL_NUM_ALGORITHMS
};

/* Collective types */
enum ucclCollective_t {
    UCCL_COLL_ALLREDUCE = 0,
    UCCL_COLL_ALLGATHER = 1,
    UCCL_COLL_REDUCESCATTER = 2,
    UCCL_NUM_COLLECTIVES
};
