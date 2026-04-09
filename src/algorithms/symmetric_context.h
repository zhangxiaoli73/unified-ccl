#pragma once

#include <cstddef>

namespace uccl {

/* Shared symmetric-memory context used by symmetric/copy-engine algorithms. */
struct SymmetricMemoryContext {
    int nGpus;
    void** remoteBuffs;     /* remoteBuffs[i] = GPU i's buffer mapped locally */
    size_t buffSize;
    void* localBuff;
};

} /* namespace uccl */
