#include "../include/uccl_common.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

/* General utility functions */

namespace uccl {

/* Compute hash of hostname for node identification */
uint64_t getHostHash() {
    char hostname[256];
    std::memset(hostname, 0, sizeof(hostname));

#ifdef __linux__
    gethostname(hostname, sizeof(hostname) - 1);
#else
    std::strncpy(hostname, "localhost", sizeof(hostname) - 1);
#endif

    return std::hash<std::string>{}(std::string(hostname));
}

/* Compute hash of process ID */
uint64_t getPidHash() {
#ifdef __linux__
    return static_cast<uint64_t>(getpid());
#else
    return 0;
#endif
}

/* Get data type size in bytes */
size_t ucclDataTypeSize(ucclDataType_t type) {
    switch (type) {
        case ucclFloat16:  return 2;
        case ucclBfloat16: return 2;
        default:           return 0;
    }
}

/* Align size up to given alignment */
size_t alignUp(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

} /* namespace uccl */
