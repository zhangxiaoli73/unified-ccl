#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

/* Debug / Logging utilities for Unified-CCL.
 *
 * Log levels controlled by UCCL_DEBUG environment variable:
 *   UCCL_DEBUG=INFO    — informational messages
 *   UCCL_DEBUG=WARN    — warnings
 *   UCCL_DEBUG=ERROR   — errors only (default)
 *   UCCL_DEBUG=TRACE   — verbose tracing
 */

namespace uccl {

enum ucclDebugLevel {
    UCCL_LOG_NONE  = 0,
    UCCL_LOG_ERROR = 1,
    UCCL_LOG_WARN  = 2,
    UCCL_LOG_INFO  = 3,
    UCCL_LOG_TRACE = 4
};

/* Get current debug level from environment */
inline ucclDebugLevel getDebugLevel() {
    static ucclDebugLevel level = []() {
        const char* env = std::getenv("UCCL_DEBUG");
        if (env == nullptr) return UCCL_LOG_ERROR;
        if (std::string(env) == "TRACE") return UCCL_LOG_TRACE;
        if (std::string(env) == "INFO") return UCCL_LOG_INFO;
        if (std::string(env) == "WARN") return UCCL_LOG_WARN;
        if (std::string(env) == "ERROR") return UCCL_LOG_ERROR;
        if (std::string(env) == "NONE") return UCCL_LOG_NONE;
        return UCCL_LOG_ERROR;
    }();
    return level;
}

} /* namespace uccl */

/* Logging macro */
#define UCCL_LOG(level, fmt, ...)                                       \
    do {                                                                 \
        if (uccl::getDebugLevel() >= uccl::UCCL_LOG_##level) {          \
            std::fprintf(stderr, "[UCCL %s] %s:%d " fmt "\n",           \
                         #level, __FILE__, __LINE__, ##__VA_ARGS__);     \
        }                                                                \
    } while (0)
