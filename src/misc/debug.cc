#include "debug.h"
#include "../include/uccl_common.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

/* Debug and logging implementation */

namespace uccl {

/* Initialize debug subsystem.
 * Called once during library initialization. */
void debugInit() {
    ucclDebugLevel level = getDebugLevel();
    if (level >= UCCL_LOG_INFO) {
        std::fprintf(stderr, "[UCCL INFO] Unified-CCL v%d.%d.%d initialized "
                     "(debug level: %d)\n",
                     UCCL_MAJOR, UCCL_MINOR, UCCL_PATCH,
                     static_cast<int>(level));
    }
}

} /* namespace uccl */
