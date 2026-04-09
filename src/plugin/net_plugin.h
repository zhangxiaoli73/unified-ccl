#pragma once

#include "../include/uccl_common.h"
#include "../include/uccl_net.h"

#include <string>

namespace uccl {

/* Network Plugin Loader
 *
 * Loads network plugins either:
 * 1. Built-in (e.g., UCX plugin compiled into the library)
 * 2. External (dynamically loaded .so via dlopen)
 *
 * Plugin discovery order:
 * 1. Check UCCL_NET_PLUGIN environment variable
 * 2. Try built-in UCX plugin
 * 3. Search for libuccl-net-*.so in library path */

/* Get built-in UCX plugin (if compiled in) */
ucclNet_t* getBuiltinUcxPlugin();

/* Load external network plugin from shared library */
ucclResult_t loadNetPlugin(const std::string& path, ucclNet_t** plugin);

/* Auto-detect and load the best available network plugin */
ucclResult_t autoLoadNetPlugin(ucclNet_t** plugin);

/* Unload a dynamically loaded plugin */
ucclResult_t unloadNetPlugin(ucclNet_t* plugin);

} /* namespace uccl */
