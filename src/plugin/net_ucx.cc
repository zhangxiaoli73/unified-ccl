#include "net_plugin.h"
#include "../misc/debug.h"

#include <cstdlib>
#include <cstring>

#ifdef __linux__
#include <dlfcn.h>
#endif

/* Built-in UCX network plugin implementation.
 *
 * This provides a basic UCX plugin compiled directly into libuccl.so.
 * For a full-featured plugin, see plugins/ucx/plugin.cc which can
 * be compiled as a separate libuccl-net-ucx.so.
 *
 * UCX (Unified Communication X) provides:
 * - RDMA support via multiple transports (InfiniBand, RoCE, etc.)
 * - GPU Direct RDMA for zero-copy GPU-GPU transfers across nodes
 * - Automatic transport selection based on available hardware */

namespace uccl {

/* ============================================================
 * Built-in UCX Plugin Stubs
 *
 * These are placeholder implementations. A full UCX plugin would
 * use libucp API calls (ucp_init, ucp_ep_create, etc.).
 * ============================================================ */

static ucclResult_t ucxInit(void** handle) {
    UCCL_LOG(INFO, "UCX plugin: init");
    *handle = nullptr;

    /* In full implementation:
     *   ucp_params_t params = {};
     *   params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA;
     *   ucp_init(&params, NULL, &ucp_context);
     *   *handle = ucp_context; */

    return ucclSuccess;
}

static ucclResult_t ucxFinalize(void* handle) {
    UCCL_LOG(INFO, "UCX plugin: finalize");
    (void)handle;

    /* In full implementation:
     *   ucp_cleanup(handle); */

    return ucclSuccess;
}

static ucclResult_t ucxDevices(int* ndev) {
    /* Report number of available network devices */
    *ndev = 1; /* placeholder: at least one loopback */
    return ucclSuccess;
}

static ucclResult_t ucxGetProperties(int dev,
                                     ucclNetProperties_t* props) {
    if (props == nullptr) return ucclInvalidArgument;

    std::memset(props, 0, sizeof(*props));
    snprintf(props->name, sizeof(props->name), "ucx-%d", dev);
    props->ptrSupport = UCCL_PTR_HOST | UCCL_PTR_DEVICE;
    props->speed = 25000;     /* 25 Gbps typical */
    props->latency = 1000;    /* 1 us typical */
    props->maxComms = 65536;

    return ucclSuccess;
}

static ucclResult_t ucxListen(void* handle, void* listenAddr,
                              void** listenComm) {
    (void)handle;
    (void)listenAddr;
    *listenComm = nullptr;
    UCCL_LOG(INFO, "UCX plugin: listen");
    return ucclSuccess;
}

static ucclResult_t ucxConnect(void* handle, void* connectAddr,
                               void** sendComm) {
    (void)handle;
    (void)connectAddr;
    *sendComm = nullptr;
    UCCL_LOG(INFO, "UCX plugin: connect");

    /* In full implementation:
     *   ucp_ep_params_t ep_params = {};
     *   ucp_ep_create(worker, &ep_params, &ep);
     *   *sendComm = ep; */

    return ucclSuccess;
}

static ucclResult_t ucxAccept(void* listenComm, void** recvComm) {
    (void)listenComm;
    *recvComm = nullptr;
    UCCL_LOG(INFO, "UCX plugin: accept");
    return ucclSuccess;
}

static ucclResult_t ucxRegMr(void* comm, void* data, size_t size,
                             int type, void** mhandle) {
    (void)comm;
    (void)data;
    (void)size;
    (void)type;
    *mhandle = nullptr;

    /* In full implementation:
     *   ucp_mem_map_params_t params = {};
     *   params.address = data;
     *   params.length = size;
     *   ucp_mem_map(context, &params, mhandle); */

    return ucclSuccess;
}

static ucclResult_t ucxDeregMr(void* comm, void* mhandle) {
    (void)comm;
    (void)mhandle;

    /* In full implementation:
     *   ucp_mem_unmap(context, mhandle); */

    return ucclSuccess;
}

static ucclResult_t ucxIsend(void* sendComm, void* data, size_t size,
                             void* mhandle, void** request) {
    (void)sendComm;
    (void)data;
    (void)size;
    (void)mhandle;
    *request = nullptr;

    /* In full implementation:
     *   ucp_tag_send_nbx(ep, data, size, tag, &params); */

    return ucclSuccess;
}

static ucclResult_t ucxIrecv(void* recvComm, void* data, size_t size,
                             void* mhandle, void** request) {
    (void)recvComm;
    (void)data;
    (void)size;
    (void)mhandle;
    *request = nullptr;

    /* In full implementation:
     *   ucp_tag_recv_nbx(worker, data, size, tag, mask, &params); */

    return ucclSuccess;
}

static ucclResult_t ucxTest(void* request, int* done, int* size) {
    (void)request;
    *done = 1; /* placeholder: always done */
    if (size) *size = 0;

    /* In full implementation:
     *   ucs_status_t status = ucp_request_check_status(request); */

    return ucclSuccess;
}

static ucclResult_t ucxCloseSend(void* sendComm) {
    (void)sendComm;
    return ucclSuccess;
}

static ucclResult_t ucxCloseRecv(void* recvComm) {
    (void)recvComm;
    return ucclSuccess;
}

static ucclResult_t ucxCloseListen(void* listenComm) {
    (void)listenComm;
    return ucclSuccess;
}

/* Built-in UCX plugin instance */
static ucclNet_t builtinUcxPlugin = {
    "UCX",          /* name */
    ucxInit,
    ucxFinalize,
    ucxDevices,
    ucxGetProperties,
    ucxListen,
    ucxConnect,
    ucxAccept,
    ucxRegMr,
    ucxDeregMr,
    ucxIsend,
    ucxIrecv,
    ucxTest,
    ucxCloseSend,
    ucxCloseRecv,
    ucxCloseListen
};

ucclNet_t* getBuiltinUcxPlugin() {
    return &builtinUcxPlugin;
}

/* ============================================================
 * Dynamic plugin loading
 * ============================================================ */

ucclResult_t loadNetPlugin(const std::string& path, ucclNet_t** plugin) {
    if (plugin == nullptr) return ucclInvalidArgument;

#ifdef __linux__
    /* Validate plugin path: reject empty, relative traversal, and non-.so files */
    if (path.empty()) {
        UCCL_LOG(ERROR, "Plugin path is empty");
        return ucclInvalidArgument;
    }
    if (path.find("..") != std::string::npos) {
        UCCL_LOG(ERROR, "Plugin path contains directory traversal: %s",
                 path.c_str());
        return ucclInvalidArgument;
    }
    if (path.size() < 3 ||
        path.compare(path.size() - 3, 3, ".so") != 0) {
        UCCL_LOG(ERROR, "Plugin path must end with .so: %s", path.c_str());
        return ucclInvalidArgument;
    }

    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        UCCL_LOG(ERROR, "Failed to load plugin %s: %s",
                 path.c_str(), dlerror());
        return ucclSystemError;
    }

    /* Look for the plugin symbol */
    auto* p = static_cast<ucclNet_t*>(
        dlsym(handle, "ucclNetPlugin_v1"));
    if (p == nullptr) {
        UCCL_LOG(ERROR, "Plugin %s: symbol ucclNetPlugin_v1 not found: %s",
                 path.c_str(), dlerror());
        dlclose(handle);
        return ucclInternalError;
    }

    *plugin = p;
    UCCL_LOG(INFO, "Loaded network plugin: %s (%s)", p->name, path.c_str());
    return ucclSuccess;
#else
    (void)path;
    UCCL_LOG(ERROR, "Dynamic plugin loading not supported on this platform");
    return ucclSystemError;
#endif
}

ucclResult_t autoLoadNetPlugin(ucclNet_t** plugin) {
    if (plugin == nullptr) return ucclInvalidArgument;

    /* 1. Check UCCL_NET_PLUGIN environment variable */
    const char* envPlugin = std::getenv("UCCL_NET_PLUGIN");
    if (envPlugin != nullptr && envPlugin[0] != '\0') {
        UCCL_LOG(INFO, "Loading plugin from UCCL_NET_PLUGIN=%s", envPlugin);
        return loadNetPlugin(envPlugin, plugin);
    }

    /* 2. Use built-in UCX plugin */
    *plugin = getBuiltinUcxPlugin();
    UCCL_LOG(INFO, "Using built-in UCX plugin");
    return ucclSuccess;
}

ucclResult_t unloadNetPlugin(ucclNet_t* plugin) {
    /* Built-in plugin: nothing to unload */
    if (plugin == &builtinUcxPlugin) {
        return ucclSuccess;
    }

    /* Dynamic plugin: would need to dlclose the handle.
     * In full implementation, we'd track the dlopen handle. */
    return ucclSuccess;
}

} /* namespace uccl */
