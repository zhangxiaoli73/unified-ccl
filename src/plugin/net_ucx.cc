#include "net_plugin.h"
#include "../misc/debug.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <queue>

#ifdef __linux__
#include <dlfcn.h>
#endif

#ifdef UCCL_HAS_UCX
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>
#endif

/* Built-in UCX network plugin implementation.
 *
 * This provides a UCX plugin compiled directly into libuccl.so,
 * using libucp for RDMA transports (InfiniBand, RoCE, etc.)
 * and GPU Direct RDMA for zero-copy GPU-GPU transfers across nodes.
 *
 * When UCCL_HAS_UCX is not defined, stub implementations are used. */

namespace uccl {

#ifdef UCCL_HAS_UCX

/* ============================================================
 * Internal UCX context structures
 * ============================================================ */

struct ucxContext {
    ucp_context_h ucpContext;
    ucp_worker_h  ucpWorker;
};

/* Common base for send/recv comms — allows safe casting in regMr/setTag */
struct ucxCommBase {
    ucp_ep_h ep;
    ucxContext* ctx;
    uint64_t tag;       /* tag for message disambiguation */
};

struct ucxSendComm : ucxCommBase {};
struct ucxRecvComm : ucxCommBase {};

struct ucxListenComm {
    ucp_listener_h listener;
    ucxContext* ctx;
    /* Thread-safe queue for pending connection requests */
    std::mutex connMutex;
    std::queue<ucp_conn_request_h> connRequests;
};

struct ucxMemHandle {
    ucp_mem_h memh;
    ucp_context_h ucpContext;  /* needed for ucp_mem_unmap */
};

struct ucxRequest {
    int completed;
    size_t length;
};

/* ============================================================
 * UCP request callbacks
 * ============================================================ */

static void ucxSendCallback(void* request, ucs_status_t status,
                            void* /*user_data*/) {
    auto* req = static_cast<ucxRequest*>(request);
    if (req) req->completed = 1;
    if (status != UCS_OK) {
        UCCL_LOG(ERROR, "UCX send callback error: %s",
                 ucs_status_string(status));
    }
}

static void ucxRecvCallback(void* request, ucs_status_t status,
                            const ucp_tag_recv_info_t* info,
                            void* /*user_data*/) {
    auto* req = static_cast<ucxRequest*>(request);
    if (req) {
        req->completed = 1;
        req->length = info ? info->length : 0;
    }
    if (status != UCS_OK) {
        UCCL_LOG(ERROR, "UCX recv callback error: %s",
                 ucs_status_string(status));
    }
}

static void ucxListenConnCb(ucp_conn_request_h conn_request,
                            void* arg) {
    auto* lcomm = static_cast<ucxListenComm*>(arg);
    {
        std::lock_guard<std::mutex> lock(lcomm->connMutex);
        lcomm->connRequests.push(conn_request);
    }
    UCCL_LOG(INFO, "UCX plugin: incoming connection request (queue size=%zu)",
             lcomm->connRequests.size());
}

/* ============================================================
 * UCX plugin functions
 * ============================================================ */

static ucclResult_t ucxInit(void** handle) {
    UCCL_LOG(INFO, "UCX plugin: init (libucp)");

    auto* ctx = new ucxContext();

    /* Initialize UCP context */
    ucp_params_t params = {};
    params.field_mask = UCP_PARAM_FIELD_FEATURES |
                        UCP_PARAM_FIELD_REQUEST_SIZE |
                        UCP_PARAM_FIELD_REQUEST_INIT;
    params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA | UCP_FEATURE_STREAM;
    params.request_size = sizeof(ucxRequest);
    params.request_init = [](void* request) {
        auto* req = static_cast<ucxRequest*>(request);
        req->completed = 0;
        req->length = 0;
    };

    ucs_status_t st = ucp_init(&params, nullptr, &ctx->ucpContext);
    if (st != UCS_OK) {
        UCCL_LOG(ERROR, "ucp_init failed: %s", ucs_status_string(st));
        delete ctx;
        return ucclSystemError;
    }

    /* Create UCP worker */
    ucp_worker_params_t wparams = {};
    wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wparams.thread_mode = UCS_THREAD_MODE_MULTI;

    st = ucp_worker_create(ctx->ucpContext, &wparams, &ctx->ucpWorker);
    if (st != UCS_OK) {
        UCCL_LOG(ERROR, "ucp_worker_create failed: %s",
                 ucs_status_string(st));
        ucp_cleanup(ctx->ucpContext);
        delete ctx;
        return ucclSystemError;
    }

    *handle = ctx;
    UCCL_LOG(INFO, "UCX plugin: initialized successfully");
    return ucclSuccess;
}

static ucclResult_t ucxFinalize(void* handle) {
    UCCL_LOG(INFO, "UCX plugin: finalize");
    if (handle == nullptr) return ucclSuccess;

    auto* ctx = static_cast<ucxContext*>(handle);
    if (ctx->ucpWorker) ucp_worker_destroy(ctx->ucpWorker);
    if (ctx->ucpContext) ucp_cleanup(ctx->ucpContext);
    delete ctx;
    return ucclSuccess;
}

static ucclResult_t ucxDevices(int* ndev) {
    *ndev = 1;
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
    auto* ctx = static_cast<ucxContext*>(handle);
    auto* lcomm = new ucxListenComm();
    lcomm->ctx = ctx;

    ucp_listener_params_t params = {};
    params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                        UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    params.conn_handler.cb = ucxListenConnCb;
    params.conn_handler.arg = lcomm;

    if (listenAddr != nullptr) {
        params.sockaddr.addr =
            static_cast<const struct sockaddr*>(listenAddr);
        params.sockaddr.addrlen = sizeof(struct sockaddr_storage);
    }

    ucs_status_t st = ucp_listener_create(ctx->ucpWorker,
                                           &params, &lcomm->listener);
    if (st != UCS_OK) {
        UCCL_LOG(ERROR, "ucp_listener_create failed: %s",
                 ucs_status_string(st));
        delete lcomm;
        return ucclSystemError;
    }

    /* Write back the actual bound address so the caller can
     * send it to the peer for connect(). */
    if (listenAddr != nullptr) {
        ucp_listener_attr_t attr = {};
        attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
        st = ucp_listener_query(lcomm->listener, &attr);
        if (st == UCS_OK) {
            std::memcpy(listenAddr, &attr.sockaddr,
                        sizeof(struct sockaddr_storage));
        } else {
            UCCL_LOG(WARN, "ucp_listener_query failed: %s",
                     ucs_status_string(st));
        }
    }

    *listenComm = lcomm;
    UCCL_LOG(INFO, "UCX plugin: listening");
    return ucclSuccess;
}

static ucclResult_t ucxConnect(void* handle, void* connectAddr,
                               void** sendComm) {
    auto* ctx = static_cast<ucxContext*>(handle);
    auto* scomm = new ucxSendComm();
    scomm->ctx = ctx;
    scomm->tag = 0;  /* set later via setTag() */

    ucp_ep_params_t ep_params = {};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS;
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;

    if (connectAddr != nullptr) {
        ep_params.field_mask |= UCP_EP_PARAM_FIELD_SOCK_ADDR;
        ep_params.sockaddr.addr =
            static_cast<const struct sockaddr*>(connectAddr);
        ep_params.sockaddr.addrlen = sizeof(struct sockaddr_storage);
    }

    ucs_status_t st = ucp_ep_create(ctx->ucpWorker,
                                     &ep_params, &scomm->ep);
    if (st != UCS_OK) {
        UCCL_LOG(ERROR, "ucp_ep_create failed: %s",
                 ucs_status_string(st));
        delete scomm;
        return ucclSystemError;
    }

    *sendComm = scomm;
    UCCL_LOG(INFO, "UCX plugin: connected");
    return ucclSuccess;
}

static ucclResult_t ucxAccept(void* listenComm, void** recvComm) {
    auto* lcomm = static_cast<ucxListenComm*>(listenComm);
    auto* rcomm = new ucxRecvComm();
    rcomm->ctx = lcomm->ctx;
    rcomm->tag = 0;  /* set later via setTag() */

    ucp_conn_request_h connReq = nullptr;
    {
        std::lock_guard<std::mutex> lock(lcomm->connMutex);
        if (!lcomm->connRequests.empty()) {
            connReq = lcomm->connRequests.front();
            lcomm->connRequests.pop();
        }
    }

    if (connReq == nullptr) {
        /* Progress the worker to get a pending connection */
        ucp_worker_progress(lcomm->ctx->ucpWorker);
        {
            std::lock_guard<std::mutex> lock(lcomm->connMutex);
            if (!lcomm->connRequests.empty()) {
                connReq = lcomm->connRequests.front();
                lcomm->connRequests.pop();
            }
        }
        if (connReq == nullptr) {
            UCCL_LOG(WARN, "UCX plugin: no pending connection request");
            delete rcomm;
            return ucclInProgress;
        }
    }

    ucp_ep_params_t ep_params = {};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request = connReq;

    ucs_status_t st = ucp_ep_create(lcomm->ctx->ucpWorker,
                                     &ep_params, &rcomm->ep);
    if (st != UCS_OK) {
        UCCL_LOG(ERROR, "ucp_ep_create (accept) failed: %s",
                 ucs_status_string(st));
        delete rcomm;
        return ucclSystemError;
    }

    *recvComm = rcomm;
    UCCL_LOG(INFO, "UCX plugin: accepted connection");
    return ucclSuccess;
}

static ucclResult_t ucxRegMr(void* comm, void* data, size_t size,
                             int type, void** mhandle) {
    (void)type;
    if (comm == nullptr || data == nullptr || size == 0) {
        /* No-op for null registration */
        auto* mh = new ucxMemHandle();
        mh->memh = nullptr;
        mh->ucpContext = nullptr;
        *mhandle = mh;
        return ucclSuccess;
    }

    /* comm is ucxSendComm or ucxRecvComm — both derive from ucxCommBase */
    auto* ctx = static_cast<ucxCommBase*>(comm)->ctx;

    auto* mh = new ucxMemHandle();
    mh->ucpContext = ctx->ucpContext;

    ucp_mem_map_params_t params = {};
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address = data;
    params.length = size;

    ucs_status_t st = ucp_mem_map(ctx->ucpContext, &params, &mh->memh);
    if (st != UCS_OK) {
        UCCL_LOG(WARN, "ucp_mem_map failed: %s (falling back to no registration)",
                 ucs_status_string(st));
        mh->memh = nullptr;
    }

    *mhandle = mh;
    return ucclSuccess;
}

static ucclResult_t ucxDeregMr(void* comm, void* mhandle) {
    (void)comm;
    if (mhandle == nullptr) return ucclSuccess;
    auto* mh = static_cast<ucxMemHandle*>(mhandle);
    if (mh->memh != nullptr && mh->ucpContext != nullptr) {
        ucp_mem_unmap(mh->ucpContext, mh->memh);
    }
    delete mh;
    return ucclSuccess;
}

static ucclResult_t ucxIsend(void* sendComm, void* data, size_t size,
                             void* mhandle, void** request) {
    (void)mhandle;
    auto* scomm = static_cast<ucxSendComm*>(sendComm);

    ucp_request_param_t param = {};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_DATATYPE;
    param.cb.send = ucxSendCallback;
    param.datatype = ucp_dt_make_contig(1);

    ucs_status_ptr_t st = ucp_tag_send_nbx(scomm->ep, data, size,
                                            scomm->tag, &param);
    if (UCS_PTR_IS_ERR(st)) {
        UCCL_LOG(ERROR, "ucp_tag_send_nbx failed: %s",
                 ucs_status_string(UCS_PTR_STATUS(st)));
        return ucclSystemError;
    }

    if (st == NULL) {
        /* Completed immediately */
        *request = nullptr;
    } else {
        *request = st;
    }
    return ucclSuccess;
}

static ucclResult_t ucxIrecv(void* recvComm, void* data, size_t size,
                             void* mhandle, void** request) {
    (void)mhandle;
    auto* rcomm = static_cast<ucxRecvComm*>(recvComm);

    ucp_request_param_t param = {};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_DATATYPE;
    param.cb.recv = ucxRecvCallback;
    param.datatype = ucp_dt_make_contig(1);

    ucs_status_ptr_t st = ucp_tag_recv_nbx(rcomm->ctx->ucpWorker,
                                            data, size,
                                            rcomm->tag,
                                            UINT64_MAX /* exact match */,
                                            &param);
    if (UCS_PTR_IS_ERR(st)) {
        UCCL_LOG(ERROR, "ucp_tag_recv_nbx failed: %s",
                 ucs_status_string(UCS_PTR_STATUS(st)));
        return ucclSystemError;
    }

    if (st == NULL) {
        *request = nullptr;
    } else {
        *request = st;
    }
    return ucclSuccess;
}

static ucclResult_t ucxTest(void* request, int* done, int* size) {
    if (request == nullptr) {
        *done = 1;
        if (size) *size = 0;
        return ucclSuccess;
    }

    auto* req = static_cast<ucxRequest*>(request);

    /* Drive UCP progress — the request object is allocated by UCX
     * from the worker's request pool, but we don't have a direct
     * worker pointer here. However, UCX stores the worker reference
     * internally. We use ucp_request_check_status for polling. */
    ucs_status_t status = ucp_request_check_status(request);
    if (status == UCS_INPROGRESS) {
        *done = 0;
        if (size) *size = 0;
        return ucclSuccess;
    }

    if (status != UCS_OK) {
        UCCL_LOG(ERROR, "UCX request completed with error: %s",
                 ucs_status_string(status));
        ucp_request_free(request);
        *done = 1;
        if (size) *size = 0;
        return ucclSystemError;
    }

    /* Completed successfully */
    *done = 1;
    if (size) *size = static_cast<int>(req->length);
    ucp_request_free(request);
    return ucclSuccess;
}

static ucclResult_t ucxCloseSend(void* sendComm) {
    if (sendComm == nullptr) return ucclSuccess;
    auto* scomm = static_cast<ucxSendComm*>(sendComm);

    if (scomm->ep) {
        ucp_request_param_t param = {};
        param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        param.flags = UCP_EP_CLOSE_FLAG_FORCE;
        ucs_status_ptr_t closeReq = ucp_ep_close_nbx(scomm->ep, &param);
        if (UCS_PTR_IS_PTR(closeReq)) {
            /* Wait for close to complete */
            while (ucp_request_check_status(closeReq) == UCS_INPROGRESS) {
                ucp_worker_progress(scomm->ctx->ucpWorker);
            }
            ucp_request_free(closeReq);
        }
    }
    delete scomm;
    return ucclSuccess;
}

static ucclResult_t ucxCloseRecv(void* recvComm) {
    if (recvComm == nullptr) return ucclSuccess;
    auto* rcomm = static_cast<ucxRecvComm*>(recvComm);

    if (rcomm->ep) {
        ucp_request_param_t param = {};
        param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        param.flags = UCP_EP_CLOSE_FLAG_FORCE;
        ucs_status_ptr_t closeReq = ucp_ep_close_nbx(rcomm->ep, &param);
        if (UCS_PTR_IS_PTR(closeReq)) {
            while (ucp_request_check_status(closeReq) == UCS_INPROGRESS) {
                ucp_worker_progress(rcomm->ctx->ucpWorker);
            }
            ucp_request_free(closeReq);
        }
    }
    delete rcomm;
    return ucclSuccess;
}

static ucclResult_t ucxCloseListen(void* listenComm) {
    if (listenComm == nullptr) return ucclSuccess;
    auto* lcomm = static_cast<ucxListenComm*>(listenComm);

    if (lcomm->listener) {
        ucp_listener_destroy(lcomm->listener);
    }
    delete lcomm;
    return ucclSuccess;
}

static ucclResult_t ucxProgress(void* handle) {
    if (handle == nullptr) return ucclSuccess;
    auto* ctx = static_cast<ucxContext*>(handle);
    ucp_worker_progress(ctx->ucpWorker);
    return ucclSuccess;
}

static ucclResult_t ucxSetTag(void* comm, uint64_t tag) {
    if (comm == nullptr) return ucclInvalidArgument;
    static_cast<ucxCommBase*>(comm)->tag = tag;
    return ucclSuccess;
}

#else /* !UCCL_HAS_UCX — stub fallback */

static ucclResult_t ucxInit(void** handle) {
    UCCL_LOG(WARN, "UCX plugin: stub (UCX not built). Network transport will not function.");
    *handle = nullptr;
    return ucclSuccess;
}
static ucclResult_t ucxFinalize(void* handle) { (void)handle; return ucclSuccess; }
static ucclResult_t ucxDevices(int* ndev) { *ndev = 0; return ucclSuccess; }
static ucclResult_t ucxGetProperties(int dev, ucclNetProperties_t* props) {
    (void)dev;
    if (props == nullptr) return ucclInvalidArgument;
    std::memset(props, 0, sizeof(*props));
    snprintf(props->name, sizeof(props->name), "ucx-stub");
    return ucclSuccess;
}
static ucclResult_t ucxListen(void* h, void* a, void** lc) { (void)h; (void)a; *lc = nullptr; return ucclSuccess; }
static ucclResult_t ucxConnect(void* h, void* a, void** sc) { (void)h; (void)a; *sc = nullptr; return ucclSuccess; }
static ucclResult_t ucxAccept(void* lc, void** rc) { (void)lc; *rc = nullptr; return ucclSuccess; }
static ucclResult_t ucxRegMr(void* c, void* d, size_t s, int t, void** m) { (void)c; (void)d; (void)s; (void)t; *m = nullptr; return ucclSuccess; }
static ucclResult_t ucxDeregMr(void* c, void* m) { (void)c; (void)m; return ucclSuccess; }
static ucclResult_t ucxIsend(void* sc, void* d, size_t s, void* m, void** r) { (void)sc; (void)d; (void)s; (void)m; *r = nullptr; return ucclSuccess; }
static ucclResult_t ucxIrecv(void* rc, void* d, size_t s, void* m, void** r) { (void)rc; (void)d; (void)s; (void)m; *r = nullptr; return ucclSuccess; }
static ucclResult_t ucxTest(void* r, int* done, int* sz) { (void)r; *done = 1; if (sz) *sz = 0; return ucclSuccess; }
static ucclResult_t ucxCloseSend(void* sc) { (void)sc; return ucclSuccess; }
static ucclResult_t ucxCloseRecv(void* rc) { (void)rc; return ucclSuccess; }
static ucclResult_t ucxCloseListen(void* lc) { (void)lc; return ucclSuccess; }
static ucclResult_t ucxProgress(void* h) { (void)h; return ucclSuccess; }
static ucclResult_t ucxSetTag(void* c, uint64_t t) { (void)c; (void)t; return ucclSuccess; }

#endif /* UCCL_HAS_UCX */

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
    ucxCloseListen,
    ucxProgress,
    ucxSetTag
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
