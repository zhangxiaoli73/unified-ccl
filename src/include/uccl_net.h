#ifndef UCCL_NET_H_
#define UCCL_NET_H_

#include "uccl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Network device properties */
typedef struct {
    char name[256];
    int pciPath;
    int guid;
    int ptrSupport;         /* UCCL_PTR_HOST | UCCL_PTR_DEVICE */
    int speed;              /* Network bandwidth in MB/s */
    int latency;            /* Latency in ns */
    int maxComms;
} ucclNetProperties_t;

/* Network plugin interface — mirrors ncclNet_t */
typedef struct {
    const char* name;       /* "UCX", "OFI", "custom", ... */

    /* Init / Finalize */
    ucclResult_t (*init)(void** handle);
    ucclResult_t (*finalize)(void* handle);

    /* Device management */
    ucclResult_t (*devices)(int* ndev);
    ucclResult_t (*getProperties)(int dev, ucclNetProperties_t* props);

    /* Connection establishment */
    ucclResult_t (*listen)(void* handle, void* listenAddr,
                           void** listenComm);
    ucclResult_t (*connect)(void* handle, void* connectAddr,
                            void** sendComm);
    ucclResult_t (*accept)(void* listenComm, void** recvComm);

    /* Memory registration (GPU Direct) */
    ucclResult_t (*regMr)(void* comm, void* data, size_t size,
                          int type, void** mhandle);
    ucclResult_t (*deregMr)(void* comm, void* mhandle);

    /* Async data transfer */
    ucclResult_t (*isend)(void* sendComm, void* data, size_t size,
                          void* mhandle, void** request);
    ucclResult_t (*irecv)(void* recvComm, void* data, size_t size,
                          void* mhandle, void** request);
    ucclResult_t (*test)(void* request, int* done, int* size);

    /* Connection teardown */
    ucclResult_t (*closeSend)(void* sendComm);
    ucclResult_t (*closeRecv)(void* recvComm);
    ucclResult_t (*closeListen)(void* listenComm);
} ucclNet_t;

/* Plugin discovery: dynamic library exports this symbol */
/* extern ucclNet_t ucclNetPlugin_v1; */

#ifdef __cplusplus
}
#endif

#endif /* UCCL_NET_H_ */
