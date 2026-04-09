# Unified-CCL Design Specification

#TODO:

- [x] staging buffer 的使用（见 Section 7.4）
- [ ] PTX 指令的使用
- [x] one shot allreduce 的实现（见 Section 6.3）
- [x] symmetric memory allreduce 的实现（见 Section 6.4）
- [x] 增加 allgather / reducescatter 的实现（见 Section 4, 6.5, 6.6）
- [ ] allgatherv / reducescatterv 的实现 (future)
- [ ] 参考实现 Broadcom 的 GDA 实现
- [ ] 参考实现 CX6 的 GDA 实现
- [ ] 提供 device API for GDR and GDA
- [ ] check Meta torchcomms 里面关于 QP 的处理，能不能直接从 plugin 里面接入？
- [ ] check uccl 里面关于 QP 的处理，能不能直接从 plugin 里面接入？


## 1. 项目概述

Unified-CCL 是一个面向 Intel GPU 的集合通信库，设计上对标 NVIDIA NCCL。目标是提供高性能的GPU间集合通信原语，支持单机多卡和跨节点 GPU RDMA 通信。

### 1.1 核心需求

| 维度 | 决定 |
|------|------|
| 目标硬件 | Intel GPU (Data Center GPU Max / Ponte Vecchio, Arc) |
| 编程模型 | SYCL 优先，必要时使用 Level Zero |
| 拓扑范围 | 单机多卡 + 跨节点 GPU RDMA |
| 网络传输 | UCX (Unified Communication X)，支持 network plugin 扩展 |
| Python API | 独立 Python API，接受 `torch.Tensor`，配套 Python tests |
| 算法 | MVP: Ring, One-Shot, Symmetric Memory AllReduce，架构可扩展 (Tree, CollNet 等) |
| 协议 | MVP: Simple + LL128，架构可扩展 |
| 数据类型 | bf16, fp16，reduction = sum |
| 构建系统 | CMake + pybind11 |

### 1.2 参考项目

- [NVIDIA NCCL](https://github.com/NVIDIA/nccl) — 核心架构参考
- 严格 follow NCCL 的 Protocol/Algorithm 分离设计
- 支持 Network plugin 可插拔架构

---

## 2. 架构设计

### 2.1 NCCL 三层分离模型

Unified-CCL 对标 NCCL，将 collective 执行分解为三个独立维度：

```
┌──────────────────────────────────────────────────────┐
│     Collective (AllReduce / AllGather / ReduceScatter)│
├──────────────────┬───────────────┬───────────────────┤
│    Algorithm     │   Protocol    │     Network       │
│   (通信拓扑)      │  (传输策略)    │   (网络后端)       │
│                  │               │                   │
│  Ring            │  Simple       │  UCX (内置)        │
│  One-Shot        │  LL128        │  Plugin ...       │
│  SymmetricMemory │  LL (后续)     │                   │
│  Tree (后续)      │  ...          │                   │
│  CollNet (后续)   │               │                   │
└──────────────────┴───────────────┴───────────────────┘
```

- **Algorithm**：决定数据在 rank 之间的流动拓扑（ring 环、tree 树等）
- **Protocol**：决定单步数据传输的方式（Simple 直接拷贝、LL128 128字节行+flag 低延迟等）
- **Network Plugin**：可插拔的网络后端接口（对标 NCCL 的 `ncclNet_t`）

### 2.2 整体数据流

```
Python API (pybind11)
    │
    ▼
┌─────────────────────────────────────────┐
│   C API: ucclAllReduce(...)             │  ← uccl.h 公共接口
├─────────────────────────────────────────┤
│   collectives.cc → enqueue.cc           │  ← 选择 algorithm + protocol
├──────────┬──────────────────────────────┤
│          │        Primitives            │
│Algorithm │  (send/recv/reduce/copy)     │  ← protocol 实现通信原语
│ (Ring)   │   ┌────────┬────────┐        │
│          │   │ Simple │ LL128  │        │
├──────────┴───┴────────┴────────┴────────┤
│         Transport Layer                 │
│   ┌──────────┬──────────────────┐       │
│   │  P2P     │   Net            │       │
│   │(intra-   │ (inter-node,     │       │
│   │ node)    │  calls plugin)   │       │
│   └──────────┴──────────────────┘       │
├─────────────────────────────────────────┤
│   Network Plugin Interface (ucclNet_t)  │
│   ┌─────────┬───────────────────┐       │
│   │ UCX     │  Custom Plugin    │       │
│   │(built-in)│ (动态加载 .so)    │       │
│   └─────────┴───────────────────┘       │
└─────────────────────────────────────────┘
```

### 2.3 对标 NCCL 模块映射

| NCCL 模块 | Unified-CCL 模块 | 说明 |
|-----------|-----------------|------|
| `src/init.cc` | `src/init.cc` | CommInitRank, CommDestroy 等 |
| `src/collectives.cc` | `src/collectives.cc` | AllReduce 等 collective 调度 |
| `src/enqueue.cc` | `src/enqueue.cc` | algorithm/protocol 选择与任务入队 |
| `src/channel.cc` | `src/channel.cc` | Channel 管理 |
| `src/transport/` | `src/transport/` | NCCL: net/shm/p2p → 我们: net/p2p |
| `src/device/prims_simple.h` | `src/protocols/simple.cc` | Simple protocol |
| `src/device/prims_ll128.h` | `src/protocols/ll128.cc` | LL128 protocol |
| `src/device/op128.h` | `src/device/op128.hpp` | 128-bit load/store 原语 |
| `src/device/reduce_kernel.h` | `src/device/reduce_kernel.hpp` | Reduce kernel |
| `src/bootstrap.cc` | `src/bootstrap.cc` | Rank 发现 (MPI bootstrap) |
| `src/proxy.cc` | `src/proxy.cc` | Proxy 线程，处理网络 I/O |
| `src/include/nccl_net.h` | `src/include/uccl_net.h` | Network plugin 接口 |
| `src/graph/` | `src/topo/` | 拓扑发现 — PCIe/UPI 链路探测，ring 排序优化 |

---

## 3. 项目结构

```
unified-ccl/
├── CMakeLists.txt                          # 顶层 CMake
├── src/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── uccl.h                          # 公共 C API（对标 nccl.h）
│   │   ├── uccl_net.h                      # Network plugin 接口（对标 nccl_net.h）
│   │   └── uccl_common.h                   # 共享类型定义
│   │
│   ├── init.cc                             # Communicator 初始化/销毁
│   ├── collectives.cc                      # Collective 调度入口
│   ├── enqueue.cc                          # 任务入队 & algorithm/protocol 选择
│   ├── group.cc                            # Group semantics
│   ├── bootstrap.cc                        # 跨进程 rank 发现（MPI bootstrap）
│   ├── proxy.cc                            # Proxy 线程（网络 I/O 异步处理）
│   ├── channel.cc                          # Channel 管理
│   ├── transport.cc                        # Transport 层抽象
│   │
│   ├── topo/                               # === 拓扑发现 ===
│   │   ├── topo.h                          # 拓扑数据结构与接口
│   │   ├── topo_pci.cc                     # PCIe 拓扑探测
│   │   ├── topo_upi.cc                     # UPI 跨 socket 探测
│   │   └── topo_ring.cc                    # 基于拓扑的 ring 排序
│   │
│   ├── algorithms/                         # === Algorithm 层 ===
│   │   ├── algorithm.h                     # 算法抽象接口
│   │   ├── ring.cc                         # Ring AllReduce / AllGather / ReduceScatter（MVP）
│   │   ├── one_shot.cc                     # One-Shot AllReduce（MVP，小消息优化）
│   │   └── symmetric.cc                    # Symmetric Memory AllReduce（MVP，单节点）
│   │                                       # 后续: tree.cc, collnet.cc
│   │
│   ├── protocols/                          # === Protocol 层 ===
│   │   ├── protocol.h                      # 协议抽象接口
│   │   ├── simple.cc                       # Simple protocol（高带宽，直接 copy）
│   │   └── ll128.cc                        # LL128 protocol（128B line, flag-based 低延迟）
│   │                                       # 后续: ll.cc
│   │
│   ├── transport/                          # === Transport 层 ===
│   │   ├── transport.h                     # Transport 抽象接口
│   │   ├── p2p.cc                          # Intra-node P2P (SYCL USM / L0 IPC)
│   │   └── net.cc                          # Inter-node network（调用 plugin 接口）
│   │
│   ├── plugin/                             # === Network Plugin 框架 ===
│   │   ├── net_plugin.h                    # 插件加载器
│   │   └── net_ucx.cc                      # 内置 UCX plugin
│   │
│   ├── device/                             # === SYCL Device Kernels ===
│   │   ├── reduce_kernel.hpp               # bf16/fp16 sum reduce kernel
│   │   ├── primitives.hpp                  # 通信原语顶层（对标 nccl primitives.h）
│   │   ├── op128.hpp                       # 128-bit load/store（对标 nccl op128.h）
│   │   └── common_kernel.hpp               # 通用 kernel 工具
│   │
│   └── misc/
│       ├── debug.cc                        # 日志/调试
│       └── utils.cc                        # 通用工具
│
├── plugins/                                # 外部 network plugin
│   └── ucx/
│       ├── CMakeLists.txt
│       └── plugin.cc                       # 可独立编译为 libuccl-net-ucx.so
│
├── bindings/
│   └── python/
│       ├── CMakeLists.txt
│       ├── uccl_bindings.cc                # pybind11 绑定
│       └── uccl/
│           ├── __init__.py
│           └── uccl.py                     # 高层 Python API
│
├── tests/
│   ├── cpp/
│   │   ├── test_allreduce.cc
│   │   └── test_transport.cc
│   └── python/
│       ├── test_allreduce.py
│       └── test_perf.py
│
└── docs/
```

---

## 4. 公共 C API（uccl.h）

对标 NCCL 的 `nccl.h`，精简到 MVP 所需的最小接口：

```c
#ifndef UCCL_H_
#define UCCL_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Types
 * ============================================================ */

/* Opaque communicator handle */
typedef struct ucclComm* ucclComm_t;
#define UCCL_COMM_NULL NULL

/* Unique ID for communicator bootstrap */
#define UCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[UCCL_UNIQUE_ID_BYTES]; } ucclUniqueId;

/* Error codes */
typedef enum {
    ucclSuccess             = 0,
    ucclSystemError          = 1,
    ucclInternalError        = 2,
    ucclInvalidArgument      = 3,
    ucclInvalidUsage         = 4,
    ucclRemoteError          = 5,
    ucclInProgress           = 6,
    ucclNumResults           = 7
} ucclResult_t;

/* Data types — MVP: fp16, bf16 */
typedef enum {
    ucclFloat16    = 0,
    ucclBfloat16   = 1,
    ucclNumTypes   = 2
} ucclDataType_t;

/* Reduction operations — MVP: sum only */
typedef enum {
    ucclSum = 0,
    ucclNumOps = 1
} ucclRedOp_t;

/* ============================================================
 * Communicator Lifecycle
 * ============================================================ */

ucclResult_t ucclGetVersion(int* version);
ucclResult_t ucclGetUniqueId(ucclUniqueId* uniqueId);

ucclResult_t ucclCommInitRank(ucclComm_t* comm, int nranks,
                              ucclUniqueId commId, int rank);
ucclResult_t ucclCommInitAll(ucclComm_t* comm, int ndev,
                             const int* devlist);
ucclResult_t ucclCommFinalize(ucclComm_t comm);
ucclResult_t ucclCommDestroy(ucclComm_t comm);
ucclResult_t ucclCommAbort(ucclComm_t comm);
ucclResult_t ucclCommCount(const ucclComm_t comm, int* count);
ucclResult_t ucclCommUserRank(const ucclComm_t comm, int* rank);

/* ============================================================
 * Collective Operations — MVP: AllReduce, AllGather, ReduceScatter
 * ============================================================ */

ucclResult_t ucclAllReduce(const void* sendbuff, void* recvbuff,
                           size_t count, ucclDataType_t datatype,
                           ucclRedOp_t op, ucclComm_t comm,
                           void* stream);
/* stream: sycl::queue* ，用 void* 保持 C 兼容 */

ucclResult_t ucclAllGather(const void* sendbuff, void* recvbuff,
                           size_t sendcount, ucclDataType_t datatype,
                           ucclComm_t comm, void* stream);
/* sendbuff: 每个 rank 的输入数据，大小为 sendcount 个元素
 * recvbuff: 输出缓冲区，大小为 sendcount * nranks 个元素
 * 每个 rank 贡献 sendcount 个元素，结果是所有 rank 数据的拼接 */

ucclResult_t ucclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, ucclDataType_t datatype,
                               ucclRedOp_t op, ucclComm_t comm,
                               void* stream);
/* sendbuff: 输入缓冲区，大小为 recvcount * nranks 个元素
 * recvbuff: 每个 rank 的输出数据，大小为 recvcount 个元素
 * 对输入数据先做 reduce，然后 scatter 到各 rank */

/* ============================================================
 * Group Semantics
 * ============================================================ */

ucclResult_t ucclGroupStart(void);
ucclResult_t ucclGroupEnd(void);

/* ============================================================
 * Error Reporting
 * ============================================================ */

const char* ucclGetErrorString(ucclResult_t result);
const char* ucclGetLastError(ucclComm_t comm);

#ifdef __cplusplus
}
#endif

#endif /* UCCL_H_ */
```

---

## 5. Network Plugin 接口（uccl_net.h）

对标 NCCL 的 `nccl_net.h`，外部 network plugin 实现此接口：

```c
#ifndef UCCL_NET_H_
#define UCCL_NET_H_

#include "uccl_common.h"

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

#define UCCL_PTR_HOST   0x1
#define UCCL_PTR_DEVICE 0x2  /* GPU Direct RDMA support */

/* Network plugin interface — 对标 ncclNet_t */
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

/* Plugin discovery: 动态库需导出此符号 */
/* extern ucclNet_t ucclNetPlugin_v1; */

#endif /* UCCL_NET_H_ */
```

---

## 6. Algorithm 层

### 6.1 抽象接口

```cpp
// algorithms/algorithm.h
#pragma once
#include "uccl_common.h"

enum ucclAlgorithm_t {
    UCCL_ALGO_RING            = 0,
    UCCL_ALGO_ONE_SHOT        = 1,   // One-Shot AllReduce（小消息优化）
    UCCL_ALGO_SYMMETRIC_MEM   = 2,   // Symmetric Memory AllReduce（单节点）
    // UCCL_ALGO_TREE = 3,            // 后续扩展
    // UCCL_ALGO_COLLNET = 4,         // 后续扩展
    UCCL_NUM_ALGORITHMS
};
```

算法不直接操作数据传输，而是通过 `Primitives`（由 Protocol 实现）提供的 `send()`/`recv()`/`recvReduceSend()`/`recvReduceCopySend()` 等原语来编排数据流动。这是 NCCL 的核心设计模式。

> **例外：** Symmetric Memory AllReduce 不使用 Primitives 抽象，而是直接操作 symmetric memory 地址进行 load/reduce/store。

### 6.2 Ring AllReduce

Ring AllReduce 分两个阶段：

1. **Reduce-Scatter**：沿 ring 传播，每个 rank 累加收到的数据块
2. **AllGather**：沿 ring 广播最终结果

```cpp
// 伪代码 — 对标 NCCL src/device/sendrecv.h 中 ring allreduce kernel
template <typename T, typename RedOp, typename Proto>
__device__ void runRingAllReduce(
    Primitives<T, RedOp, Proto>& prims,
    const T* sendbuff, T* recvbuff,
    size_t count, int nranks, int rank)
{
    size_t chunkCount = count / nranks;

    // Phase 1: Reduce-Scatter
    for (int step = 0; step < nranks - 1; step++) {
        int sendChunk = (rank - step) % nranks;
        int recvChunk = (rank - step - 1) % nranks;
        size_t sendOff = sendChunk * chunkCount;
        size_t recvOff = recvChunk * chunkCount;

        if (step == 0)
            prims.recvReduceCopySend(sendOff, recvOff, chunkCount);
        else
            prims.recvReduceSend(sendOff, chunkCount);
    }

    // Phase 2: AllGather
    for (int step = 0; step < nranks - 1; step++) {
        int chunk = (rank + 1 - step) % nranks;
        size_t offset = chunk * chunkCount;

        prims.recvCopySend(offset, chunkCount);
    }
}
```

### 6.3 One-Shot AllReduce

One-Shot AllReduce 是针对**小消息**优化的算法。与 Ring 需要 `2*(N-1)` 步不同，One-Shot 在**一轮通信**内完成 AllReduce，牺牲带宽换取最低延迟。

**核心思路：**

- 每个 rank 将自己的完整数据广播给所有其他 rank
- 每个 rank 本地对收到的所有数据做 reduce
- 通信步骤 = 1（all-to-all exchange），但每步传输的数据量 = 全量

**适用场景：**

| 特征 | 说明 |
|------|------|
| 消息大小 | < 4KB（典型：梯度同步中的小 tensor、标量聚合） |
| 优势 | 最低延迟，无多步流水线开销 |
| 劣势 | 带宽效率低（每个 rank 发送全量数据，总通信量 = N * dataSize） |
| GPU 数量 | 适合 2-8 卡，卡数过多时带宽浪费严重 |

```
One-Shot AllReduce 数据流（4 ranks, data = [A, B, C, D]）:

步骤 1: 每个 rank 将自己的数据写入所有其他 rank 的接收 buffer
  Rank 0: A → [Rank1.buf, Rank2.buf, Rank3.buf]
  Rank 1: B → [Rank0.buf, Rank2.buf, Rank3.buf]
  Rank 2: C → [Rank0.buf, Rank1.buf, Rank3.buf]
  Rank 3: D → [Rank0.buf, Rank1.buf, Rank2.buf]

步骤 2: 每个 rank 本地 reduce
  Rank 0: result = A + B + C + D
  Rank 1: result = A + B + C + D
  ...
```

```cpp
// algorithms/one_shot.cc
// One-Shot AllReduce — 一轮通信完成
template <typename T, typename RedOp, typename Proto>
void runOneShotAllReduce(
    Primitives<T, RedOp, Proto>& prims,
    const T* sendbuff, T* recvbuff,
    size_t count, int nranks, int rank)
{
    // Phase 1: 将自己的数据发送给所有其他 rank
    for (int peer = 0; peer < nranks; peer++) {
        if (peer == rank) continue;
        prims.send(0, count);  // send full data to each peer
    }

    // Phase 2: 接收所有 peer 的数据并 reduce
    // 先 copy 本地数据到 output
    prims.copySend(0, 0, count);

    for (int peer = 0; peer < nranks; peer++) {
        if (peer == rank) continue;
        // 接收 + reduce 到 output buffer
        prims.recvReduceSend(0, count);
    }
}
```

**与 Ring 的对比（4 ranks, message size = M）：**

| 指标 | Ring AllReduce | One-Shot AllReduce |
|------|---------------|-------------------|
| 通信步数 | 2*(4-1) = 6 步 | 1 步（all-to-all） |
| 每步数据量 | M/4 | M |
| 总通信量 | 2*(4-1)/4 * M = 1.5M | (4-1) * M = 3M |
| 延迟（small M）| 6 * latency | 1 * latency |
| 延迟（large M）| 受限于 6 × M/4 ÷ BW | 受限于 3M ÷ BW |

### 6.4 Symmetric Memory AllReduce

Symmetric Memory AllReduce 是基于**对称内存映射**的算法，**仅限单节点**使用。核心思想是所有 GPU 通过 P2P 内存映射，每个 GPU 都能直接读取其他所有 GPU 的 device memory，无需显式的 send/recv 通信。

**前提条件：**
- 所有 GPU 在同一节点内（intra-node）
- 每个 GPU 的通信 buffer 通过 Level Zero IPC（`zeMemGetIpcHandle`/`zeMemOpenIpcHandle`）映射到所有其他 GPU 的地址空间
- 初始化阶段通过 bootstrap 交换 IPC handle，建立 **symmetric memory mapping**

**核心设计：**

```
Symmetric Memory Layout (4 GPUs):

GPU 0 地址空间:
  ┌─────────────┐
  │ local_buf   │ ← 自己的 device memory (sycl::malloc_device)
  ├─────────────┤
  │ remote[0]   │ ← 映射自 GPU 0（= local_buf 本身）
  │ remote[1]   │ ← 映射自 GPU 1 的 device memory (via IPC)
  │ remote[2]   │ ← 映射自 GPU 2 的 device memory (via IPC)
  │ remote[3]   │ ← 映射自 GPU 3 的 device memory (via IPC)
  └─────────────┘

每个 GPU 都有一个 remote[] 指针数组，可以直接读取其他 GPU 的数据。
```

**算法流程：**

```
Symmetric Memory AllReduce（read-reduce 模式）:

1. 每个 rank 将 sendbuff 数据写入自己的 symmetric buffer
2. Barrier（确保所有 rank 的写入完成）
3. 每个 rank 从所有其他 rank 的 symmetric buffer 中读取数据
4. 在本地 GPU 上对读取的所有数据做 reduce
5. 将结果写入 recvbuff
```

```cpp
// algorithms/symmetric.cc
// Symmetric Memory AllReduce — 基于对称内存直接读取

struct SymmetricMemoryContext {
    int nGpus;
    void** remoteBuffs;     // remoteBuffs[i] = GPU i 的 buffer 在本 GPU 地址空间的映射
    size_t buffSize;
    void* localBuff;        // 本 GPU 的 buffer
};

template <typename T, typename RedOp>
void runSymmetricAllReduce(
    const SymmetricMemoryContext& ctx,
    const T* sendbuff, T* recvbuff,
    size_t count, int nGpus, int myGpu,
    sycl::nd_item<1> item)
{
    int lid = item.get_local_id(0);
    int groupSize = item.get_local_range(0);
    RedOp op;

    // 每个 work-item 处理一部分元素
    for (size_t i = lid; i < count; i += groupSize) {
        // 从自己的 sendbuff 初始化
        T acc = sendbuff[i];

        // 从所有其他 GPU 的 symmetric buffer 读取并 reduce
        for (int g = 0; g < nGpus; g++) {
            if (g == myGpu) continue;
            const T* remoteBuf = static_cast<const T*>(ctx.remoteBuffs[g]);
            T remoteVal = remoteBuf[i];  // 直接 P2P 读取！
            acc = op(acc, remoteVal);
        }

        recvbuff[i] = acc;
    }
}
```

**性能特征：**

| 指标 | 说明 |
|------|------|
| 通信模式 | GPU 发起 P2P read（PCIe / UPI） |
| 通信步数 | 0（无显式通信，靠 P2P load） |
| 带宽效率 | 每个 GPU 读取 (N-1) * M 数据，总 P2P 读取量 = N*(N-1)*M |
| 适用场景 | 单节点 2-8 卡，中小消息 |
| 优势 | 无需 channel/proxy/FIFO 开销，kernel 内直接完成 |
| 劣势 | P2P 读取带宽受 PCIe/UPI 限制；卡数多时 read 扇入成为瓶颈 |

**Symmetric Memory 初始化流程：**

```
1. 每个 rank 分配 device memory:
     buf = sycl::malloc_device(size, queue)

2. 获取 IPC handle:
     zeMemGetIpcHandle(context, buf, &ipcHandle)

3. 通过 MPI AllGather 交换所有 rank 的 IPC handle:
     MPI_Allgather(&myHandle, ..., allHandles, ...)

4. 每个 rank 打开其他 rank 的 IPC handle:
     for each peer:
         zeMemOpenIpcHandle(context, device, peerHandle, &remotePtrs[peer])

5. 现在 remotePtrs[] 包含所有 GPU buffer 的本地映射指针
```

**与 Ring 的对比：**

| 场景 | Ring AllReduce | Symmetric Memory AllReduce |
|------|---------------|--------------------------|
| 2 GPU, 1KB | 2 步 × 延迟 | 1 次 P2P read |
| 8 GPU, 1KB | 14 步 × 延迟 | 7 次 P2P read（并行） |
| 8 GPU, 1MB | Ring 带宽优势明显 | P2P read 带宽扇入瓶颈 |
| 跨节点 | ✓ 支持 | ✗ 不支持（仅 intra-node） |

### 6.5 Ring AllGather

Ring AllGather 是 Ring AllReduce 的第二阶段（Phase 2）独立运行。每个 rank 贡献一块数据，结果是所有 rank 数据的拼接。

```
AllGather (4 ranks, 每个 rank 贡献 chunk):

输入:  Rank0=[A]  Rank1=[B]  Rank2=[C]  Rank3=[D]
输出:  所有 rank 都得到 [A, B, C, D]
```

**Ring 实现：** 沿 ring 传递 `nranks - 1` 步，每步将收到的 chunk 转发给下一个 rank：

```cpp
// algorithms/ring.cc — Ring AllGather
template <typename T, typename Proto>
void runRingAllGather(
    Primitives<T, ReduceSum<T>, Proto>& prims,
    const T* sendbuff, T* recvbuff,
    size_t sendcount, int nranks, int rank)
{
    size_t chunkCount = sendcount;  // 每个 rank 贡献 sendcount 个元素

    // 先将自己的数据 copy 到 recvbuff 中对应位置
    // recvbuff[rank * sendcount ... (rank+1) * sendcount - 1] = sendbuff
    prims.copySend(0, rank * chunkCount, chunkCount);

    // nranks - 1 步: 每步接收前一个 rank 转发的数据，copy 到 output 并继续转发
    for (int step = 0; step < nranks - 1; step++) {
        int chunk = ((rank - step) % nranks + nranks) % nranks;
        size_t offset = chunk * chunkCount;

        prims.recvCopySend(offset, chunkCount);
    }
}
```

### 6.6 Ring ReduceScatter

Ring ReduceScatter 是 Ring AllReduce 的第一阶段（Phase 1）独立运行。对输入数据做 reduce 后 scatter 到各 rank。

```
ReduceScatter (4 ranks, sum):

输入:  每个 rank 都有 [chunk0, chunk1, chunk2, chunk3]
输出:  Rank0=sum(所有rank的chunk0)
       Rank1=sum(所有rank的chunk1)
       Rank2=sum(所有rank的chunk2)
       Rank3=sum(所有rank的chunk3)
```

```cpp
// algorithms/ring.cc — Ring ReduceScatter
template <typename T, typename RedOp, typename Proto>
void runRingReduceScatter(
    Primitives<T, RedOp, Proto>& prims,
    const T* sendbuff, T* recvbuff,
    size_t recvcount, int nranks, int rank)
{
    size_t chunkCount = recvcount;  // 每个 rank 最终得到 recvcount 个元素

    // nranks - 1 步: 沿 ring 传播，每步累加收到的数据块
    for (int step = 0; step < nranks - 1; step++) {
        int sendChunk = ((rank - step) % nranks + nranks) % nranks;
        int recvChunk = ((rank - step - 1) % nranks + nranks) % nranks;
        intptr_t sendOff = sendChunk * chunkCount;
        intptr_t recvOff = recvChunk * chunkCount;

        if (step == 0)
            prims.recvReduceCopySend(sendOff, recvOff, chunkCount);
        else
            prims.recvReduceSend(recvOff, chunkCount);
    }

    // 最终结果在 recvbuff 中，对应 rank 负责的那个 chunk
}
```

---

## 7. Protocol 层

### 7.1 Protocol 抽象

对标 NCCL 的 `ProtoSimple`, `ProtoLL`, `ProtoLL128`：

```cpp
// protocols/protocol.h
#pragma once

enum ucclProtocol_t {
    UCCL_PROTO_SIMPLE = 0,
    UCCL_PROTO_LL128  = 1,
    // UCCL_PROTO_LL  = 2,   // 后续扩展
    UCCL_NUM_PROTOCOLS
};

/* Protocol 配置常量 — 对标 NCCL */
struct ProtoSimple {
    static constexpr int Id = UCCL_PROTO_SIMPLE;
    static constexpr int SlicePerChunk = 2;
    static constexpr int StepPerSlice = 4;
};

struct ProtoLL128 {
    static constexpr int Id = UCCL_PROTO_LL128;
    /* 128 bytes = 16 x uint64_t 构成一行
     * 其中 14 个是数据，2 个是 flag */
    static constexpr int LineElems = 16;   // NCCL_LL128_LINEELEMS
    static constexpr int DataElems = 14;   // NCCL_LL128_DATAELEMS
    static constexpr int FlagElems = 2;    // 每行 2 个 flag 位
};
```

### 7.2 Simple Protocol

高带宽协议，每步直接拷贝整块数据。使用 head/tail 指针同步生产者/消费者：

- 发送方写数据到 FIFO buffer，更新 tail
- 接收方轮询 head，确认数据可读后消费

### 7.3 LL128 Protocol

低延迟协议（对标 NCCL `prims_ll128.h`），核心思路：

- 每个传输单元是 **128 字节的行**（16 个 `uint64_t`）
- 其中 14 个 `uint64_t` 携带数据，2 个 `uint64_t` 携带 flag
- 接收方通过检测 **flag 值** 判断数据是否到达，避免额外的同步开销
- 一个 warp 中的 **flag thread**（第 7 个 lane）负责写/读 flag

```
128-byte line layout:
┌─────────────────────────────────────────────────┐
│ u64[0] u64[1] ... u64[6] │ u64[7]  ← flag      │
│ u64[8] u64[9] ... u64[13]│ u64[14] ← data      │  
│                           │ u64[15] ← flag      │
└─────────────────────────────────────────────────┘
14 data words + 2 flag words = 16 x 8B = 128B
```

LL128 提供的 Primitives 操作与 Simple 相同（send, recv, recvReduceSend 等），算法层无需感知差异。

### 7.4 LL128 Staging Buffer 机制

LL128 协议使用 **Shared Local Memory (SLM) 作为 staging buffer**，这是 NCCL LL128 实现中的关键性能优化。Staging buffer 的作用是减少对高延迟 global memory 的访问次数，并允许 warp 内多线程协作完成 128 字节行的组装和拆解。

#### 7.4.1 为什么需要 Staging Buffer

**问题：** LL128 的 128 字节行需要 16 个线程协作处理（每个线程负责一个 `uint64_t`）。如果直接在 global memory 上操作：

1. 每次 reduce 操作需要从 global memory 读取两份数据、reduce、再写回 — 3 次 global memory 访问
2. Flag 检查和数据读取之间存在竞争 — 需要反复 poll global memory
3. 跨步访问 global memory 模式不利于合并（coalescing）

**解决：** 使用 SLM (Shared Local Memory / `__shared__` / `sycl::local_accessor`) 作为中间 staging area：

1. 从 global memory **批量读取**多行 LL128 数据到 SLM（一次 128B 对齐 load）
2. 在 SLM 中进行 **reduce 计算**（SLM 带宽远高于 global memory）
3. 将结果从 SLM **批量写回** global memory（一次 128B 对齐 store）

#### 7.4.2 Staging Buffer 数据流

```
                   LL128 recvReduceSend 数据流:

    ┌──────────────────────────────────────────────────────┐
    │                Global Memory (接收端)                  │
    │  ┌──────────────────────────────────────────────┐    │
    │  │ LL128 Line 0: [d0..d6, FLAG, d7..d13, FLAG]  │    │
    │  │ LL128 Line 1: [d0..d6, FLAG, d7..d13, FLAG]  │    │
    │  │ LL128 Line 2: ...                             │    │
    │  └──────────────────────────────────────────────┘    │
    └───────────────────┬──────────────────────────────────┘
                        │ ① 128B 对齐 load (全 warp 协作)
                        │    等待 FLAG 匹配后才读取数据
                        ▼
    ┌──────────────────────────────────────────────────────┐
    │          SLM Staging Buffer (Shared Local Memory)      │
    │  ┌──────────────────────────────────────────────┐    │
    │  │ staging[0..15] = 接收到的 LL128 line (去掉flag) │    │
    │  │ staging[16..31] = 本地 input 数据              │    │
    │  └──────────────────────────────────────────────┘    │
    │            ② 在 SLM 中做 reduce:                      │
    │               result[i] = staging_recv[i] +           │
    │                           staging_local[i]            │
    │                                                       │
    │  ┌──────────────────────────────────────────────┐    │
    │  │ staging_result[0..13] = reduced data          │    │
    │  └──────────────────────────────────────────────┘    │
    └───────────────────┬──────────────────────────────────┘
                        │ ③ 打包 result + 新 FLAG，
                        │    128B 对齐 store 写回 global
                        ▼
    ┌──────────────────────────────────────────────────────┐
    │                Global Memory (发送端)                  │
    │  ┌──────────────────────────────────────────────┐    │
    │  │ LL128 Line: [result + new FLAG]               │    │
    │  └──────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────┘
```

#### 7.4.3 SLM Buffer 布局

```cpp
// LL128 staging buffer 在 SLM 中的布局
// 每个 work-group 分配的 SLM 大小:
static constexpr int SLM_LL128_LINES = 8;  // 同时 staging 的行数
static constexpr int SLM_LL128_SIZE = SLM_LL128_LINES * 128;  // 1KB

// SLM 分区:
// [0, 8*128)          staging_recv:  从 peer 接收的 LL128 行（去掉 flag 后的纯数据）
// [8*128, 16*128)     staging_local: 本地 input 数据（用于 reduce）
// [16*128, 24*128)    staging_out:   reduce 结果（待写入发送端 global memory）
```

#### 7.4.4 Staging Buffer 的关键操作

```
     操作                  SLM 使用                  Global Memory 访问
  ─────────────────   ─────────────────────   ──────────────────────────
  recv (接收):         SLM ← global (批量)     poll FLAG + 128B load
  reduce (计算):       SLM 内部操作              无 global 访问
  send (发送):         SLM → global (批量)     128B store + 写 FLAG
  recvReduceSend:     上述三步流水线            2 次 global 访问（而非 3 次）
```

**性能收益：**

| 场景 | 无 Staging Buffer | 有 Staging Buffer |
|------|-------------------|-------------------|
| recvReduceSend 每行 | 3 次 global load/store | 2 次 global load/store |
| Reduce 计算 | global memory 延迟 | SLM 延迟（~10x 快） |
| 多行流水线 | 无法批量处理 | 8 行同时在 SLM 中处理 |
| FLAG poll | 每次独立 poll | 批量 poll + 预取到 SLM |

#### 7.4.5 SYCL 实现要点

```cpp
// LL128 staging buffer 的 SYCL 实现
queue.submit([&](sycl::handler& cgh) {
    // 分配 SLM staging buffer
    sycl::local_accessor<uint64_t, 1> staging(
        sycl::range<1>(SLM_LL128_SIZE / sizeof(uint64_t)), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(globalSize, workGroupSize),
        [=](sycl::nd_item<1> item) {
            auto sg = item.get_sub_group();
            int lane = sg.get_local_id()[0];

            // ① 从 global memory 读取 LL128 行到 SLM
            // 先 poll flag，flag 匹配后用 128-bit load 读取整行
            uint64_t* slmPtr = &staging[lineOffset];
            // ... poll flag in global, then:
            slmPtr[lane] = globalLine[lane];  // 全 sub-group 协作 load

            sycl::group_barrier(sg);  // 确保 SLM 写入完成

            // ② 在 SLM 中 reduce
            if (lane < ProtoLL128::DataElems) {
                uint64_t recvVal = staging[recv_offset + lane];
                uint64_t localVal = staging[local_offset + lane];
                staging[result_offset + lane] = reduce(recvVal, localVal);
            }

            sycl::group_barrier(sg);  // 确保 reduce 完成

            // ③ 从 SLM 写回 global memory（打包 flag）
            globalOutLine[lane] = (lane == 7 || lane == 15)
                ? newFlagVal
                : staging[result_offset + lane];
        });
});
```

---

## 8. 拓扑发现（Topology Discovery）

对标 NCCL 的 `src/graph/` 模块。在 Intel 平台上，单机内多 GPU 的互联拓扑直接影响 ring 排序和 transport 选择。

### 8.1 Intel GPU 单机拓扑模型

典型的 Intel 多卡服务器拓扑：

```
                        UPI (cross-socket)
         ┌──────────────────────────────────────┐
         │                                      │
   ┌─────┴──────┐                        ┌──────┴─────┐
   │  Socket 0   │                        │  Socket 1   │
   │  (CPU)      │                        │  (CPU)      │
   ├─────────────┤                        ├─────────────┤
   │ PCIe Root 0 │                        │ PCIe Root 1 │
   ├──┬──┬──┬────┤                        ├──┬──┬──┬────┤
   │G0│G1│G2│G3  │                        │G4│G5│G6│G7  │
   └──┴──┴──┴────┘                        └──┴──┴──┴────┘
```

**关键链路特征：**

| 链路类型 | 场景 | 带宽特征 | 延迟特征 |
|---------|------|---------|----------|
| PCIe intra-switch | 同一 PCIe switch 下的 GPU | 高 (PCIe Gen5 x16: ~64GB/s) | 低 |
| PCIe cross-switch | 同 socket 不同 switch 下的 GPU | 受 root complex 限制 | 中 |
| UPI cross-socket | 跨 socket 的 GPU 通信 | 受 UPI 带宽限制 (~40-80GB/s) | 高 |
| Network (RDMA) | 跨节点 | 取决于网卡 | 最高 |

### 8.2 拓扑探测实现

```cpp
// topo/topo.h
#pragma once
#include <vector>

namespace uccl {

enum ucclLinkType {
    UCCL_LINK_PCIE_SAME_SWITCH = 0,  // 同 PCIe switch
    UCCL_LINK_PCIE_CROSS_SWITCH = 1, // 同 socket, 跨 PCIe switch
    UCCL_LINK_UPI = 2,               // 跨 socket (UPI)
    UCCL_LINK_NET = 3,               // 跨节点 (network)
    UCCL_NUM_LINK_TYPES
};

struct ucclGpuInfo {
    int devIndex;           // SYCL device index
    int pciDomain;
    int pciBus;
    int pciDevice;
    int pciFunction;
    int numaNode;           // NUMA node → 对应 socket
    int socketId;           // CPU socket ID
};

struct ucclTopoLink {
    int gpu1, gpu2;
    ucclLinkType type;
    float bandwidth;        // GB/s
    float latency;          // us
};

struct ucclTopology {
    int nGpus;
    std::vector<ucclGpuInfo> gpus;
    std::vector<ucclTopoLink> links;

    // 获取两个 GPU 之间的链路类型
    ucclLinkType getLinkType(int gpu1, int gpu2) const;
};

// 探测本地拓扑
ucclResult_t ucclTopoDetect(ucclTopology* topo);

} // namespace uccl
```

### 8.3 拓扑探测方法

**PCIe 拓扑**（`topo_pci.cc`）：
- Linux: 解析 `/sys/bus/pci/devices/` 和 `/sys/class/drm/` 获取 GPU 的 PCI BDF (Bus/Device/Function)
- 通过 Level Zero: `zeDevicePciGetPropertiesExt()` 获取 PCI 地址
- 比较 PCI bus 层级判断是否在同一 PCIe switch 下

**NUMA/Socket 探测**（`topo_upi.cc`）：
- Linux: 读取 `/sys/bus/pci/devices/<bdf>/numa_node` 确定 GPU 所属 NUMA 节点
- NUMA 节点不同 → 跨 socket → 通信经过 UPI
- 通过 `libnuma` 或直接读取 sysfs 获取 NUMA 拓扑

### 8.4 Ring 排序优化

基于拓扑信息优化 ring 中 rank 的排列顺序，**最小化跨 UPI 跳数**：

```
不优化的 ring:  G0 → G4 → G1 → G5 → G2 → G6 → G3 → G7
                    ↑UPI     ↑UPI     ↑UPI     ↑UPI     ↑UPI
                    (5 次 UPI 跨越)

优化后的 ring:   G0 → G1 → G2 → G3 → G4 → G5 → G6 → G7
                                     ↑UPI               ↑UPI
                    (2 次 UPI 跨越 — 最小值)
```

```cpp
// topo/topo_ring.cc
// 基于拓扑的 ring 排序
namespace uccl {

// 对标 NCCL ncclTopoCompute() 中 ring 排序逻辑
// 策略：同 socket 的 GPU 尽量相邻排列
std::vector<int> computeRingOrder(const ucclTopology& topo) {
    // 1. 按 socketId 分组
    // 2. 每组内按 PCIe bus 排序
    // 3. Socket 间顺序连接，形成只有 2 次 UPI 跨越的 ring
    // 返回 rank 排列顺序
}

} // namespace uccl
```

### 8.5 拓扑对 Transport 选择的影响

| 拓扑关系 | Transport 选择 | 原因 |
|---------|---------------|------|
| 同 PCIe switch | P2P (direct) | 最低延迟，最高带宽 |
| 同 socket, 跨 switch | P2P (via root complex) | 仍然高效 |
| 跨 socket (UPI) | P2P (经过 UPI) | 额外延迟和带宽限制，需感知 |
| 跨节点 | Net (UCX plugin) | 走网络 |

初始化阶段 (`init.cc`) 调用 `ucclTopoDetect()` 构建拓扑，然后在 `enqueue.cc` 中根据拓扑选择最优的 algorithm/protocol 组合和 ring 排列。

### 8.6 拓扑感知的调优参数

根据拓扑信息，为每种 algorithm/protocol 组合生成最优的调优参数：

```cpp
// topo/topo.h
struct ucclTopoTuning {
    int nChannels;              // 并行通道数
    int nThreads;               // 每个 kernel 的线程数
    size_t chunkSize;           // 每次传输的数据块大小
    ucclAlgorithm_t algorithm;  // 选择的算法
    ucclProtocol_t protocol;    // 选择的协议
    float bandwidth;            // 预估带宽
    float latency;              // 预估延迟
};

// 对标 NCCL ncclTopoTuneModel()
// 根据拓扑和消息大小选择最优参数
ucclResult_t ucclTopoTune(
    const ucclTopology* topo,
    size_t messageSize,
    ucclTopoTuning* tuning);
```

**调优策略：**

| 消息大小 | Protocol 选择 | 原因 |
|---------|--------------|------|
| < 4KB | LL128 | 低延迟优先，flag-based 避免同步开销 |
| 4KB - 512KB | LL128 or Simple | 过渡区，具体取决于拓扑 |
| > 512KB | Simple | 高带宽优先，管道化传输 |

---

## 9. Channel 并行化与 Proxy 线程模型

### 9.1 Channel 概念（对标 NCCL）

**Channel** 是 NCCL 中实现带宽并行化的核心机制。每个 channel 是一个独立的通信通道，拥有自己的 ring 排序、buffer、和传输连接。多个 channel 同时运行，每个处理数据的一个子集，从而充分利用硬件带宽。

```
数据分块：
┌───────────────────────────────────────────────┐
│              全部数据 (AllReduce)                │
├───────────┬───────────┬───────────┬───────────┤
│ Channel 0  │ Channel 1  │ Channel 2  │ Channel 3  │
│ (ring A)   │ (ring B)   │ (ring C)   │ (ring D)   │
└───────────┴───────────┴───────────┴───────────┘
     │               │               │               │
     ▼               ▼               ▼               ▼
  并行执行，每个 channel 独立进行 ring allreduce
```

```cpp
// src/include/channel.h
#define UCCL_MAX_CHANNELS 16

struct ucclChannel {
    int id;

    // Ring 拓扑（channel 可以有不同的 ring 排序）
    struct ucclRing {
        int prev;                   // ring 中的前一个 rank
        int next;                   // ring 中的后一个 rank
        int userRanks[UCCL_MAX_RANKS]; // rank 排列
    } ring;

    // 每个 peer 的连接信息
    struct ucclChannelPeer {
        struct ucclConnInfo send;   // 发送端连接
        struct ucclConnInfo recv;   // 接收端连接
    }* peers;

    // Protocol buffers
    void* buffs[UCCL_NUM_PROTOCOLS]; // 每个 protocol 的 FIFO buffer
    size_t buffSizes[UCCL_NUM_PROTOCOLS];
};
```

### 9.2 多 NIC 场景下的带宽并行化（对标 NCCL）

跨节点通信时，核心挑战是如何同时利用**卡内带宽（PCIe/P2P）**和**卡间带宽（NIC/RDMA）**。NCCL 通过 **Channel + Proxy 线程** 架构解决这个问题。

#### 9.2.1 问题：多 NIC 场景

```
Node 0                                     Node 1
┌─────────────────────────┐          ┌─────────────────────────┐
│ GPU0  GPU1  GPU2  GPU3  │          │ GPU4  GPU5  GPU6  GPU7  │
│  │     │     │     │    │          │  │     │     │     │    │
│  └──PCIe──┘     └──PCIe─┘    │          │  └──PCIe──┘     └──PCIe─┘    │
│         │               │    │          │         │               │    │
│       NIC0            NIC1   │          │       NIC0            NIC1   │
└─────────┬──────────────┬─────┘          └────┬──────────────┬─────────┘
          │              │                      │              │
          └─────Network───────────────────┘              │
                         └─────────────Network─────────────┘
```

每个节点可能有多个 NIC，各自亲和不同的 GPU。如果只用一个 NIC，网络带宽成为瓶颈。

#### 9.2.2 NCCL 的解决方案：Channel + Proxy

NCCL 的核心思路是——**GPU kernel 处理卡内通信，CPU proxy 线程处理网络 I/O，两者并行执行**：

```
        GPU Kernel (device)            CPU Proxy Thread (host)
        ────────────────────            ──────────────────────
        │                              │
        │ Intra-node P2P               │ Network send/recv
        │ (GPU→GPU via PCIe/UPI)        │ (via ucclNet plugin)
        │                              │
        │ reduce kernel                │ regMr / isend / irecv
        │ (bf16/fp16 sum)              │ (UCX RDMA ops)
        │                              │
        │──── FIFO buffer ──────────→│ 读 FIFO，proxy 发网络
        │──── FIFO buffer ▒▒▒▒▒▒▒▒▒▒─│ proxy 收网络，写 FIFO
        │                              │
     head/tail 指针同步             轮询 head/tail
```

**工作流程（Ring AllReduce 跨节点）：**

1. **GPU kernel** 通过 P2P 完成本节点内的 reduce
2. GPU kernel 将需要发送到远程节点的数据写入 **FIFO buffer**，更新 tail 指针
3. **Proxy 线程**轮询 tail，发现新数据后调用 `ucclNet->isend()` 发送
4. 远程 proxy 调用 `ucclNet->irecv()` 接收，写入接收端 FIFO buffer，更新 head
5. 远程 GPU kernel 轮询 head，发现新数据后继续 reduce
6. 各 channel **管道化执行**，当第一个 chunk 在网络上传输时，下一个 chunk 的本地 reduce 已在进行

#### 9.2.3 Channel 分配策略：NIC 亲和性

NCCL 根据 GPU 和 NIC 的 PCIe 拓扑亲和性，将不同 channel 分配给不同 NIC：

```
Channel 0,1 ──→ NIC0 (PCIe 亲和 GPU0, GPU1)
Channel 2,3 ──→ NIC1 (PCIe 亲和 GPU2, GPU3)
```

这样每个 NIC 的带宽都被充分利用，且 GPU-NIC 之间的 PCIe 传输路径最短。

#### 9.2.4 管道化时序图

单个 channel 的跨节点 ring allreduce 时序：

```
GPU kernel:   [reduce chunk0]  [reduce chunk1]  [reduce chunk2]  ...
                    │                 │                 │
               write FIFO       write FIFO       write FIFO
                    │                 │                 │
                    ▼                 ▼                 ▼
Proxy thread: [send chunk0]    [send chunk1]    [send chunk2]    ...
              [recv chunk0']   [recv chunk1']   [recv chunk2']   ...
                    │                 │                 │
               write FIFO       write FIFO       write FIFO
                    │                 │                 │
                    ▼                 ▼                 ▼
GPU kernel:   [reduce chunk0'] [reduce chunk1'] [reduce chunk2'] ...

              ←──── 重叠执行，管道化 ────→
```

多个 channel 并行，每个 channel 处理数据的 1/nChannels，充分利用所有 NIC 带宽。

### 9.3 Proxy 线程实现

对标 NCCL 的 `proxy.cc`：

```cpp
// src/proxy.cc
namespace uccl {

struct ucclProxyOp {
    int channelId;
    int peer;                       // 远程 rank
    size_t nbytes;
    size_t chunkSize;
    int nsteps;                     // 传输步数
    int protocol;                   // SIMPLE / LL128
    struct ucclConnInfo* connection; // transport 连接
};

struct ucclProxyState {
    std::thread progressThread;     // proxy 进度线程
    volatile int stop;              // 停止标志
    volatile int* abortFlag;        // 中断标志

    // 网络 plugin 引用
    ucclNet_t* net;
    void* netContext;
};

// Proxy 进度函数：在专用线程上运行
// 轮询 FIFO buffer，调用 ucclNet->isend/irecv 处理网络 I/O
void* ucclProxyProgress(void* state);

// 启动/停止 proxy 线程
ucclResult_t ucclProxyCreate(ucclComm_t comm);
ucclResult_t ucclProxyDestroy(ucclComm_t comm);

} // namespace uccl
```

### 9.4 GPU Kernel 与 Proxy 的同步机制

通过 **host-visible FIFO buffer** + **head/tail 指针** 进行生产者-消费者同步：

```cpp
struct ucclConnFifo {
    volatile uint64_t head;      // proxy 更新（已接收/可读）
    volatile uint64_t tail;      // GPU kernel 更新（已写入/可发送）
    void* buffs[UCCL_STEPS];     // 环形 buffer 槽位
    size_t sizes[UCCL_STEPS];    // 每个槽位的数据大小
};

#define UCCL_STEPS 8  // 对标 NCCL NCCL_STEPS
```

- **GPU kernel 发送方**：写数据到 `buffs[tail % STEPS]`，更新 `tail++`，等待 `head` 追上再写下一个
- **Proxy 发送方**：轮询 `tail`，发现新数据后调用 `net->isend()`，完成后更新 `head++`
- **Proxy 接收方**：调用 `net->irecv()` 接收数据到 `buffs`，更新 `tail++`
- **GPU kernel 接收方**：轮询 `tail`，发现新数据后读取并 reduce

这个 buffer 必须用 `sycl::malloc_host()` 分配（pinned memory），确保 GPU 和 CPU 都可以访问。

---

## 10. Bootstrap（进程管理）

### 10.1 使用 MPI 作为 Bootstrap

相比 NCCL 自己实现 TCP socket bootstrap，我们直接使用 **MPI** 作为多进程管理层：

| 功能 | TCP Bootstrap (NCCL) | MPI Bootstrap (我们) |
|------|---------------------|---------------------|
| Rank 发现 | 自实现 TCP 服务器 | `MPI_Comm_rank()` / `MPI_Comm_size()` |
| UniqueId 广播 | TCP 手动分发 | `MPI_Bcast()` |
| Barrier | 自实现 | `MPI_Barrier()` |
| AllGather 元数据 | TCP | `MPI_Allgather()` |
| 进程启动 | 用户负责 | `mpirun -n N ./app` |
| 代码复杂度 | ~500 行 TCP 代码 | ~50 行 MPI 调用 |

**优势：**
- 大幅简化 bootstrap 实现（不需要自己实现 TCP 服务器、连接管理、序列化）
- MPI 在 HPC 环境中无处不在，Intel MPI 与 Intel GPU 工具链深度集成
- 多节点启动直接使用 `mpirun`，用户体验一致

```cpp
// src/bootstrap.cc
#include <mpi.h>
#include "uccl.h"

namespace uccl {

ucclResult_t bootstrapInit(ucclComm_t comm) {
    MPI_Comm_rank(MPI_COMM_WORLD, &comm->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm->nRanks);
    return ucclSuccess;
}

ucclResult_t bootstrapBcastUniqueId(ucclUniqueId* id, int root) {
    MPI_Bcast(id, sizeof(ucclUniqueId), MPI_BYTE, root, MPI_COMM_WORLD);
    return ucclSuccess;
}

ucclResult_t bootstrapAllGather(void* sendbuf, void* recvbuf, size_t size) {
    MPI_Allgather(sendbuf, size, MPI_BYTE,
                  recvbuf, size, MPI_BYTE, MPI_COMM_WORLD);
    return ucclSuccess;
}

ucclResult_t bootstrapBarrier() {
    MPI_Barrier(MPI_COMM_WORLD);
    return ucclSuccess;
}

} // namespace uccl
```

---

## 11. Transport 层

### 11.1 P2P Transport（Intra-node）

单机内 GPU 间通信，两种实现路径：

1. **SYCL USM (Unified Shared Memory)**：
   - `sycl::malloc_device()` 分配 device memory
   - `queue.memcpy()` 或直接指针访问进行 P2P 传输
   - 依赖 Intel 运行时的 P2P 支持

2. **Level Zero IPC**：
   - `zeMemGetIpcHandle()` / `zeMemOpenIpcHandle()` 实现跨进程 GPU 内存映射
   - 性能最优路径，用于进程间的 intra-node 通信

### 11.2 Net Transport（Inter-node）

跨节点通信，通过 `ucclNet_t` plugin 接口调用：

- `net.cc` 实现 Transport 的 connect/send/recv 逻辑
- 将 transport 操作转发到已加载的 network plugin
- 支持 GPU Direct RDMA（plugin 通过 `regMr` 注册 GPU 内存）
- **Proxy 线程** 调用 plugin 的 `isend()`/`irecv()`/`test()` 完成异步网络传输

---

## 12. CUDA → SYCL API 映射

NCCL 大量使用 CUDA 特有的 API。以下是实现 Unified-CCL（尤其是 LL128 protocol）所需的关键 CUDA API 及其 SYCL/Level Zero 对应：

### 12.1 Device Kernel 执行

| CUDA API | 用途 (NCCL) | SYCL 对应 | 说明 |
|----------|------------|-----------|------|
| `cudaLaunchKernel` / `<<<>>>` | 启动 collective kernel | `queue.submit()` + `parallel_for` | SYCL kernel 提交 |
| `cudaStream_t` | 异步流 | `sycl::queue` | 作为 stream 参数传入 API |
| `cudaStreamSynchronize` | 同步 | `queue.wait()` | |
| `cudaEventCreate/Record/Synchronize` | 事件同步 | `sycl::event` | `queue.submit()` 返回 event |

### 12.2 内存管理

| CUDA API | 用途 (NCCL) | SYCL/L0 对应 | 说明 |
|----------|------------|-------------|------|
| `cudaMalloc` | Device 内存分配 | `sycl::malloc_device()` | |
| `cudaFree` | 释放 | `sycl::free()` | |
| `cudaMemcpy` | 内存拷贝 | `queue.memcpy()` | |
| `cudaMemcpyAsync` | 异步拷贝 | `queue.memcpy()` | SYCL 默认异步 |
| `cudaHostAlloc` / `cudaMallocHost` | Pinned host 内存 | `sycl::malloc_host()` | |
| `cudaIpcGetMemHandle` | IPC 内存句柄导出 | `zeMemGetIpcHandle()` | **需要 Level Zero** |
| `cudaIpcOpenMemHandle` | IPC 内存句柄导入 | `zeMemOpenIpcHandle()` | **需要 Level Zero** |
| `cudaMemPool_t` | 内存池 | 无直接对应 | 自行实现或使用 L0 |

### 12.3 Warp 级原语（LL128 核心依赖）

| CUDA API | 用途 (NCCL LL128) | SYCL 对应 | 说明 |
|----------|------------------|-----------|------|
| `__syncwarp()` | Warp 内同步 | `sycl::group_barrier(sub_group)` | SYCL sub-group barrier |
| `__any_sync(mask, pred)` | Warp vote: 任一为真 | `sycl::any_of_group(sub_group, pred)` | |
| `__shfl_sync(mask, val, lane)` | Warp shuffle | `sycl::select_from_group(sub_group, val, lane)` | Intel GPU sub-group 宽度通常 16/32 |
| `__funnelshift_r(lo, hi, shift)` | 双字右移 | 手动实现: `(lo >> shift) \| (hi << (32-shift))` | 无直接对应 |
| `WARP_SIZE` (32) | Warp 宽度 | `sub_group.get_local_range()[0]` | Intel GPU 可能是 16 或 32 |

### 12.4 128-bit Load/Store（LL128 核心依赖）

| CUDA API / PTX | 用途 (NCCL op128.h) | SYCL 对应 | 说明 |
|----------------|---------------------|-----------|------|
| `ld.volatile.global.v2.u64` | 128-bit volatile load | `sycl::vec<uint64_t, 2>` load + `volatile` | 或使用 `sycl::atomic_ref` |
| `st.volatile.global.v2.u64` | 128-bit volatile store | `sycl::vec<uint64_t, 2>` store | |
| `ld.volatile.shared.v2.u64` | 128-bit shared mem load | `sycl::local_accessor` + vec load | SLM (Shared Local Memory) |
| `st.volatile.shared.v2.u64` | 128-bit shared mem store | `sycl::local_accessor` + vec store | |
| `ld.relaxed.gpu.global.u64` | Relaxed atomic load | `sycl::atomic_ref<memory_order::relaxed>` | |
| `st.release.sys.global.u64` | Release store | `sycl::atomic_ref<memory_order::release>` | |
| `ld.acquire.sys.global.u64` | Acquire load | `sycl::atomic_ref<memory_order::acquire>` | |
| `fence.acq_rel.sys` | System fence | `sycl::atomic_fence(memory_order::acq_rel, memory_scope::system)` | |
| `fence.acq_rel.gpu` | GPU fence | `sycl::atomic_fence(memory_order::acq_rel, memory_scope::device)` | |

### 12.5 Thread Block 同步

| CUDA API | 用途 (NCCL) | SYCL 对应 | 说明 |
|----------|------------|-----------|------|
| `__syncthreads()` | Block 同步 | `sycl::group_barrier(work_group)` | |
| `barrier_sync(id, nthreads)` | Named barrier | `sycl::group_barrier(sub_group)` | SYCL 无 named barrier，需用 sub-group 或 work-group barrier 替代 |
| `__threadfence()` | GPU memory fence | `sycl::atomic_fence(memory_order::seq_cst, memory_scope::device)` | |
| `__threadfence_system()` | System memory fence | `sycl::atomic_fence(memory_order::seq_cst, memory_scope::system)` | |

### 12.6 Shared Memory

| CUDA API | 用途 (NCCL) | SYCL 对应 | 说明 |
|----------|------------|-----------|------|
| `__shared__` | 共享内存声明 | `sycl::local_accessor` | SLM - Shared Local Memory |
| `cvta.to.shared.u64` | Generic → shared 指针转换 | 无需要 | SYCL local_accessor 直接访问 |

### 12.7 Device 查询与管理

| CUDA API | 用途 (NCCL) | SYCL/L0 对应 | 说明 |
|----------|------------|-------------|------|
| `cudaGetDeviceCount` | GPU 数量 | `sycl::device::get_devices()` | |
| `cudaSetDevice` | 选择 GPU | 创建对应 device 的 `sycl::queue` | |
| `cudaGetDeviceProperties` | GPU 属性 | `device.get_info<>()` | |
| `cudaDeviceCanAccessPeer` | P2P 可达性 | `zeDeviceCanAccessPeer()` | **需要 Level Zero** |
| `cudaDeviceEnablePeerAccess` | 启用 P2P | `zeContextMakeMemoryResident()` | **需要 Level Zero** |

### 12.8 需要 Level Zero 的场景汇总

以下场景 SYCL 标准 API 不足，**必须** 使用 Level Zero：

1. **IPC 内存共享** — `zeMemGetIpcHandle()` / `zeMemOpenIpcHandle()`
   - 多进程 intra-node 通信的基础
2. **P2P 可达性查询** — `zeDeviceCanAccessPeer()`
3. **PCI 拓扑探测** — `zeDevicePciGetPropertiesExt()` 获取 GPU 的 PCI BDF 地址
4. **细粒度内存控制** — 如需要显式控制 cache policy、memory residency

---

## 13. Primitives 通信原语

对标 NCCL 的 `primitives.h`，所有 Protocol 都实现统一的原语接口：

```cpp
// device/primitives.hpp
template <typename T, typename RedOp, typename Proto>
class Primitives {
public:
    // 发送 input buffer 中的数据
    void send(intptr_t inpIx, int eltN);

    // 从 output buffer 发送
    void sendFromOutput(intptr_t outIx, int eltN);

    // 接收到 output buffer
    void recv(intptr_t outIx, int eltN, bool postOp = false);

    // 接收 + reduce + 发送（ring reduce-scatter 核心）
    void recvReduceSend(intptr_t inpIx, int eltN);

    // 接收 + reduce + copy 到 output + 发送
    void recvReduceCopySend(intptr_t inpIx, intptr_t outIx,
                            int eltN, bool postOp = false);

    // 接收 + copy 到 output + 发送（ring allgather 核心）
    void recvCopySend(intptr_t outIx, int eltN, bool postOp = false);

    // copy input 到 output + 发送
    void copySend(intptr_t inpIx, intptr_t outIx,
                  int eltN, bool postOp = false);
};
```

Algorithm 层通过这些原语编排数据流动，完全不关心底层是 Simple 还是 LL128。

---

## 14. SYCL Reduce Kernel

```cpp
// device/reduce_kernel.hpp
#pragma once
#include <sycl/sycl.hpp>

namespace uccl {

// bf16 sum reduce
template <typename T>
struct ReduceSum {
    T operator()(T a, T b) const { return a + b; }
};

// 对标 NCCL applyReduce / applyPreOp / applyPostOp
template <typename T, typename RedOp>
inline T applyReduce(RedOp op, T a, T b) {
    return op(a, b);
}

/* 支持的类型:
 *   sycl::half     — fp16
 *   sycl::ext::oneapi::bfloat16 — bf16
 */

} // namespace uccl
```

---

## 15. Python API

### 15.1 Python 接口设计

```python
# uccl/uccl.py
import torch
import _uccl_bindings  # pybind11 module

class Communicator:
    """UCCL communicator wrapping ucclComm_t."""

    def __init__(self, rank: int, world_size: int):
        """Initialize communicator for given rank."""
        self._comm = _uccl_bindings.comm_init_rank(world_size, rank)
        self._rank = rank
        self._world_size = world_size

    def allreduce(self, tensor: torch.Tensor,
                  op: str = "sum") -> torch.Tensor:
        """
        In-place allreduce on a torch.Tensor (GPU).

        Args:
            tensor: fp16 or bf16 tensor on Intel GPU
            op: reduction operation ("sum")

        Returns:
            The same tensor, modified in-place.
        """
        assert tensor.dtype in (torch.float16, torch.bfloat16), \
            f"Unsupported dtype: {tensor.dtype}"
        assert tensor.is_xpu, "Tensor must be on Intel GPU (xpu device)"

        _uccl_bindings.allreduce(
            self._comm,
            tensor.data_ptr(),
            tensor.data_ptr(),   # in-place: sendbuff == recvbuff
            tensor.numel(),
            tensor.dtype,
            op
        )
        return tensor

    def allgather(self, send_tensor: torch.Tensor,
                   recv_tensor: torch.Tensor) -> torch.Tensor:
        """
        AllGather: gather data from all ranks.

        Args:
            send_tensor: local data tensor on Intel GPU
            recv_tensor: output tensor with size = send_tensor.numel() * world_size

        Returns:
            recv_tensor with gathered data from all ranks.
        """
        assert send_tensor.dtype in (torch.float16, torch.bfloat16)
        assert recv_tensor.dtype == send_tensor.dtype
        assert send_tensor.is_xpu and recv_tensor.is_xpu
        assert recv_tensor.numel() == send_tensor.numel() * self._world_size

        _uccl_bindings.allgather(
            self._comm,
            send_tensor.data_ptr(),
            recv_tensor.data_ptr(),
            send_tensor.numel(),
            send_tensor.dtype
        )
        return recv_tensor

    def reduce_scatter(self, send_tensor: torch.Tensor,
                       recv_tensor: torch.Tensor,
                       op: str = "sum") -> torch.Tensor:
        """
        ReduceScatter: reduce then scatter across ranks.

        Args:
            send_tensor: input tensor with size = recv_tensor.numel() * world_size
            recv_tensor: output tensor for local chunk
            op: reduction operation ("sum")

        Returns:
            recv_tensor with reduced-scattered result.
        """
        assert send_tensor.dtype in (torch.float16, torch.bfloat16)
        assert recv_tensor.dtype == send_tensor.dtype
        assert send_tensor.is_xpu and recv_tensor.is_xpu
        assert send_tensor.numel() == recv_tensor.numel() * self._world_size

        _uccl_bindings.reduce_scatter(
            self._comm,
            send_tensor.data_ptr(),
            recv_tensor.data_ptr(),
            recv_tensor.numel(),
            send_tensor.dtype,
            op
        )
        return recv_tensor

    def destroy(self):
        """Finalize and destroy communicator."""
        if self._comm is not None:
            _uccl_bindings.comm_finalize(self._comm)
            _uccl_bindings.comm_destroy(self._comm)
            self._comm = None

    def __del__(self):
        self.destroy()
```

### 15.2 pybind11 绑定

```cpp
// bindings/python/uccl_bindings.cc
#include <pybind11/pybind11.h>
#include "uccl.h"

namespace py = pybind11;

PYBIND11_MODULE(_uccl_bindings, m) {
    m.doc() = "Unified-CCL Python bindings";

    m.def("comm_init_rank", [](int nranks, int rank) {
        ucclUniqueId id;
        if (rank == 0) ucclGetUniqueId(&id);
        // broadcast id to all ranks (user responsibility or via bootstrap)
        ucclComm_t comm;
        ucclResult_t res = ucclCommInitRank(&comm, nranks, id, rank);
        if (res != ucclSuccess)
            throw std::runtime_error(ucclGetErrorString(res));
        return reinterpret_cast<uintptr_t>(comm);
    });

    m.def("allreduce", [](uintptr_t comm_handle,
                          uintptr_t sendbuff, uintptr_t recvbuff,
                          size_t count, py::object dtype, std::string op) {
        ucclComm_t comm = reinterpret_cast<ucclComm_t>(comm_handle);

        ucclDataType_t dt = ucclFloat16;  // resolve from torch dtype
        // ... dtype mapping logic ...

        ucclResult_t res = ucclAllReduce(
            reinterpret_cast<const void*>(sendbuff),
            reinterpret_cast<void*>(recvbuff),
            count, dt, ucclSum, comm, nullptr /* default stream */);
        if (res != ucclSuccess)
            throw std::runtime_error(ucclGetErrorString(res));
    });

    m.def("allgather", [](uintptr_t comm_handle,
                           uintptr_t sendbuff, uintptr_t recvbuff,
                           size_t sendcount, py::object dtype) {
        ucclComm_t comm = reinterpret_cast<ucclComm_t>(comm_handle);
        ucclDataType_t dt = ucclFloat16;  // resolve from torch dtype
        ucclResult_t res = ucclAllGather(
            reinterpret_cast<const void*>(sendbuff),
            reinterpret_cast<void*>(recvbuff),
            sendcount, dt, comm, nullptr);
        if (res != ucclSuccess)
            throw std::runtime_error(ucclGetErrorString(res));
    });

    m.def("reduce_scatter", [](uintptr_t comm_handle,
                               uintptr_t sendbuff, uintptr_t recvbuff,
                               size_t recvcount, py::object dtype, std::string op) {
        ucclComm_t comm = reinterpret_cast<ucclComm_t>(comm_handle);
        ucclDataType_t dt = ucclFloat16;  // resolve from torch dtype
        ucclResult_t res = ucclReduceScatter(
            reinterpret_cast<const void*>(sendbuff),
            reinterpret_cast<void*>(recvbuff),
            recvcount, dt, ucclSum, comm, nullptr);
        if (res != ucclSuccess)
            throw std::runtime_error(ucclGetErrorString(res));
    });

    m.def("comm_finalize", [](uintptr_t h) {
        ucclCommFinalize(reinterpret_cast<ucclComm_t>(h));
    });
    m.def("comm_destroy", [](uintptr_t h) {
        ucclCommDestroy(reinterpret_cast<ucclComm_t>(h));
    });
}
```

---

## 16. 内部核心数据结构

对标 NCCL 的 `ncclComm`：

```cpp
// src/include/comm.h
struct ucclComm {
    int rank;
    int nRanks;
    int nNodes;
    int localRank;
    int localRanks;

    // SYCL device & queue
    sycl::device device;
    sycl::queue* defaultQueue;

    // Topology
    struct ucclTopology* topo;

    // Channels — 数据并行通道
    struct ucclChannel channels[UCCL_MAX_CHANNELS];

    // Transport connections
    struct ucclPeerInfo* peerInfo;

    // Network plugin
    ucclNet_t* net;
    void* netContext;

    // Proxy state
    struct ucclProxyState* proxyState;

    // Bootstrap (MPI)
    MPI_Comm mpiComm;

    // Buffer sizes per protocol
    size_t buffSizes[UCCL_NUM_PROTOCOLS];

    // Error state
    ucclResult_t asyncError;
    volatile int abortFlag;
};
```

---

## 17. MVP 范围与后续扩展

| 组件 | MVP | 后续扩展 |
|------|-----|---------|
| Algorithm | Ring, One-Shot, Symmetric Memory | Tree, CollNet |
| Protocol | Simple, LL128 | LL |
| Transport | P2P (intra-node), Net (inter-node) | — |
| Network Plugin | 内置 UCX | 外部 plugin 动态加载 (.so) |
| Data Type | bf16, fp16 | fp32, fp8, int8 |
| Collective | AllReduce, AllGather, ReduceScatter (sum) | Broadcast, Send/Recv |
| Reduction Op | Sum | Prod, Max, Min, Avg |

---

## 18. 构建系统

### CMake 结构

```cmake
# 顶层 CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(unified-ccl LANGUAGES CXX)

# 要求 Intel DPC++ 编译器
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

# 依赖
find_package(IntelSYCL REQUIRED)
find_package(MPI REQUIRED)     # Intel MPI
find_package(pybind11 REQUIRED)
find_package(UCX REQUIRED)     # libucp, libuct, libucs

# 核心库
add_subdirectory(src)

# Python 绑定
add_subdirectory(bindings/python)

# 测试
enable_testing()
add_subdirectory(tests)

# 外部 plugin（可选）
add_subdirectory(plugins/ucx)
```

### 编译产物

- `libuccl.so` — 核心 C/C++ 库
- `_uccl_bindings.so` — pybind11 Python 模块
- `libuccl-net-ucx.so` — UCX network plugin（可内置或外部）

---

## 19. 测试策略

### Python 测试

```python
# tests/python/test_allreduce.py
import torch
import uccl

def test_allreduce_fp16():
    comm = uccl.Communicator(rank=0, world_size=1)
    t = torch.ones(1024, dtype=torch.float16, device="xpu")
    result = comm.allreduce(t)
    assert torch.allclose(result, torch.ones_like(result))
    comm.destroy()

def test_allreduce_bf16():
    comm = uccl.Communicator(rank=0, world_size=1)
    t = torch.ones(1024, dtype=torch.bfloat16, device="xpu")
    result = comm.allreduce(t)
    assert torch.allclose(result, torch.ones_like(result))
    comm.destroy()

def test_allreduce_multi_gpu():
    """Multi-process test using torch.multiprocessing."""
    # 使用 mp.spawn 启动多进程测试
    pass
```

### C++ 测试

```cpp
// tests/cpp/test_allreduce.cc
// 对标 nccl-tests 的 all_reduce_perf
// 测试不同 message size 下的正确性和带宽
```

topo check的时候，请注意存在有一种场景是单机内多卡之间是PCIE 链接，然后cross sockets会cross-upi。