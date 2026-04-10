# GPU RDMA Proxy Design Spec

**Date**: 2026-04-10  
**Status**: Draft  
**Scope**: unified-ccl proxy 实现方案，支持跨 NIC 的 Ring AllReduce 异步执行和细粒度 Device RDMA API

---

## 1. 目标

1. **Ring AllReduce 跨 NIC 异步化**：collective API 提交到 SYCL queue 后立即返回，网络传输由 proxy 线程后台驱动，stream ordering 保证后续 kernel 看到完整结果
2. **细粒度 Device RDMA API**：暴露 `put/signal/wait` 等 device-side 接口，通过 host proxy 完成实际 RDMA，让用户可以组合这些原语写自定义 collective
3. **统一的 P2P/Net 抽象**：Device API 对用户透明——同节点走 IPC 直接 copy，跨节点走 FIFO → proxy → UCX

---

## 2. 选定方案：统一 FIFO 转发（方案 A）

所有跨 NIC 操作通过 **channel-per-peer 的 pinned host memory FIFO** 传递给 proxy 线程。FIFO entry 用 tagged union 区分 collective 操作和 device API 操作。

备选方案 B（双层 FIFO）和方案 C（CTRAN 风格 KernelElem + GPE）已记录在 `desgin/proxy.md` 第 10 节，作为未来演进参考。

---

## 3. 核心约束

| 约束 | 值 |
|------|-----|
| 目标硬件 | Intel B60 GPU, 160 Xe Cores |
| 编程模型 | SYCL (DPC++), Level Zero fallback |
| 网络栈 | UCX (ucp) via ucclNet_t plugin interface |
| Bootstrap | MPI |
| Host blocking | **零**（kernel launch 后全异步） |
| Collective kernel EU 占用 | 1 workgroup = 1 Xe Core (0.6%) |

---

## 4. 架构总览

```
┌───────────────────────────────────────────────────┐
│              User Code                            │
│  ucclAllReduce(sendbuf, recvbuf, count, comm, q)  │
│  ucclPut / ucclSignal / ucclWait (device-side)    │
└───────────────┬───────────────────────────────────┘
                │
        ┌───────┴────────┐
        │ Transport      │
        │ Selection      │
        │ (same node?)   │
        └───┬────────┬───┘
            │        │
    ┌───────▼──┐ ┌───▼──────────────────────────────┐
    │ P2P Path │ │ Net Path                          │
    │          │ │                                    │
    │ IPC/USM  │ │ Persistent Kernel                  │
    │ direct   │ │   ├─ copy → bounce buffer          │
    │ copy +   │ │   ├─ write FIFO entry              │
    │ reduce   │ │   ├─ fence + tail++                │
    │          │ │   └─ poll head/done (overlap reduce)│
    │ (no FIFO │ │                    │                │
    │  no proxy│ │              ┌─────▼──────┐        │
    │  needed) │ │              │ FIFO       │        │
    └──────────┘ │              │ (pinned    │        │
                 │              │  host mem) │        │
                 │              └─────┬──────┘        │
                 │                    │               │
                 │              ┌─────▼──────┐        │
                 │              │ Proxy      │        │
                 │              │ Thread     │        │
                 │              │  ├─ poll   │        │
                 │              │  ├─ isend  │        │
                 │              │  ├─ irecv  │        │
                 │              │  ├─ test   │        │
                 │              │  └─ progress        │
                 │              └─────┬──────┘        │
                 │                    │               │
                 │              ┌─────▼──────┐        │
                 │              │ UCX        │        │
                 │              │ (ucp)      │        │
                 │              └────────────┘        │
                 └────────────────────────────────────┘
```

---

## 5. 单机 vs 跨机 AllReduce：分离的 Kernel 实现

AllReduce 根据 transport 类型选择不同的 kernel 实现：

### 5.1 单机 AllReduce（P2P Kernel）

- **不经过 FIFO 和 proxy**
- 直接通过 Level Zero IPC 映射读写 peer GPU 内存
- Kernel 内使用 `queue.memcpy` 或 device-side pointer load/store
- 与现有 `p2pTransportSend/Recv` 对齐
- 可用算法：Ring (P2P)、Symmetric Memory、One-Shot

```cpp
// 单机 Ring AllReduce kernel — 直接 P2P
void allReduceRingP2P(nd_item<1> item, ...) {
    for (int step = 0; step < totalSteps; step++) {
        // 直接从 peer 的 IPC 映射地址读取
        T* peerBuf = (T*)ipcRemoteBufs[prevRank];
        // reduce: output[i] += peerBuf[i]
        cooperativeReduce(item, peerBuf + recvOff, outputBuf + recvOff, eltN);
        // 直接写到 peer 的 IPC 映射地址
        cooperativeCopy(item, outputBuf + sendOff, 
                        ipcRemoteBufs[nextRank] + sendOff, nbytes);
        // P2P fence — 比 system fence 轻量
        sycl::atomic_fence(memory_order::release, memory_scope::device);
        // 用 device-side atomic 同步（不需要 host 介入）
        deviceBarrier(item, syncFlags, step);
    }
}
```

**特点**：零 host 介入，零网络开销，纯 GPU 计算

### 5.2 跨机 AllReduce（Net Kernel）

- **经过 FIFO → proxy → UCX**
- Persistent kernel，1 workgroup
- 使用 bounce buffer 隔离 kernel 写入和 NIC DMA
- 通过 system-scope fence 保证 device store 对 host 可见

```cpp
// 跨机 Ring AllReduce kernel — 通过 FIFO 和 proxy
void allReduceRingNet(nd_item<1> item, ...) {
    for (int step = 0; step < totalSteps; step++) {
        // 1. workgroup 协作 copy → bounce buffer
        cooperativeCopy(item, sendData, bounceSend[slot], nbytes);
        item.barrier();

        // 2. 提交 FIFO entry (send)
        if (lid == 0) {
            writeSendFifoEntry(sendFifo, slot, bounceSend[slot], nbytes, mhandle);
        }

        // 3. 提交 FIFO entry (recv)
        if (lid == 0) {
            writeRecvFifoEntry(recvFifo, slot, bounceRecv[slot], nbytes, mhandle);
        }

        // 4. overlap: reduce 上一步 recv 数据
        if (step > 0) {
            cooperativeReduce(item, prevBounceRecv, outputBuf + prevOff, prevNbytes);
        }

        // 5. poll recv done
        if (lid == 0) {
            while (recvFifo->entries[slot].done == 0) {}
        }
        item.barrier();

        // 6. copy recv 数据到 output (最后一步)
        // ...
    }
}
```

**特点**：1 Xe Core 占用，compute/communication overlap，零 host blocking

### 5.3 Dispatch 逻辑

```cpp
sycl::event ucclAllReduce(void* sendbuf, void* recvbuf, size_t count,
                           ucclDataType_t datatype, ucclRedOp_t op,
                           ucclComm_t comm, sycl::queue& queue) {
    if (comm->nNodes == 1) {
        // 单机：选择 P2P kernel
        return allReduceRingP2P_launch(sendbuf, recvbuf, count, comm, queue);
    } else {
        // 跨机：选择 Net kernel
        return allReduceRingNet_launch(sendbuf, recvbuf, count, comm, queue);
    }
    // 未来：混合场景（intra-node reduce + inter-node allreduce）
}
```

---

## 6. FIFO 数据结构

### 6.1 FIFO Entry

```cpp
enum ucclFifoOpType : uint32_t {
    UCCL_OP_SEND,     // collective: 发送 bounce buffer 数据
    UCCL_OP_RECV,     // collective: 接收数据到 bounce buffer
    UCCL_OP_PUT,      // device API: 零拷贝 RDMA 用户 buffer
    UCCL_OP_SIGNAL,   // device API: 零字节通知远端
    UCCL_OP_WAIT,     // device API: 等待远端 signal
};

struct ucclFifoEntry {
    ucclFifoOpType opType;
    void* buff;              // collective: bounce buf; device API: user buf
    size_t size;
    int peer;
    void* remoteBuff;        // PUT: 远端虚拟地址
    void* mhandle;           // NIC 注册 handle
    volatile int done;       // proxy → kernel 完成通知
    void* request;           // proxy 内部: async request handle
};
```

### 6.2 FIFO 本体

```cpp
#define UCCL_STEPS 8

struct ucclConnFifo {
    volatile uint64_t head;              // proxy 更新（kernel 读，判断 slot 释放）
    volatile uint64_t tail;              // kernel 更新（proxy 读，发现新 entry）
    uint64_t pendingHead;                // proxy 内部（已发起但未完成的位置）
    ucclFifoEntry entries[UCCL_STEPS];
};
```

- 分配方式：`sycl::malloc_host`（pinned host memory，GPU 可直接 load/store）
- 每个 channel 双向各一个：`sendFifo` 和 `recvFifo`
- Ring buffer 语义：`tail - head < UCCL_STEPS` 时有可用 slot
- `pendingHead` 仅 proxy 线程使用（不需要 volatile），跟踪已 post 但未完成的位置

---

## 7. Proxy Thread

### 7.1 主循环

```cpp
void proxyProgressFunc(ucclProxyState* state) {
    ucclComm* comm = state->comm;
    ucclNet_t* net = comm->net;

    while (!state->exitFlag) {
        bool anyWork = false;

        for (int ch = 0; ch < comm->nChannels; ch++) {
            anyWork |= processFifo(net, &comm->channels[ch].sendFifo,
                                   comm->channels[ch].sendComm, /*isSend=*/true);
            anyWork |= processFifo(net, &comm->channels[ch].recvFifo,
                                   comm->channels[ch].recvComm, /*isSend=*/false);
        }

        net->progress(comm->netContext);

        if (!anyWork) std::this_thread::yield();
    }
}
```

### 7.2 FIFO 处理

```cpp
bool processFifo(ucclNet_t* net, ucclConnFifo* fifo,
                 void* netComm, bool isSend) {
    bool worked = false;
    uint64_t pending = fifo->pendingHead;

    while (pending < fifo->tail) {
        int slot = pending % UCCL_STEPS;
        ucclFifoEntry* entry = &fifo->entries[slot];

        if (entry->request == nullptr) {
            // 新 entry: 发起网络操作
            switch (entry->opType) {
            case UCCL_OP_SEND:
            case UCCL_OP_PUT:
                net->isend(netComm, entry->buff, entry->size,
                           entry->mhandle, &entry->request);
                break;
            case UCCL_OP_RECV:
            case UCCL_OP_WAIT:
                net->irecv(netComm, 1, &entry->buff, &entry->size,
                           entry->mhandle, &entry->request);
                break;
            case UCCL_OP_SIGNAL:
                net->isend(netComm, nullptr, 0, nullptr, &entry->request);
                break;
            }
            worked = true;
        } else {
            // 已发起: 检查完成
            int done = 0;
            net->test(entry->request, &done, nullptr);
            if (done) {
                entry->done = 1;
                entry->request = nullptr;
                fifo->pendingHead = ++pending;
                fifo->head = pending;
                worked = true;
            } else {
                break;  // 顺序完成，当前未完成则停止
            }
        }
        pending++;
    }
    return worked;
}
```

### 7.3 线程生命周期

- `ucclCommInit` 时启动（仅当 `nNodes > 1` 或有 net plugin 时）
- `ucclCommDestroy` 时设 `exitFlag = true`，`thread.join()` 等待退出
- 单线程服务所有 channel，CPU 占用 1 个核心

---

## 8. Device RDMA API

### 8.1 Host-side 资源管理

```cpp
// 注册 GPU buffer（跨节点 regMr + 同节点 IPC export）
ucclResult_t ucclRegMem(void* buf, size_t size, ucclComm_t comm,
                        ucclMemHandle_t* handle);
ucclResult_t ucclDeregMem(ucclMemHandle_t handle);

// 交换远端 buffer 信息（MPI_Allgather rkey + addr + IPC handle）
ucclResult_t ucclExchangeMem(ucclMemHandle_t localHandle,
                             ucclRemoteMemHandle_t* remoteHandles,
                             ucclComm_t comm);
```

### 8.2 Device-side 操作原语

```cpp
// RDMA PUT: 同节点 → IPC copy; 跨节点 → FIFO → proxy → UCX (零拷贝)
SYCL_EXTERNAL void ucclPut(void* localBuf, size_t offset, size_t size,
                           int peer, ucclDeviceComm_t devComm);

// SIGNAL: 同节点 → atomic flag; 跨节点 → FIFO → proxy → 零字节 send
SYCL_EXTERNAL void ucclSignal(int peer, ucclDeviceComm_t devComm);

// WAIT: 同节点 → poll atomic flag; 跨节点 → FIFO → proxy → irecv
SYCL_EXTERNAL void ucclWait(int peer, ucclDeviceComm_t devComm);
```

### 8.3 ucclDeviceComm 结构

```cpp
struct ucclDeviceComm {
    int rank, nRanks;
    ucclConnFifo* fifo;              // 指向 channel 的 send FIFO
    bool* peerIsLocal;               // peerIsLocal[peer]: 同节点?
    struct {
        void* ipcAddr;               // 同节点: IPC 映射地址
        void* addr;                  // 跨节点: 远端虚拟地址
        void* localMhandle;          // 本地 regMr handle
    }* remotes;                      // remotes[peer]
    volatile int* signalFlags;       // 同节点 signal 用
    volatile int* waitFlags;         // 同节点 wait 用
};
```

分配方式：`sycl::malloc_host`，kernel 直接访问。

### 8.4 Transport 透明性

Device API 内部根据 `peerIsLocal[peer]` 自动选择路径：
- 同节点：直接 GPU-to-GPU 操作，不经过 FIFO 和 proxy
- 跨节点：写 FIFO entry → proxy 驱动 UCX → 完成后设 `entry->done`

用户无需关心 peer 在哪个节点。

---

## 9. Memory Registration

### 9.1 Bounce Buffer（Collective 专用）

- `ucclCommInit` 时预分配，每个 channel × UCCL_STEPS 个 slot
- `sycl::malloc_device` 分配 + `net->regMr` 注册
- 大小：sliceSize = buffSize / (slicesPerChunk × stepsPerSlice) = 512KB (Simple protocol)
- 总量：8 slots × 512KB × 2 (send + recv) = 8MB per channel
- 用途：隔离 kernel reduce 写入和 NIC DMA 读取

### 9.2 User Buffer（Device API 专用）

- 用户显式调用 `ucclRegMem` 注册
- 跨节点 peer: `net->regMr()` → 缓存 mhandle
- 同节点 peer: `zeMemGetIpcHandle()` → 缓存 IPC mapping
- `ucclExchangeMem()` 通过 `MPI_Allgather` 交换信息
- NIC 直接从用户 buffer DMA（零拷贝）

### 9.3 生命周期

```
ucclCommInit:
  ├─ 分配 FIFO (malloc_host)
  ├─ 分配 bounce buffers (malloc_device) + regMr
  ├─ 分配 ucclDeviceComm (malloc_host)
  └─ 启动 proxy thread

ucclRegMem / ucclExchangeMem:    ← 用户 buffer 按需注册

ucclAllReduce / ucclPut:         ← 复用已注册资源

ucclDeregMem:                    ← 用户释放
ucclCommDestroy:
  ├─ 停止 proxy thread
  ├─ deregMr + free bounce buffers
  └─ free FIFO, devComm
```

---

## 10. 性能特征

| 指标 | 值 |
|------|-----|
| Host blocking | **零**（kernel launch 后全异步） |
| 单机 AllReduce | 纯 GPU P2P，无 FIFO/proxy 开销 |
| 跨机 AllReduce EU 占用 | 1 workgroup / 160 Xe Cores = **0.6%** |
| Collective bounce copy | 每步 1 次 device→device memcpy (workgroup 协作) |
| Device API (ucclPut) | **零拷贝**，NIC 直接 DMA 用户 buffer |
| PCIe poll 开销 | 与 reduce 计算交错，**延迟隐藏** |
| In-flight operations | UCCL_STEPS = 8 并行 |
| Memory registration | 初始化时一次 (bounce); 用户按需 (ucclRegMem) |

---

## 11. MVP 范围

### Phase 1: Ring AllReduce 跨 NIC
- [ ] FIFO 数据结构 (`ucclConnFifo`, `ucclFifoEntry`)
- [ ] Bounce buffer 分配 + regMr
- [ ] Proxy thread 主循环 (poll + isend/irecv + test + progress)
- [ ] Net kernel: persistent ring allreduce (reduce-scatter + allgather)
- [ ] P2P kernel: 单机 ring allreduce (独立实现，不经过 FIFO)
- [ ] Dispatch: 根据 nNodes 选择 P2P/Net kernel
- [ ] Stream-ordered 完成模型

### Phase 2: Device RDMA API
- [ ] `ucclRegMem` / `ucclDeregMem` / `ucclExchangeMem` host API
- [ ] `ucclDeviceComm` 结构分配与初始化
- [ ] `ucclPut` device-side 实现（同节点 IPC + 跨节点 FIFO）
- [ ] `ucclSignal` / `ucclWait` device-side 实现
- [ ] 用户示例：custom AllGather

---

## 12. 未来演进

- **方案 B 拆分**：当 FIFO entry tagged union 变得复杂时，拆为 collective FIFO + device command queue
- **方案 C 参考**：如果Intel GPU 支持类似 CUDA graph 的 capture/replay，考虑 KernelElem + GPE 模式
- **混合拓扑**：intra-node reduce (P2P) + inter-node allreduce (Net) 的 hierarchical 算法
- **多 channel 并行**：多个 channel 同时进行 ring steps，提升带宽利用率
- **LL128 协议**：在 Net kernel 中使用 LL128 替代 Simple，减少延迟
