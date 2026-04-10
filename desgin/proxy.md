# GPU RDMA Proxy 架构研究：Meta torchcomms/ctran

本文档基于 Meta 的 [torchcomms/ctran](https://github.com/meta-pytorch/torchcomms/tree/main/comms/ctran) 代码库，详细分析其 GPU RDMA Proxy 的设计与实现，为 unified-ccl 的跨 NIC 通信提供参考。

---

## 1. 整体架构概览

CTRAN (Collective Transport) 是 Meta 为 NCCL/RCCL 构建的模块化集合通信架构，核心设计理念是**将 GPU kernel 与网络 I/O 解耦**：

```
┌─────────────────────────────────────────────────────────────┐
│                     Algorithm Layer                         │
│  (AllReduce, AllGather, AllToAll, ReduceScatter, Broadcast) │
└──────────────────────┬──────────────────────────────────────┘
                       │ submit(opGroup, func, kernelConfig)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  GPE (GPU Proxy Engine)                      │
│  ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │  GPU Kernel   │◄──►│  GPE Thread (CPU proxy thread)   │   │
│  │  (on stream)  │    │  (dequeue cmd → run collective)  │   │
│  └──────────────┘    └──────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │ iput / isendCtrl / irecvCtrl / waitNotify
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Mapper Layer                              │
│    (Memory Registration, Export/Import, Request Tracking)    │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ┌─────────┐ ┌──────────┐ ┌──────────┐
     │CtranIb  │ │CtranNvl  │ │CtranSock │
     │(IB RDMA)│ │(NVLink)  │ │(Socket)  │
     └─────────┘ └──────────┘ └──────────┘
```

关键设计原则：
- **GPU kernel 不直接发起网络 I/O**：kernel 只负责本地计算（reduce、copy）和通过共享内存与 NVLink peer 通信
- **所有跨 NIC 的 RDMA 操作由 CPU GPE 线程驱动**：kernel 通过 flag/KernelElem 与 GPE 线程同步
- **一次性 RDMA (One-sided)**：使用 RDMA WRITE + IMM 而非 Send/Recv，避免远端 CPU 参与数据路径

---

## 2. GPE (GPU Proxy Engine) 详细机制

### 2.1 核心组件

GPE 由以下关键组件构成：

| 组件 | 作用 |
|------|------|
| `CtranGpe` | 公共接口，管理 GPE 线程生命周期 |
| `CtranGpe::Impl` | 内部实现，包含线程函数和命令队列 |
| `CtranGpeCmd` | 命令对象，封装 opGroup + func + kernelFlag |
| `KernelFlagPool` | cudaHostAlloc 分配的 flag 池，用于 kernel↔GPE 同步 |
| `KernelElemPool` | cudaHostAlloc 分配的 KernelElem 池，用于传递操作参数 |
| `GpeKernelSync` | cudaHostAlloc 分配的同步对象，支持多步协作 |

### 2.2 submit() 流程：从 collective 到 kernel+GPE

当算法层调用 `CtranGpe::submit()` 时，完整流程如下：

```
Algorithm (e.g., AllReduce Ring)
  │
  ├─ 1. 构建 opGroup (vector<OpElem>)
  │     - 每个 OpElem 包含 sendbuff/recvbuff/count/datatype 等参数
  │     - 以及 KernelElem* 指针（GPE↔kernel 的共享控制结构）
  │
  ├─ 2. 构建 KernelConfig
  │     - 指定 kernel 类型、numBlocks、numThreads、stream
  │     - 设置 CtranKernelArgs（包含 devState_d 指针）
  │
  └─ 3. 调用 gpe->submit(opGroup, func, kernelConfig, ncclKernel)
        │
        ├─ 3a. 从 KernelFlagPool 弹出一个 kernelFlag
        │      （volatile int* 数组，驻留在 pinned host memory）
        │
        ├─ 3b. 创建 CtranGpeCmd，绑定 opGroup + func + kernelFlag
        │
        ├─ 3c. 将 cmd 入队到 GPE 线程的命令队列 (cmdEnqueue)
        │       - 非 capture 模式：直接 cmdEnqueue
        │       - graph capture 模式：通过 cudaGraphAddHostNode 延迟入队
        │
        └─ 3d. 在 user stream 上 launch GPU kernel
              - 第一个参数是 kernel flag 指针
              - 第二个参数是 devState_d（共享设备内存）
              - 第三个参数是 collective-specific args
```

### 2.3 GPE Thread 主循环

```cpp
void CtranGpe::Impl::gpeThreadFn() {
    cudaSetDevice(cudaDev);
    while (1) {
        auto cmd = cmdDequeue();  // 阻塞等待新命令

        if (cmd->type == TERMINATE) { delete cmd; return; }

        // ===== 等待 kernel 启动 =====
        if (kernelFlag) {
            volatile int* flag_d = kernelFlag->flag_;
            while (flag_d[0] != KERNEL_STARTED &&
                   flag_d[0] != KERNEL_STARTED_AND_EXIT) {
                std::this_thread::yield();
            }
        }

        // ===== 执行 collective 函数 (网络 I/O) =====
        // 这里调用的 func 会调用 mapper->iput / irecvCtrl 等
        // 进而调用 CtranIb->iput (RDMA WRITE) 等
        cmd->coll.func(cmd->coll.opGroup);

        // ===== 通知 kernel 结束 =====
        if (kernelFlag) {
            kernelFlag->setFlagPerGroup(KERNEL_TERMINATE);
            // 等待 kernel 确认退出
        }

        delete cmd;
    }
}
```

**关键同步协议**：
1. GPU kernel 启动后，将 `flag[0]` 设为 `KERNEL_STARTED`
2. GPE 线程看到 KERNEL_STARTED 后开始执行网络 I/O
3. 网络 I/O 完成后，GPE 设置 `KERNEL_TERMINATE`
4. GPU kernel 看到 TERMINATE 后退出

### 2.4 KernelFlag 同步机制

```
            GPU kernel                    GPE Thread
            ==========                    ==========
                │                              │
         launch kernel                    cmdDequeue()
                │                              │
    flag[0] = KERNEL_STARTED ──────────►  yield loop exits
                │                              │
     waitNotify() / reduce()              func(opGroup)
     (等待 GPE 完成 RDMA)                  - iput(RDMA_WRITE)
                │                          - waitNotify()
     poll KernelElem->status              - post(KernelElem)
     等待 POSTED/DONE                          │
                │                              │
    flag[0] = KERNEL_UNSET ◄──────────── flag=KERNEL_TERMINATE
                │                              │
          kernel exits                    delete cmd
```

### 2.5 KernelElem：GPE↔Kernel 的操作级同步

`KernelElem` 是 GPE 与 kernel 之间传递**单个操作**状态的共享结构（cudaHostAlloc pinned memory）：

```cpp
struct alignas(16) KernelElem {
    enum ElemStatus { RESET, INUSE, POSTED, REVOKED, DONE };

    union {                          // 操作参数
        struct { ... } staged;       // staged send/recv
        struct {                     // putNotify
            const void* sendbuff;
            volatile uint64_t recvbuff;  // GPE 填写远端地址
            size_t nbytes;
            int ngroups;
            bool notify;
            int peerLocalRank;
        } putNotify;
        struct { ... } waitNotify;   // 等待远端通知
        struct { ... } reduce;       // local reduce
    };

    volatile int status[MAX_THREAD_BLOCKS];  // 每个 thread block 独立状态
    int ngroups;
    KernelElem* next;                         // 链表（pool 管理用）
};
```

**状态转换流程**：
```
RESET → INUSE (算法层 allocKernelElem)
      → POSTED (GPE 填好参数后 post())
      → kernel 看到 POSTED，执行操作
      → DONE (kernel 设置)
      → RESET (pool 回收)
```

或者：
```
INUSE → REVOKED (GPE 决定不走 NVL path，revoke elem)
      → kernel 看到 REVOKED，跳过该操作
```

---

## 3. 内存注册与远端访问

### 3.1 Registration Cache

CTRAN 使用 `RegCache` 缓存 GPU 内存注册：

```cpp
// 注册流程
mapper->regMem(buf, len, &segHdl);
// 内部：
//   1. 查 RegCache，已注册则返回
//   2. 调用 CtranIb::regMem(buf, len, cudaDev, &ibRegElem)
//      → ibv_reg_mr() 或 ibv_reg_dmabuf_mr()（DMA-BUF 优化）
//   3. 缓存到 RegCache
```

### 3.2 Export/Import 流程

跨节点通信需要交换远端 buffer 的 rkey 和 virtual address：

```
Rank 0 (sender)                           Rank 1 (receiver)
==============                            ================

regMem(sendbuf) → segHdl                   regMem(recvbuf) → segHdl
                                           │
                                           ▼
                                      isendCtrl(recvbuf, hdl, ←rank0)
                                      // 导出: addr + rkey → ControlMsg
                                           │
    irecvCtrl(&remoteBuf, &rkey, ←rank1)   │
    // 收到 ControlMsg                      │
    // 导入: remoteBuf = msg.remoteAddr     │
    //        rkey = msg.ibDesc.rkeys       │
         │                                 │
         ▼                                 │
    iput(sendbuf, remoteBuf, len,          │
         rank1, config{rkey, notify=true}) │
    // RDMA WRITE + IMM                    │
         │                                 │
         └─────────── data ───────────────►│
         └─────────── notify (IMM) ────────►│
                                           │
                                      waitNotify(rank0)
                                      // poll CQ for IMM
```

### 3.3 ControlMsg 结构

```cpp
struct ControlMsg {
    ControlMsgType type;  // IB_EXPORT_MEM, NVL_EXPORT_MEM, SYNC, ...
    union {
        struct {
            uint64_t remoteAddr;        // 远端 buffer 虚拟地址
            int nKeys;
            uint32_t rkeys[MAX_DEVS];   // 每个 IB device 的 rkey
        } ibDesc;
        IpcDesc ipcDesc;                // NVL IPC handle
    };
};
```

---

## 4. IB Backend (CtranIb) 详细实现

### 4.1 Virtual Connection (VC) 架构

每对 peer rank 之间建立一个 `CtranIbVirtualConn`，包含多个 QP：

| QP 类型 | 用途 |
|---------|------|
| Control QP | 传输 ControlMsg（buffer 地址、rkey 交换） |
| Data QP(s) | RDMA WRITE / READ 数据传输，支持多 QP 负载均衡 |
| Notify QP | 零字节 RDMA_WRITE_WITH_IMM，通知远端数据到达 |
| Atomic QP | RDMA atomic 操作（fetchAndAdd、atomicSet） |

QP 数量可配置（`NCCL_CTRAN_IB_DEVICES_PER_RANK`），支持多 NIC 并行。

### 4.2 Bootstrap 连接建立

```
Rank 0                                    Rank 1
======                                    ======
init:                                      init:
  listenSocket->bindAndListen()              listenSocket->bindAndListen()
  allGather(listenSocketAddr)                allGather(listenSocketAddr)
  bootstrapAccept thread started             bootstrapAccept thread started

首次通信时 (rank < peerRank 的一方主动连接):
  bootstrapConnect(peerRank=1)
    │
    ├─ socket->connect(rank1_listenAddr)
    ├─ send(magic + myRank)
    └─ connectVc(sock, isServer=false)
         ├─ createVc(peerRank)
         ├─ getLocalBusCard() // QP nums, GID, LID
         ├─ send(localBusCard)     ────────►  bootstrapAccept():
         ├─ recv(remoteBusCard)    ◄────────    recv(magic + peerRank)
         ├─ setupVc(remoteBusCard)              connectVc(sock, isServer=true)
         │   // ibv_modify_qp: INIT→RTR→RTS       同样的 busCard 交换
         └─ ACK exchange                        setupVc + ACK
```

### 4.3 iput 实现（RDMA WRITE）

```cpp
commResult_t CtranIbVirtualConn::iput(sbuf, dbuf, len,
                                       ibRegElem, remoteAccessKey,
                                       notify, config, req, fast) {
    // 1. 选择 QP（多 QP 负载均衡 - DQPLB 模式）
    int qpIdx = selectDataQp(len);

    // 2. 构建 ibv_send_wr
    ibv_sge sge = { .addr = sbuf, .length = len, .lkey = lkey };
    ibv_send_wr wr = {
        .opcode = IBV_WR_RDMA_WRITE,  // 如果 notify: IBV_WR_RDMA_WRITE_WITH_IMM
        .send_flags = IBV_SEND_SIGNALED,  // 可选，由信令策略决定
        .wr.rdma = { .remote_addr = dbuf, .rkey = remoteAccessKey.rkeys[dev] }
    };

    // 3. 大消息分片（按 QP scaling threshold 分）
    //    每个分片发到不同 QP 实现负载均衡

    // 4. ibv_post_send
    dataQps[qpIdx].postSend(wr);
}
```

### 4.4 Notify 机制

CTRAN 使用 `RDMA_WRITE_WITH_IMM` 实现零字节通知：
- 发送端：最后一个 RDMA WRITE 带 IMM data
- 接收端：在 Notify QP 上看到 IMM CQE → 数据已到达
- 这避免了额外的 Send/Recv 开销

```cpp
// 发送端
vc->notify(req);
// → ibv_post_send with IBV_WR_RDMA_WRITE_WITH_IMM, zero length

// 接收端
vc->checkNotify(&done);
// → poll CQ, 检查是否有 IBV_WC_RECV_RDMA_WITH_IMM
```

### 4.5 Progress 模型

```cpp
commResult_t CtranIb::progressInternal() {
    while (1) {
        bool more = false;
        for (int device = 0; device < DEVICES_PER_RANK; device++) {
            ibv_wc wc;
            int count = cqs[device].pollCq(1);
            if (count == 0) continue;
            more = true;

            // 根据 QP number 找到对应的 VC
            auto vc = getVcByQp(wc.qp_num, device);
            // 处理 CQE：更新 request 完成状态、处理 notify 等
            vc->processCqe(wc.opcode, wc.qp_num, wc.imm_data, wc.wr_id);
        }
        if (!more) break;
    }
    progressPendingOps();  // 处理等待 VC 建立的延迟操作
}
```

Progress 由 GPE 线程在执行 collective 函数时隐式驱动（`mapper->testRequest()` → `ctranIb->progress()`）。

---

## 5. Mapper Layer：统一抽象

`CtranMapper` 是算法层与后端之间的统一抽象层：

### 5.1 后端选择策略

```cpp
CtranMapperBackend queryPeerBackend(regElem, rank) {
    if (ctranNvl && nvl->isSupported(rank) && regElem->ipcRegElem) {
        // NVLink peer + cuMem buffer → NVL backend
        return NVL;
    }
    if (ctranIb && regElem->ibRegElem) {
        // IB registered → IB backend
        return IB;
    }
    if (ctranTcpDm && regElem->tcpRegElem) {
        return TCPDM;
    }
    return UNSET;
}
```

规则：同节点 NVLink peer 优先走 NVL；跨节点走 IB RDMA。

### 5.2 典型 AllReduce Ring 流程

以 IB 路径为例的 AllReduce Ring 算法交互：

```
步骤 1: 控制面（地址交换）
  - 每个 rank 注册 sendbuf/recvbuf → segHdl
  - allGatherCtrl: 每个 rank 广播自己的 recvbuf 地址 + rkey
  - 此时每个 rank 都知道所有 peer 的 recvbuf 远端地址

步骤 2: GPE submit
  - 算法构造 opGroup + kernelConfig
  - gpe->submit() → launch kernel + enqueue GPE cmd

步骤 3: Kernel 启动
  - kernel 设置 flag=KERNEL_STARTED
  - kernel 开始计算 local reduce（如果有）
  - kernel 等待 KernelElem->status == POSTED（等网络数据到达）

步骤 4: GPE 线程执行网络 I/O
  - Ring 的每一步：
    a. iput(local_chunk → remote_recvbuf, notify=true)
    b. waitNotify(from_prev_rank)  // 等待上游 RDMA 完成
    c. post(KernelElem) // 通知 kernel 数据已到
  - kernel 收到 POSTED，执行 reduce
  - 重复 ring 步骤

步骤 5: 清理
  - GPE 设置 KERNEL_TERMINATE
  - kernel 退出
  - stream 上的后续操作可以继续
```

---

## 6. GpeKernelSync：多步协作机制

对于需要多轮 GPU 计算与网络 I/O 交替的算法，CTRAN 提供 `GpeKernelSync`：

```cpp
struct GpeKernelSync {
    int postFlag[MAX_THREAD_BLOCKS];      // GPE → kernel
    int completeFlag[MAX_THREAD_BLOCKS];  // kernel → GPE

    void post(int step);           // GPE: 通知 kernel 第 step 步的数据已就绪
    bool isComplete(int step);     // GPE: 检查 kernel 是否完成第 step 步
    void waitComplete(int step);   // GPE: 阻塞等待 kernel 完成第 step 步
};
```

**在 kernel 侧** (device code)：
```cuda
// 等待 GPE 通知第 step 步数据就绪
while (gpeKernelSync->postFlag[blockIdx.x] < step) { /* spin */ }

// 执行 reduce / copy
reduce_kernel(src, dst, count);

// 通知 GPE 第 step 步计算完成
gpeKernelSync->completeFlag[blockIdx.x] = step;
```

**在 GPE 侧** (host code)：
```cpp
for (int step = 0; step < nSteps; step++) {
    // 等待 kernel 上一步的 reduce 完成
    if (step > 0) {
        gpeKernelSync->waitComplete(step - 1);
    }

    // 发起 RDMA WRITE
    mapper->iput(src, remoteDst, len, peerRank, config, &req);
    mapper->waitRequest(&req);

    // 等待远端 RDMA 到达
    mapper->waitNotify(&notify);

    // 通知 kernel 数据已到，可以做下一步 reduce
    gpeKernelSync->post(step);
}
```

---

## 7. 设备侧共享状态 (CtranAlgoDeviceState)

```cpp
struct CtranAlgoDeviceState {
    // NVLink peer 的同步结构（intra-node）
    CtranAlgoDeviceSync* remoteSyncsMap[MAX_NVL_PEERS];
    CtranAlgoDeviceSync* localSyncsMap[MAX_NVL_PEERS];

    // NVLink peer 的 staging buffer（intra-node）
    void* remoteStagingBufsMap[MAX_NVL_PEERS];
    void* localStagingBufsMap[MAX_NVL_PEERS];

    // 通信器元信息
    size_t bufSize;
    bool enableTraceLog;
    CommStateXDev statex;         // rank, nRanks, localRank 等
    bool enableCancellableWaits;  // 支持 abort
    uint64_t opCount;
};
```

这个结构通过 `cudaMalloc` 分配在 device memory，作为 kernel 的第二个参数传入。它**不包含**跨 NIC 的远端地址——那些由 GPE 线程通过 KernelElem 动态传递。

---

## 8. 关键设计决策与启示

### 8.1 为什么 GPU kernel 不直接发起 RDMA？

1. **GPU 无法直接调用 ibv_post_send**：IB verbs 是用户态 CPU API
2. **GPU 内存不能直接映射 CQ/QP 的 doorbell**：虽然 ibverbx 有 `mapToDevice()` 实验性支持（Mlx5dv），但生产环境中 CTRAN 仍使用 CPU proxy
3. **复杂的连接管理与错误处理**：QP 状态机、重传、超时等在 CPU 上更容易处理
4. **CUDA graph 兼容性**：通过 host node callback，GPE 命令可以被 graph capture

### 8.2 CTRAN 的 Proxy 模式 vs NCCL 的 Proxy 模式

| 特性 | NCCL | CTRAN |
|------|------|-------|
| 通信模式 | Send/Recv (two-sided) | RDMA WRITE (one-sided) |
| Proxy 触发 | FIFO ring buffer | CUDA stream callback / host node |
| Kernel↔Proxy 同步 | FIFO head/tail | KernelFlag + KernelElem |
| 数据暂存 | Bounce buffer + GDR copy | 直接 RDMA 到目标 buffer（zero-copy） |
| 多 QP 支持 | 单 QP per channel | 多 QP per VC (DQPLB) |

### 8.3 对 unified-ccl 的设计启示

基于 CTRAN 的架构，unified-ccl 的跨 NIC 通信应该：

1. **实现 Primitives → FIFO → Proxy 流水线**
   - GPU kernel 向 pinned memory FIFO 写入操作描述
   - Proxy 线程轮询 FIFO，调用 UCX API 执行网络 I/O
   - 类似 CTRAN 的 KernelElem，但简化为 FIFO entry

2. **使用 RDMA WRITE 而非 Send/Recv**
   - UCX 的 `ucp_put_nbi()` + `ucp_ep_flush()`
   - 需要预先交换远端 buffer 的 rkey（通过 MPI 或 bootstrap）

3. **KernelFlag 同步协议可直接复用**
   - 在 SYCL/Level Zero 上，使用 USM host allocation 代替 cudaHostAlloc
   - volatile int* flag 在 Intel GPU 上同样有效

4. **简化的两步模型**
   - Step 1: kernel 写 FIFO + 设 READY flag
   - Step 2: proxy 读 FIFO + RDMA + 设 DONE flag + kernel poll

5. **分层注册**
   - 延用 CTRAN 的 RegCache 思路
   - 首次使用时 ucp_mem_map()，缓存 rkey
   - deregMem 时 ucp_mem_unmap()

---

## 9. 附录：关键源文件索引

| 文件路径 | 内容 |
|---------|------|
| `gpe/CtranGpe.h` | GPE 公共接口：submit(), submitHost(), allocKernelElems() |
| `gpe/CtranGpeImpl.cc` | GPE 线程主循环 gpeThreadFn()，submit 实现 |
| `gpe/CtranGpeDev.h` | KernelElem, KernelConfig, CtranKernelArgs 定义 |
| `algos/CtranAlgoDev.h` | CtranAlgoDeviceState, CtranAlgoDeviceSync 定义 |
| `algos/common/GpeKernelSync.h` | GpeKernelSync 多步同步结构 |
| `mapper/CtranMapper.h` | Mapper 公共接口：regMem, iput, isendCtrl, irecvCtrl |
| `backends/ib/CtranIb.h` | IB backend：regMem, iput, notify, progress |
| `backends/ib/CtranIb.cc` | IB 初始化、bootstrap、VC 管理、CQ polling |
| `backends/CtranCtrl.cc` | 控制消息回调管理 |
| `CtranPipes.cc` | Pipes (MultiPeerTransport) NVL staging buffer 管理 |
| `ibverbx/` | IB verbs C++ 封装库（IbvVirtualQp, IbvMr, Mlx5dv 等） |

---

## 10. unified-ccl Proxy 实现方案对比

基于 CTRAN 和 NCCL 的经验，我们为 unified-ccl 评估了三种 proxy 实现方案。

### 10.1 方案 A：统一 FIFO 转发（✅ 选定方案）

**核心思想**：primitives 层的 send/recv 写 FIFO → proxy poll → net->isend/irecv。Device RDMA API 也复用同一套 FIFO，只是 entry type 不同。

```
GPU Kernel (Primitives::send)        Proxy Thread
────────────────────────────         ────────────
write data to fifo->buffs[slot]  →   poll tail
update fifo->tail                    read entry type
                                     if SEND: net->isend(buff, size)
                                     if PUT:  net->isend(to remote addr)
                                     if SIGNAL: net->isend(zero-byte + tag)
                                     net->test() → done
                                     update fifo->head
```

**优点**：
- 与现有 `ucclConnFifo` 设计完全对齐，改动最小
- Ring AllReduce 和 device API 共享 proxy 线程，无资源浪费
- 增量实现——先跑通 Ring，再加 put/signal 的 entry type

**缺点**：
- FIFO entry 需要 tagged union（不同 op type 不同字段），结构稍复杂
- 所有操作串行化在同一个 FIFO 里，高并发时可能成为瓶颈

**选定理由**：与现有代码最对齐，可以增量实现，后续瓶颈出现时再演进到方案 B。

### 10.2 方案 B：双层架构 — Collective FIFO + Device Command Queue（🔮 未来演进）

**核心思想**：Ring AllReduce 走现有 FIFO 路径。Device RDMA API 用独立的 Command Queue（也是 pinned memory），proxy 线程同时 poll 两套队列。

```
Collective 路径:                     Device API 路径:
  Primitives::send()                   rdma_put(dst, src, len, peer)
  → connFifo->buffs[slot]             → cmdQueue->entries[slot]
  → connFifo->tail++                  → cmdQueue->tail++
       │                                    │
       └──── Proxy Thread ─────────────────┘
             poll connFifo->tail
             poll cmdQueue->tail
             dispatch to net->isend / net->irecv
```

**优点**：
- 关注点分离：collective 的 FIFO 格式不需要为 device API 妥协
- Command Queue entry 可以针对 RDMA 优化（含 remote_addr, rkey, op_type）
- 未来容易加 get、atomic 等操作

**缺点**：
- 两套数据结构 + proxy 需要同时 poll 两个源
- 稍多的内存开销（两组 pinned buffer）
- 需要处理两套队列的优先级/公平调度

**演进时机**：当方案 A 的 FIFO entry tagged union 变得难以管理，或需要独立的 device API 优先级调度时。

### 10.3 方案 C：CTRAN 风格 — KernelElem + GPE Submit（🔮 长期参考）

**核心思想**：不使用传统 FIFO，而是每次 collective submit 时创建一个 GPE command（含 opGroup），proxy 线程 dequeue command 后一次性执行所有 RDMA。Device API 通过 KernelElem（pinned memory 控制块）与 proxy 交互。

```
Algorithm:                          Device API kernel:
  gpe->submit(opGroup, func)          elem = allocKernelElem()
  → enqueue CtranGpeCmd               elem->put.src = ...
  → launch kernel                     elem->status = INUSE
       │                              poll(elem->status == POSTED)
       │                                   │
       └──── GPE Thread ──────────────────┘
             dequeue cmd
             wait KERNEL_STARTED
             func(opGroup): mapper->iput() ...
             post(KernelElem) → elem->status = POSTED
             set KERNEL_TERMINATE
```

**优点**：
- 最接近 CTRAN 参考实现，架构成熟度最高
- KernelElem 的 status 状态机（RESET→INUSE→POSTED→DONE）非常灵活
- 天然支持多步交替（GpeKernelSync）

**缺点**：
- 与现有代码差距最大——需要重写 primitives 层和 proxy
- 复杂度高：KernelElem pool、flag pool 等基础设施都要从头建
- CTRAN 依赖 CUDA graph capture，Intel GPU 无对应机制

**参考时机**：如果未来需要支持类似 CUDA graph 的 kernel capture / replay 场景，或需要复杂的多步 kernel↔proxy 协作。
