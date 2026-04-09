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
