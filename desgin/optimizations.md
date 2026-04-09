如何做通信的性能优化，可以从以下几点考虑。

1. code 实现里面不要出现queue wait这种逻辑。
2. 如果一个collective的实现里面用到了多个sycl kernel，那这些kernel应该尽可能fuse成一个，并在kernel内部实现pipeline，来hide不必要的操作。当然copy engine的path要排除掉，因为copy engine无法在普通的sycl kernel内部使用。
3. 尽可能接近projection的performance数据。
4. 对于有TMA的场景，有没有可能有更好的优化？