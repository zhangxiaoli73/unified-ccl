"""Performance benchmark for Unified-CCL AllReduce."""

import time
import torch
import sys


def benchmark_allreduce(comm, size, dtype, warmup=5, iterations=20):
    """Benchmark AllReduce for a given size and dtype.

    Args:
        comm: UCCL Communicator
        size: Number of elements
        dtype: torch dtype (float16 or bfloat16)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        dict with timing and bandwidth info
    """
    tensor = torch.ones(size, dtype=dtype, device="xpu")
    nbytes = size * tensor.element_size()

    # Warmup
    for _ in range(warmup):
        comm.allreduce(tensor)

    # Timed iterations
    torch.xpu.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        comm.allreduce(tensor)
    torch.xpu.synchronize()

    elapsed = time.perf_counter() - start
    avg_time_us = (elapsed / iterations) * 1e6
    # Ring AllReduce: 2*(N-1)/N * data_size algorithmic bandwidth
    nranks = comm.world_size
    algo_factor = 2.0 * (nranks - 1) / nranks if nranks > 1 else 1.0
    bus_bw = (nbytes * algo_factor) / (elapsed / iterations) / 1e9

    return {
        "size": size,
        "nbytes": nbytes,
        "dtype": str(dtype),
        "avg_time_us": avg_time_us,
        "bus_bw_gbps": bus_bw,
    }


def run_perf_test():
    """Run performance benchmarks across various message sizes."""
    try:
        import uccl
    except ImportError:
        print("ERROR: uccl module not available. "
              "Build and install first.")
        sys.exit(1)

    comm = uccl.Communicator(rank=0, world_size=1)
    print(f"Unified-CCL AllReduce Performance Test")
    print(f"Rank: {comm.rank}, World Size: {comm.world_size}")
    print()

    sizes = [
        64,         # 128 B
        256,        # 512 B
        1024,       # 2 KB
        4096,       # 8 KB
        16384,      # 32 KB
        65536,      # 128 KB
        262144,     # 512 KB
        1048576,    # 2 MB
        4194304,    # 8 MB
        16777216,   # 32 MB
    ]

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"\n--- {dtype} ---")
        print(f"{'Size':>12s}  {'Bytes':>12s}  "
              f"{'Time (us)':>12s}  {'BusBW (GB/s)':>14s}")
        print("-" * 56)

        for size in sizes:
            try:
                result = benchmark_allreduce(comm, size, dtype)
                print(f"{result['size']:>12d}  {result['nbytes']:>12d}  "
                      f"{result['avg_time_us']:>12.1f}  "
                      f"{result['bus_bw_gbps']:>14.2f}")
            except Exception as e:
                print(f"{size:>12d}  ERROR: {e}")

    comm.destroy()
    print("\nDone.")


if __name__ == "__main__":
    run_perf_test()
