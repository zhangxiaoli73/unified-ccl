# Unified-CCL Documentation

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=icpx
make -j$(nproc)
```

## Run Tests

```bash
# Single-process transport test
./tests/cpp/test_transport

# Multi-process AllReduce test
mpirun -n 2 ./tests/cpp/test_allreduce

# Python tests
pytest tests/python/test_allreduce.py

# Performance benchmark
python tests/python/test_perf.py
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `UCCL_DEBUG` | Debug level: NONE, ERROR (default), WARN, INFO, TRACE |
| `UCCL_NET_PLUGIN` | Path to network plugin .so |
