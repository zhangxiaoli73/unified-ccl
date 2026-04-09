"""AllReduce correctness tests for Unified-CCL Python API."""

import torch
import pytest


def test_allreduce_fp16():
    """Test AllReduce with float16 tensors."""
    try:
        import uccl
    except ImportError:
        pytest.skip("uccl module not available")

    comm = uccl.Communicator(rank=0, world_size=1)
    t = torch.ones(1024, dtype=torch.float16, device="xpu")
    result = comm.allreduce(t)

    # Single rank: result should equal input
    assert torch.allclose(result, torch.ones_like(result)), \
        "AllReduce fp16 failed: single rank result != input"
    comm.destroy()


def test_allreduce_bf16():
    """Test AllReduce with bfloat16 tensors."""
    try:
        import uccl
    except ImportError:
        pytest.skip("uccl module not available")

    comm = uccl.Communicator(rank=0, world_size=1)
    t = torch.ones(1024, dtype=torch.bfloat16, device="xpu")
    result = comm.allreduce(t)

    assert torch.allclose(result, torch.ones_like(result)), \
        "AllReduce bf16 failed: single rank result != input"
    comm.destroy()


def test_allreduce_various_sizes():
    """Test AllReduce with various tensor sizes."""
    try:
        import uccl
    except ImportError:
        pytest.skip("uccl module not available")

    comm = uccl.Communicator(rank=0, world_size=1)
    sizes = [1, 16, 256, 1024, 4096, 65536, 262144, 1048576]

    for size in sizes:
        t = torch.ones(size, dtype=torch.float16, device="xpu")
        result = comm.allreduce(t)
        assert torch.allclose(result, torch.ones_like(result)), \
            f"AllReduce failed for size={size}"

    comm.destroy()


def test_allreduce_inplace():
    """Test that AllReduce modifies tensor in-place."""
    try:
        import uccl
    except ImportError:
        pytest.skip("uccl module not available")

    comm = uccl.Communicator(rank=0, world_size=1)
    t = torch.ones(1024, dtype=torch.float16, device="xpu") * 42.0
    data_ptr_before = t.data_ptr()
    result = comm.allreduce(t)

    # Should be the same tensor (in-place)
    assert result.data_ptr() == data_ptr_before, \
        "AllReduce should modify tensor in-place"
    comm.destroy()


def test_communicator_lifecycle():
    """Test communicator creation and destruction."""
    try:
        import uccl
    except ImportError:
        pytest.skip("uccl module not available")

    comm = uccl.Communicator(rank=0, world_size=1)
    assert comm.rank == 0
    assert comm.world_size == 1
    assert repr(comm).startswith("Communicator(")

    comm.destroy()
    # Should be safe to call destroy again
    comm.destroy()


def test_invalid_dtype():
    """Test that unsupported dtype raises error."""
    try:
        import uccl
    except ImportError:
        pytest.skip("uccl module not available")

    comm = uccl.Communicator(rank=0, world_size=1)
    t = torch.ones(1024, dtype=torch.float32, device="xpu")

    with pytest.raises(AssertionError):
        comm.allreduce(t)

    comm.destroy()


def test_invalid_device():
    """Test that CPU tensor raises error."""
    try:
        import uccl
    except ImportError:
        pytest.skip("uccl module not available")

    comm = uccl.Communicator(rank=0, world_size=1)
    t = torch.ones(1024, dtype=torch.float16)  # CPU tensor

    with pytest.raises(AssertionError):
        comm.allreduce(t)

    comm.destroy()


def test_invalid_rank():
    """Test that invalid rank raises error."""
    try:
        import uccl
    except ImportError:
        pytest.skip("uccl module not available")

    with pytest.raises(ValueError):
        uccl.Communicator(rank=-1, world_size=1)

    with pytest.raises(ValueError):
        uccl.Communicator(rank=2, world_size=2)


def test_allreduce_multi_gpu():
    """Multi-process test using torch.multiprocessing.

    To be run with:
        mpirun -n <nranks> python -m pytest test_allreduce.py::test_allreduce_multi_gpu
    """
    # This test requires multi-process setup
    # Skipped in single-process test runner
    pytest.skip("Multi-GPU test requires mpirun")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
