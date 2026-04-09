"""High-level Python API for Unified-CCL."""

import torch
import _uccl_bindings


class Communicator:
    """UCCL communicator wrapping ucclComm_t.

    Provides collective communication primitives for Intel GPU tensors.

    Usage:
        comm = Communicator(rank=0, world_size=4)
        tensor = torch.ones(1024, dtype=torch.float16, device="xpu")
        comm.allreduce(tensor)
        comm.destroy()
    """

    def __init__(self, rank: int, world_size: int):
        """Initialize communicator for given rank.

        Args:
            rank: This process's rank (0 to world_size-1).
            world_size: Total number of participating ranks.
        """
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"rank must be in [0, {world_size}), got {rank}")
        if world_size < 1:
            raise ValueError(
                f"world_size must be >= 1, got {world_size}")

        self._comm = _uccl_bindings.comm_init_rank(world_size, rank)
        self._rank = rank
        self._world_size = world_size

    @property
    def rank(self) -> int:
        """This communicator's rank."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Total number of ranks."""
        return self._world_size

    def allreduce(self, tensor: torch.Tensor,
                  op: str = "sum") -> torch.Tensor:
        """In-place allreduce on a torch.Tensor (Intel GPU).

        Args:
            tensor: fp16 or bf16 tensor on Intel GPU (xpu device).
            op: Reduction operation. Currently only "sum" is supported.

        Returns:
            The same tensor, modified in-place with the reduction result.

        Raises:
            AssertionError: If tensor dtype or device is unsupported.
        """
        assert tensor.dtype in (torch.float16, torch.bfloat16), \
            f"Unsupported dtype: {tensor.dtype}. Use float16 or bfloat16."
        assert tensor.is_xpu, \
            "Tensor must be on Intel GPU (xpu device)"

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
        """AllGather: gather data from all ranks.

        Args:
            send_tensor: Local data tensor on Intel GPU.
            recv_tensor: Output tensor with numel = send_tensor.numel() * world_size.

        Returns:
            recv_tensor with gathered data from all ranks.
        """
        assert send_tensor.dtype in (torch.float16, torch.bfloat16), \
            f"Unsupported dtype: {send_tensor.dtype}"
        assert recv_tensor.dtype == send_tensor.dtype, \
            "send and recv tensors must have same dtype"
        assert send_tensor.is_xpu and recv_tensor.is_xpu, \
            "Tensors must be on Intel GPU (xpu device)"
        assert recv_tensor.numel() == send_tensor.numel() * self._world_size, \
            f"recv_tensor must have {send_tensor.numel() * self._world_size} elements"

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
        """ReduceScatter: reduce then scatter across ranks.

        Args:
            send_tensor: Input tensor with numel = recv_tensor.numel() * world_size.
            recv_tensor: Output tensor for this rank's chunk.
            op: Reduction operation. Currently only "sum" is supported.

        Returns:
            recv_tensor with reduced-scattered result.
        """
        assert send_tensor.dtype in (torch.float16, torch.bfloat16), \
            f"Unsupported dtype: {send_tensor.dtype}"
        assert recv_tensor.dtype == send_tensor.dtype, \
            "send and recv tensors must have same dtype"
        assert send_tensor.is_xpu and recv_tensor.is_xpu, \
            "Tensors must be on Intel GPU (xpu device)"
        assert send_tensor.numel() == recv_tensor.numel() * self._world_size, \
            f"send_tensor must have {recv_tensor.numel() * self._world_size} elements"

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

    def __repr__(self) -> str:
        status = "active" if self._comm is not None else "destroyed"
        return (f"Communicator(rank={self._rank}, "
                f"world_size={self._world_size}, status={status})")
