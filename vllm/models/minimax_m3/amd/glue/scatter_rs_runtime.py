"""Host side of the fused top-4 combine + reduce-scatter (scatter_rs.py) for the SP prefill post-MoE path.

    srs = ScatterRs(cpu_group, rank, device)            # collective (peer-buffer handshake) + compile; at setup
    out = srs.launch(inter, reverse_sorted, sorted_weights, m)  # [R, 6144] bf16, R = ceil(m / 4): this rank's rows
    srs.assert_healthy()                                 # device-side trap if any rank's wait timed out
    srs.reset()                                          # collective, nothing in flight: after a fault only

``inter`` / ``reverse_sorted`` / ``sorted_weights``: what AITER's FlyDSL scatter stage 2 hands
``mxfp4_moe_scatter_reduce`` (the ``scatter_intermediate`` dict of the patched ``fused_moe.py``).

Memory: one uncached IPC peer buffer per rank, ``peer_bytes(max_rows)`` (151 MB + 54 KB of flags at the default
16,384 rows, single-buffered; see README), allocated by the constructor (collective). Plus 1 KB of colours.
"""

from __future__ import annotations

import os

import torch

try:
    from . import scatter_rs as K
except ImportError:  # standalone (tests next to scatter_rs.py)
    import scatter_rs as K

FAULT_MESSAGE = "MiniMax-M3 SP scatter-RS: error record set (a peer wait timed out)"
# a bounded wait, generous: in eager prefill one rank can reach the kernel seconds after another (first-use JIT of
# the MoE kernels for a new M bucket); s_memrealtime runs at 100 MHz
DEFAULT_WAIT_S = float(os.environ.get("VLLM_M3_SP_SCATTER_RS_WAIT_S", "60"))


class ScatterRs:
    def __init__(self, cpu_group, rank: int, device: torch.device, max_rows: int = K.DEFAULT_MAX_ROWS,
                 wait_s: float = DEFAULT_WAIT_S, key: K.SrsBuild | None = None) -> None:
        import flydsl.compiler as flyc

        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer

        self.cpu_group = cpu_group
        self.key = key or K.SrsBuild(max_rows=max_rows)
        self.max_rows = self.key.max_rows
        self.rows_max = K.r_max(self.max_rows)
        self.rank, self.device = rank, device
        self.ticks = int(wait_s * 1e8)
        self.peer_nbytes = K.peer_bytes(self.max_rows)
        self.peer = PeerBuffer(self.peer_nbytes, cpu_group, rank, K.TP, device)  # collective
        self.sym = int(self.peer.local)
        self.peers_ptr = self.peer.addresses.data_ptr()
        self.fault_word = self.peer.bytes[:16].view(torch.int32)  # [failed, rank, colour, site]
        self.colors = torch.zeros(K.BLOCKS, dtype=torch.int32, device=device)
        self.tl = torch.zeros(K.BLOCKS * K.TL_POINTS, dtype=torch.int64, device=device)
        self._clear()
        dummy = torch.zeros(16, dtype=torch.int32, device=device)
        self.compiled = flyc.compile(K.build_scatter_rs(self.key), *self._args(dummy, dummy, dummy, dummy, 1, 1, 1, 0))
        torch.cuda.synchronize(device)

    def _clear(self) -> None:
        lay = K.layout(self.max_rows)
        end = lay["inbox"][0]  # error record, flags, done words (the inbox data needs no clearing)
        self.peer.bytes[:end].zero_()
        self.colors.zero_()

    def _args(self, inter, rev, sw, out, m, rows, max_sorted, mode, ticks=None):
        return (inter.data_ptr(), rev.data_ptr(), sw.data_ptr(), out.data_ptr(), self.colors.data_ptr(),
                self.tl.data_ptr(), self.sym, self.peers_ptr, self.ticks if ticks is None else ticks, m, rows,
                max_sorted, self.rank, mode, torch.cuda.current_stream())

    def fits(self, m: int) -> bool:
        return 0 < m <= self.max_rows

    def launch(self, inter: torch.Tensor, rev: torch.Tensor, sw: torch.Tensor, m: int,
               out: torch.Tensor | None = None, ticks: int | None = None) -> torch.Tensor:
        """Collective (every rank, same m, same order). inter: [max_sorted, 6144] bf16; rev: [m * 4] int32;
        sw: [>= max_sorted] fp32. -> this rank's [R, 6144] bf16 rows of the reduce-scattered combined MoE output."""
        assert self.fits(m), (m, self.max_rows)
        assert inter.dtype == torch.bfloat16 and inter.shape[1] == K.HIDDEN and inter.is_contiguous()
        assert rev.dtype == torch.int32 and rev.numel() >= m * K.TOPK and rev.is_contiguous()
        assert sw.dtype == torch.float32 and sw.is_contiguous()
        rows = -(-m // K.TP)
        if out is None:
            out = torch.empty((rows, K.HIDDEN), dtype=torch.bfloat16, device=inter.device)
        max_sorted = min(inter.shape[0], sw.numel())
        self.compiled(*self._args(inter, rev, sw, out, m, rows, max_sorted, 1, ticks))
        return out

    def assert_healthy(self) -> None:
        torch._assert_async(self.fault_word[0] == 0, FAULT_MESSAGE)

    def reset(self) -> None:
        """Collective, no launch in flight on any rank: flags, done words, colours and the error record cleared
        (only needed after a fault: a healthy run never needs it)."""
        import torch.distributed as dist

        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu_group)  # nobody launches while a peer still clears
        self._clear()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu_group)

    def close(self) -> None:
        if self.peer is not None:
            self.peer.close()
            self.peer = None
