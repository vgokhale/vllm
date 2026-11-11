# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the Gluon fused top-k combine + reduce-scatter (scatter_rs_gluon.py) for the SP prefill post-MoE
path: the API of scatter_rs_runtime.ScatterRs (FlyDSL), on the same peer-buffer layout and protocol.

    srs = GluonScatterRs(cpu_group, rank, device)        # collective (peer-buffer handshake) + compile; at setup
    out = srs.launch(inter, reverse_sorted, sorted_weights, m)  # [R, 6144] bf16, R = ceil(m / 4): this rank's rows
    srs.assert_healthy()                                  # device-side trap if any rank's wait timed out
    srs.reset()                                           # collective, nothing in flight: after a fault only

The kernel is compiled once in the constructor (Triton warmup, no launch) and launched through that compiled
handle: M, R, max_sorted, the rank and the wait budget are runtime arguments, so serving never JIT-compiles.
Only an intermediate / output pointer that is not 16-byte aligned (never seen: both are fresh allocations) would
go through the JIT path, which may compile a variant (logged once).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from . import scatter_rs as K
from .scatter_rs_gluon import scatter_rs_kernel

FAULT_MESSAGE = (
    "MiniMax-M3 SP scatter-RS (gluon): error record set (a peer wait timed out)"
)
DEFAULT_WAIT_S = 60.0


@dataclass(frozen=True)
class GCfg:
    warps: int = 16  # CTA = 64 * warps threads; tuned (8: 3-5% slower, 4 spills)
    blocks: int = 256  # grid = the FlyDSL grid (the layout's done words are per CTA)
    lag: int = 2  # iterations between a row's push and its reduce (3: 1-2% slower)
    ntl: int = 0  # 1: intermediate loads with .cs (sc0 nt): within noise
    unroll: int = 2  # 2: iteration pairs in the walk loop (1: 1-2% slower)
    fence: int = 0  # 1: data fences (needed at warps=8; 1-1.5% slower at 16)
    hidden: int = K.HIDDEN
    topk: int = K.TOPK


class _RawPtr:
    """A raw device address as a Triton pointer argument (`*i32`)."""

    def __init__(self, addr: int):
        self.addr, self.dtype = int(addr), torch.int32

    def data_ptr(self) -> int:
        return self.addr


class GluonScatterRs:
    def __init__(
        self,
        cpu_group,
        rank: int,
        device: torch.device,
        max_rows: int = K.DEFAULT_MAX_ROWS,
        wait_s: float = DEFAULT_WAIT_S,
        cfg: GCfg | None = None,
    ) -> None:
        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer

        self.cfg = c = cfg or GCfg()
        assert c.blocks == K.BLOCKS and c.hidden == K.HIDDEN and c.topk == K.TOPK, (
            "scatter_rs.layout is M3's"
        )
        self.cpu_group, self.rank, self.device = cpu_group, rank, device
        self.max_rows = max_rows
        self.rows_max = K.r_max(max_rows)
        self.lay = K.layout(max_rows)
        self.wait_us = int(wait_s * 1e6)
        self.peer_nbytes = K.peer_bytes(max_rows)
        self.peer = PeerBuffer(
            self.peer_nbytes, cpu_group, rank, K.TP, device
        )  # collective
        self.colors = torch.zeros(c.blocks, dtype=torch.int32, device=device)
        addrs = [int(a) for a in self.peer.addresses.tolist()]
        self.ptrs = tuple(
            _RawPtr(addrs[(rank + d) % K.TP]) for d in range(K.TP)
        )  # own, rank+1, +2, +3
        self.fault_word = self.peer.bytes[:16].view(
            torch.int32
        )  # [failed, rank, colour, site]
        self._clear()
        self.consts = dict(
            H=c.hidden,
            TOPK=c.topk,
            W=c.warps,
            BLOCKS=c.blocks,
            RMAX=self.rows_max,
            FLAG0=self.lay["flags"][0],
            DONE0=self.lay["done"][0],
            INBOX0=self.lay["inbox"][0],
            LAG=c.lag,
            NTL=c.ntl,
            UNROLL=c.unroll,
            FENCE=c.fence,
            DBG=0,
        )
        # compile now (warmup: no launch), with the argument kinds launch() passes (pointer dtypes, 16-B aligned)
        t0 = time.perf_counter()
        i32 = torch.zeros(16, dtype=torch.int32, device=device)
        f32 = torch.zeros(16, dtype=torch.float32, device=device)
        self.compiled = scatter_rs_kernel.warmup(
            i32,
            i32,
            f32,
            i32,
            self.colors,
            *self.ptrs,
            self.wait_us,
            1,
            1,
            1,
            rank,
            **self.consts,
            grid=(c.blocks,),
            num_warps=c.warps,
        )
        self.compile_s = time.perf_counter() - t0
        self._run = self.compiled[(c.blocks, 1, 1)]
        self._const_vals = tuple(self.consts.values())
        self._jit_warned = False
        self.launches = 0  # launches so far (a test / log hook)
        md = self.compiled.metadata
        self.describe = (
            f"gluon W={c.warps} blocks={c.blocks} lag={c.lag}: {getattr(md, 'num_warps', c.warps)} "
            f"warps, compile {self.compile_s:.1f} s"
        )
        torch.cuda.synchronize(device)

    def _clear(self) -> None:
        end = self.lay["inbox"][
            0
        ]  # error record, flags, done words (the inbox data needs no clearing)
        self.peer.bytes[:end].zero_()
        self.colors.zero_()

    def fits(self, m: int) -> bool:
        return 0 < m <= self.max_rows

    def launch(
        self,
        inter: torch.Tensor,
        rev: torch.Tensor,
        sw: torch.Tensor,
        m: int,
        out: torch.Tensor | None = None,
        ticks: int | None = None,
    ) -> torch.Tensor:
        """Collective (every rank, same m, same order). inter: [max_sorted, 6144] bf16; rev: [m * 4] int32;
        sw: [>= max_sorted] fp32. -> this rank's [R, 6144] bf16 rows of the reduce-scattered combined MoE output.
        `ticks`: wait bound in 100 MHz ticks (default wait_s)."""
        c = self.cfg
        assert self.fits(m), (m, self.max_rows)
        assert (
            inter.dtype == torch.bfloat16
            and inter.shape[1] == c.hidden
            and inter.is_contiguous()
        )
        assert (
            rev.dtype == torch.int32
            and rev.numel() >= m * c.topk
            and rev.is_contiguous()
        )
        assert sw.dtype == torch.float32 and sw.is_contiguous()
        rows = -(-m // K.TP)
        if out is None:
            out = torch.empty(
                (rows, c.hidden), dtype=torch.bfloat16, device=inter.device
            )
        max_sorted = min(inter.shape[0], sw.numel())
        wait_us = self.wait_us if ticks is None else max(1, -(-int(ticks) // 100))
        args = (
            inter.view(torch.int32),
            rev,
            sw,
            out.view(torch.int32),
            self.colors,
            *self.ptrs,
            wait_us,
            m,
            rows,
            max_sorted,
            self.rank,
        )
        self.launches += 1
        if (
            inter.data_ptr() % 16 == 0
            and out.data_ptr() % 16 == 0
            and rev.data_ptr() % 4 == 0
            and sw.data_ptr() % 4 == 0
        ):
            self._run(*args, *self._const_vals)
        else:  # the compiled handle assumes 16-B aligned vectors: let the JIT specialize (may compile once)
            if not self._jit_warned:
                self._jit_warned = True
                from vllm.logger import init_logger

                init_logger(__name__).warning(
                    "MiniMax-M3 SP scatter-RS (gluon): unaligned input, JIT launch path"
                )
            scatter_rs_kernel[(c.blocks,)](*args, **self.consts, num_warps=c.warps)
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
