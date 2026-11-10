"""Host side of the post-MoE collective (postar.py) for the vLLM decode glue: one peer buffer + colour array shared by
every width, a compiled kernel and persistent outputs per width.

Usage (every TP rank, same order; see README.md "Integration"):

    pa = PostAr(widths, group.cpu_group, rank, device, mode="rsag8")   # collective (peer-buffer handshake)
    out = pa.launch(moe_out, residual, norm_w)          # in or out of graph capture; returns PostArOut
    # out.x_q [S, 6144] e4m3, out.x_s [S, 192] E8M0, out.res [S, 6144] bf16 (rsag8: only rows rank*S/4 .. valid)
    pa.assert_healthy()                                  # device-side trap if any rank's wait timed out
    pa.reset()                                           # collective, no launch in flight: after a fault only

CUDA graphs: the launch reads its parity from the device colour array and every launch runs the full grid, so graph
replays and interleaved widths need no host-side state and no epoch reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from . import postar as PA
except ImportError:  # standalone (bench / tests next to postar.py)
    import postar as PA

FAULT_MESSAGE = "MiniMax-M3 post-MoE collective: error record set (a peer wait timed out)"


@dataclass
class PostArOut:
    s: int
    compiled: Any
    x_q: torch.Tensor
    x_s: torch.Tensor
    res: torch.Tensor


class PostAr:
    def __init__(self, widths, cpu_group, rank: int, device: torch.device, mode: str = "rsag8",
                 wait_ticks: int = PA.WAIT_TICKS, timeline: bool = False) -> None:
        import flydsl.compiler as flyc

        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer

        assert mode in PA.MODES and mode != "oneshot", mode
        self.mode, self.rank, self.device = mode, rank, device
        self.peer = PeerBuffer(PA.peer_bytes(mode), cpu_group, rank, PA.TP, device)  # collective
        self.sym = int(self.peer.local)
        self.peers_ptr = self.peer.addresses.data_ptr()
        self.fault_word = self.peer.bytes[:16].view(torch.int32)  # [failed, rank, colour, site]
        self.colors = torch.zeros(PA.BLOCKS, dtype=torch.int32, device=device)
        self.tl = torch.zeros(PA.BLOCKS * PA.TL_POINTS, dtype=torch.int64, device=device)
        self._clear()
        self.bufs: dict[int, PostArOut] = {}
        h = PA.HIDDEN
        dummy = torch.zeros(1, h, dtype=torch.bfloat16, device=device)
        for s in sorted(set(widths)):
            assert s % PA.TP == 0 and s <= PA.MAX_ROWS, s
            b = PostArOut(
                s=s, compiled=None,
                x_q=torch.empty((s, h), dtype=torch.float8_e4m3fn, device=device),
                x_s=torch.empty((s, h // 32), dtype=torch.uint8, device=device),
                res=torch.empty((s, h), dtype=torch.bfloat16, device=device),
            )
            key = PA.PostArBuild(rows=s, mode=mode, wait_ticks=wait_ticks, timeline=timeline)
            b.compiled = flyc.compile(PA.build_postar(key), *self._args(b, dummy, dummy, dummy.view(-1), 0))
            self.bufs[s] = b
        torch.cuda.synchronize(device)

    def _clear(self) -> None:
        self.peer.bytes.fill_(0xFF)  # every inbox half-word = the sentinel
        self.peer.bytes[:PA.ERROR_BYTES].zero_()
        self.colors.zero_()

    def _args(self, b: PostArOut, inp, res, norm_w, mode: int):
        return (inp.data_ptr(), res.data_ptr(), norm_w.data_ptr(), b.x_q.data_ptr(), b.x_s.data_ptr(),
                b.res.data_ptr(), self.colors.data_ptr(), self.tl.data_ptr(), self.sym, self.peers_ptr, self.rank,
                mode, torch.cuda.current_stream())

    def launch(self, inp: torch.Tensor, res: torch.Tensor, norm_w: torch.Tensor) -> PostArOut:
        """inp: this rank's [S, 6144] bf16 MoE output (contiguous); res: the replicated residual [S, 6144] bf16;
        norm_w: the next layer's input-norm weight [6144] bf16. Collective: every rank, same S, same order."""
        b = self.bufs[inp.shape[0]]
        assert inp.is_contiguous() and res.is_contiguous() and inp.dtype == torch.bfloat16
        b.compiled(*self._args(b, inp, res, norm_w, 1))
        return b

    def assert_healthy(self) -> None:
        torch._assert_async(self.fault_word[0] == 0, FAULT_MESSAGE)

    def reset(self) -> None:
        """Collective, no launch in flight on any rank: inboxes back to the sentinel, colours and the error record
        cleared (only needed after a fault: a healthy run never needs it)."""
        torch.cuda.synchronize(self.device)
        self._clear()
        torch.cuda.synchronize(self.device)

    def close(self) -> None:
        if self.peer is not None:
            self.peer.close()
            self.peer = None
