# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode glue: one persistent FlyDSL launch (kernel.py) from the unreduced o_proj
partial to the MoE inputs, then AITER's A16W4 gemm1 / gemm2 on its sort, the
shared expert on its MXFP8 rows and the final TP all-reduce.

Replaces, per MoE layer of a pure-decode step: the post-attention all-reduce, the
residual add, the Gemma RMSNorm (+ MXFP8 copy), the router, the top-k (x routed
scale), moe_sorting and the moe_buf zeroing.

Host state (built once, collectively, outside graph capture): one peer buffer and
one device epoch shared by every width and layer, and per width a compiled kernel,
its scratch and persistent outputs (graph replays bake their addresses). Layers
differ only in the weight pointers passed per launch. The epoch moves on the
device; every rank runs the same launches, so all ranks agree on it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

from . import kernel as K
from . import post_kernel as P
from .envs import envs
from .shared import SharedWeights, shared_decode
from .sp import SpCarry

logger = init_logger(__name__)

# shared-expert stream policy (measured, TP4 MI355X): forked to the side stream
# up to this many rows; in-stream ahead of gemm1 above (the side stream loses
# 15-17 us at 96-128 rows)
SIDE_STREAM_MAX_ROWS = 64
FAULT_MESSAGE = "MiniMax-M3 decode glue: error word set (a peer wait timed out)"


def _writable_flydsl_cache() -> None:
    """AITER points FlyDSL at its bundled read-only cache: as the upstream mono
    runner, switch to a writable one for the process (here under ~/.cache, which
    run_agentx.sh mounts, so compiles persist across runs)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".cache" / "flydsl"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


@dataclass
class _Width:
    s: int
    compiled: Any
    scratch: torch.Tensor
    res_out: torch.Tensor  # [S, H] bf16: the residual after attention
    xn: torch.Tensor  # [S, H] bf16: normed rows (gemm1 input)
    xq: torch.Tensor  # [S, H] fp8: normed rows, MXFP8 (shared expert input)
    xsc: torch.Tensor  # [S, H / 32] uint8 E8M0
    ids: torch.Tensor  # [S, 4] int32 top-k ids
    w: torch.Tensor  # [S, 4] fp32 top-k weights, x routed scale
    sids: torch.Tensor  # moe_sorting outputs (block 16)
    sw: torch.Tensor
    seids: torch.Tensor
    nv: torch.Tensor
    moe_buf: torch.Tensor  # [S, H] bf16, zeroed by the kernel; gemm2 adds into it
    # post-MoE glue (post_kernel.py): the next layer's MXFP8 attention input and the sharded residual
    post: Any = None
    xq_n: torch.Tensor | None = None  # [S, H] fp8
    xs_n: torch.Tensor | None = None  # [S, H / 32] uint8 E8M0
    res_sh: torch.Tensor | None = (
        None  # [S, H] bf16: this rank's owner rows only (S / 4 rows at rank S / 4)
    )
    sh_inter: torch.Tensor | None = (
        None  # [S, I/TP] bf16: the fused shared expert's stage-1 rows
    )
    sh_zero: torch.Tensor | None = (
        None  # [S, H] bf16 zeros: the post-MoE glue's shared input when fused
    )


class DecodeGlue:
    def __init__(
        self,
        widths: tuple[int, ...],
        group,
        device: torch.device,
        post: bool = False,
        shared_inter: int = 0,
        postar: str = "",
    ) -> None:
        self.widths = tuple(widths)
        self.want_post = post
        self.want_postar = postar  # "", "rsag8" or "rsag16" (postar.py)
        self.postar = None
        self.shared_inter = (
            shared_inter  # > 0: the shared expert runs fused in the MoE kernels
        )
        self.group = group
        self.rank = group.rank_in_group
        self.device = device
        self.bufs: dict[int, _Width] = {}
        self.peer = None
        self.fault_word: torch.Tensor | None = None
        # the glue's cross-GPU transport (VLLM_M3_GLUE_RS): "lamport" (plain payload into sentinel inboxes) or "tagged"
        self.transport = envs.GLUE_RS

    # ------------------------------------------------------------ setup
    def setup(self) -> None:
        """Collective (the peer-buffer handshake): every TP rank, same point."""
        import flydsl.compiler as flyc

        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer
        from vllm.platforms import current_platform

        dev = self.device
        # first: the only collective (nothing may fail on one rank before it)
        self.peer = PeerBuffer(
            K.peer_bytes(), self.group.cpu_group, self.rank, K.TP, dev
        )
        self.peer.bytes.zero_()
        K.arm_peer(
            self.peer.bytes
        )  # the Lamport inboxes := the sentinel (unused by tagged builds)
        self.sym = int(self.peer.local)
        self.peers_ptr = self.peer.addresses.data_ptr()
        self.fault_word = self.peer.bytes[:16].view(
            torch.int32
        )  # [failed, rank, epoch, site]
        self.epoch = torch.zeros(K.EPOCH_WORDS, dtype=torch.int32, device=dev)
        self.tl = torch.zeros(K.BLOCKS * K.TL_POINTS, dtype=torch.int64, device=dev)
        self.stream = None
        if not _side_stream_disabled():
            from vllm.utils.torch_utils import aux_stream

            self.stream = aux_stream()
        self.ev_fork, self.ev_join = torch.cuda.Event(), torch.cuda.Event()
        cus = current_platform.num_compute_units(dev.index)
        dummy_h = torch.zeros(1, K.HIDDEN, dtype=torch.bfloat16, device=dev)
        dummy_f = torch.zeros(K.EXPERTS, K.HIDDEN, dtype=torch.float32, device=dev)
        _writable_flydsl_cache()
        for s in self.widths:  # compile and load (a mode-0 launch returns at once)
            b = self._alloc(s)
            key = K.GlueBuild(rows=s, transport=self.transport)
            args = self._args(
                b, dummy_h, dummy_h, dummy_h.view(-1), dummy_f, dummy_f[0], 0
            )
            b.compiled = flyc.compile(K.build_glue(key), *args)
            self._check_residency(b.compiled, key, cus)
            if self.want_post:
                pkey = P.PostBuild(rows=s)
                pargs = self._post_args(
                    b, dummy_h, dummy_h, dummy_h, dummy_h.view(-1), 0
                )
                b.post = flyc.compile(P.build_post(pkey), *pargs)
                self._check_residency(b.post, pkey, cus, P.kernel_name(pkey))
            self.bufs[s] = b
        logger.info(
            "MiniMax-M3 decode glue transport %s: %s",
            self.transport,
            {
                s: K.kernel_name(K.GlueBuild(rows=s, transport=self.transport))
                for s in self.widths
            },
        )
        if self.want_postar:
            # one more collective (its own peer buffer + colours); every rank reaches it after the glue's
            from . import postar as PA
            from .postar_runtime import PostAr

            self.postar = PostAr(
                [s for s in self.widths if s % PA.TP == 0 and s <= PA.MAX_ROWS],
                self.group.cpu_group,
                self.rank,
                dev,
                mode=self.want_postar,
            )
            for pb in self.postar.bufs.values():
                key = PA.PostArBuild(rows=pb.s, mode=self.want_postar)
                self._check_residency(
                    pb.compiled, key, cus, PA.kernel_name(key), PA.THREADS, PA.BLOCKS
                )
        torch.accelerator.synchronize()

    def _alloc(self, s: int) -> _Width:
        dev, h = self.device, K.HIDDEN
        mp = K.max_pad(s)

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=dev)

        return _Width(
            s=s,
            compiled=None,
            scratch=torch.zeros(K.scratch_bytes(s), dtype=torch.uint8, device=dev),
            res_out=empty((s, h), torch.bfloat16),
            xn=empty((s, h), torch.bfloat16),
            xq=empty((s, h), torch.float8_e4m3fn),
            xsc=empty((s, h // 32), torch.uint8),
            ids=empty((s, K.TOPK), torch.int32),
            w=empty((s, K.TOPK), torch.float32),
            sids=torch.zeros(mp, dtype=torch.int32, device=dev),
            sw=torch.zeros(mp, dtype=torch.float32, device=dev),
            seids=torch.zeros(mp // 16, dtype=torch.int32, device=dev),
            nv=torch.zeros(2, dtype=torch.int32, device=dev),
            moe_buf=empty((s, h), torch.bfloat16),
            xq_n=empty((s, h), torch.float8_e4m3fn) if self.want_post else None,
            xs_n=empty((s, h // 32), torch.uint8) if self.want_post else None,
            res_sh=empty((s, h), torch.bfloat16) if self.want_post else None,
            sh_inter=empty((s, self.shared_inter), torch.bfloat16)
            if self.shared_inter
            else None,
            sh_zero=(
                torch.zeros((s, h), dtype=torch.bfloat16, device=dev)
                if self.shared_inter and self.want_post
                else None
            ),
        )

    @staticmethod
    def _check_residency(
        compiled,
        key,
        cus: int,
        name: str | None = None,
        threads: int = K.THREADS,
        blocks: int = K.BLOCKS,
    ) -> None:
        """A persistent grid: every CTA must be resident at once."""
        from vllm.models.minimax_m3.amd.specialized_decode.mono.residency import (
            active_workgroups_per_cu,
        )

        name = name or K.kernel_name(key)
        per_cu = active_workgroups_per_cu(compiled, name, threads)
        if per_cu * cus < blocks:
            raise RuntimeError(
                f"{name}: {per_cu} resident CTA(s) per CU x {cus} CUs < {blocks}"
            )

    def _args(self, b: _Width, part, res_in, norm_w, router_w, bias, mode: int):
        return (
            part.data_ptr(),
            res_in.data_ptr(),
            b.res_out.data_ptr(),
            b.xn.data_ptr(),
            b.xq.data_ptr(),
            b.xsc.data_ptr(),
            b.ids.data_ptr(),
            b.w.data_ptr(),
            norm_w.data_ptr(),
            router_w.data_ptr(),
            bias.data_ptr(),
            b.sids.data_ptr(),
            b.sw.data_ptr(),
            b.seids.data_ptr(),
            b.nv.data_ptr(),
            b.moe_buf.data_ptr(),
            self.tl.data_ptr(),
            b.scratch.data_ptr(),
            self.sym,
            self.peers_ptr,
            self.rank,
            self.epoch.data_ptr(),
            mode,
            torch.cuda.current_stream(),
        )

    def _post_args(self, b: _Width, moe, sh, res, norm_w, mode: int):
        return (
            moe.data_ptr(),
            sh.data_ptr(),
            res.data_ptr(),
            norm_w.data_ptr(),
            b.xq_n.data_ptr(),
            b.xs_n.data_ptr(),
            b.res_sh.data_ptr(),
            b.scratch.data_ptr(),
            self.tl.data_ptr(),
            self.sym,
            self.peers_ptr,
            self.rank,
            self.epoch.data_ptr(),
            mode,
            torch.cuda.current_stream(),
        )

    # ------------------------------------------------------------ per layer
    def launch(self, layer, part: torch.Tensor, residual: torch.Tensor) -> _Width:
        """The glue kernel alone (collective: every rank, same layer order). Mode 2 when `residual` is the
        sharded residual the previous layer's post-MoE glue left (its owner rows only)."""
        b = self.bufs[part.shape[0]]
        moe, norm = layer.block_sparse_moe, layer.post_attention_layernorm
        sharded = b.res_sh is not None and residual.data_ptr() == b.res_sh.data_ptr()
        if (
            self.postar is not None
            and self.want_postar == "rsag8"
            and b.s in self.postar.bufs
        ):
            # rsag8 leaves the residual sharded by owner rows, in the same layout as post_kernel.py's res_sh
            sharded = (
                sharded or residual.data_ptr() == self.postar.bufs[b.s].res.data_ptr()
            )
        args = self._args(
            b,
            part.contiguous(),
            residual.contiguous(),
            norm.weight,
            moe.gate.weight,
            moe.e_score_correction_bias,
            2 if sharded else 1,
        )
        b.compiled(*args)
        return b

    def run(
        self,
        layer,
        part,
        residual,
        sw: SharedWeights,
        aiter_moe,
        md,
        q_dtype_a,
        nxt_norm=None,
    ):
        """The layer's MoE half: (all-reduced MoE output, residual after attention); with `nxt_norm` (the next
        layer's input norm) and the post-MoE glue built: (SpCarry of the next layer's MXFP8 attention input and
        the sharded residual, the sharded residual)."""
        from vllm.distributed import tensor_model_parallel_all_reduce
        from vllm.utils.torch_utils import current_stream

        b = self.launch(layer, part, residual)
        if sw.gu_bf16 is not None:
            # the shared expert runs inside the routed MoE launches (its tiles first):
            # moe_buf = routed (x2.0 in the top-k weights) + shared
            out = aiter_moe.run_sorted(
                md,
                q_dtype_a,
                b.xn,
                (b.sids, b.sw, b.seids, b.nv, b.moe_buf, None, None),
                b.ids,
                b.w,
                shared=(sw.gu_bf16, sw.dn_bf16, b.sh_inter),
            )
            if nxt_norm is not None and self._postar_ok(b.s):
                return self._run_postar(out.contiguous(), b, nxt_norm)
            if nxt_norm is not None and b.post is not None:
                b.post(
                    *self._post_args(
                        b, out.contiguous(), b.sh_zero, b.res_out, nxt_norm.weight, 1
                    )
                )
                return SpCarry(b.xq_n, b.xs_n, b.res_sh, b.s), b.res_sh
            return tensor_model_parallel_all_reduce(out), b.res_out
        mlp = layer.block_sparse_moe.shared_experts
        side = self.stream is not None and b.s <= SIDE_STREAM_MAX_ROWS
        if side:
            main = current_stream()
            self.ev_fork.record(main)
            with torch.cuda.stream(self.stream):
                self.ev_fork.wait(self.stream)
                sh = shared_decode(sw, mlp, b.xq, b.xsc)
                self.ev_join.record(self.stream)
        else:
            sh = shared_decode(sw, mlp, b.xq, b.xsc)
        # routed scale x2.0 is in the kernel's top-k weights: no output scaling
        routed = aiter_moe.run_sorted(
            md,
            q_dtype_a,
            b.xn,
            (b.sids, b.sw, b.seids, b.nv, b.moe_buf, None, None),
            b.ids,
            b.w,
        )
        if side:
            self.ev_join.wait(main)
        if nxt_norm is not None and self._postar_ok(b.s):
            # the stock chain's bf16 add (all_reduce(routed + sh)), then the one postar launch
            return self._run_postar(routed + sh, b, nxt_norm)
        if nxt_norm is not None and b.post is not None:
            b.post(
                *self._post_args(
                    b,
                    routed.contiguous(),
                    sh.contiguous(),
                    b.res_out,
                    nxt_norm.weight,
                    1,
                )
            )
            return SpCarry(b.xq_n, b.xs_n, b.res_sh, b.s), b.res_sh
        return tensor_model_parallel_all_reduce(routed + sh), b.res_out

    # ------------------------------------------------------------ postar (post-MoE all-reduce + next norm + MXFP8)
    def _postar_ok(self, s: int) -> bool:
        return self.postar is not None and s in self.postar.bufs

    def _run_postar(self, moe_out: torch.Tensor, b: _Width, nxt_norm):
        """One launch: this rank's MoE output (routed + shared) -> the next layer's MXFP8 attention input (every row)
        and the new residual (rsag8: this rank's owner rows; rsag16: every row)."""
        o = self.postar.launch(moe_out, b.res_out, nxt_norm.weight)
        return SpCarry(o.x_q, o.x_s, o.res, b.s), o.res

    # ------------------------------------------------------------ epoch
    def reset(self) -> None:
        """No launch in flight on any rank: mailboxes, scratch and the epoch
        cleared, the Lamport inboxes re-armed (the sticky error word is kept)."""
        self.peer.bytes[16:].zero_()
        K.arm_peer(self.peer.bytes)
        for b in self.bufs.values():
            b.scratch.zero_()
        self.epoch.zero_()
        torch.accelerator.synchronize()

    def assert_healthy(self) -> None:
        """Device-side: traps the queue when any rank set the error word."""
        torch._assert_async(self.fault_word[0] == 0, FAULT_MESSAGE)
        if self.postar is not None:
            self.postar.assert_healthy()

    def close(self) -> None:
        if self.postar is not None:
            self.postar.close()
            self.postar = None
        if self.peer is not None:
            self.peer.close()
            self.peer = None


def _side_stream_disabled() -> bool:
    import vllm.envs as envs

    return bool(envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM)
