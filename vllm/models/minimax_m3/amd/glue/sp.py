# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SP (sequence-parallel) MoE half for large eager steps, TP4.

Per MoE layer, rows padded to a multiple of TP, R = S_pad / TP rows a rank:

1. reduce-scatter of the attention partial (AITER custom RS, else RCCL);
2. fused add + Gemma RMSNorm (bf16 + MXFP8) of the rank's R rows with its
   residual rows;
3. router (the layer's GateLinear) and the shared expert (full replicated MXFP8
   weights, AITER preshuffled GEMM) on the R rows;
4. biased top-k (x routed scale) and MXFP4 quant of the R normed rows;
5. all-gather of the FP4 codes and of (scales | ids | weights) (AITER custom AG,
   else RCCL);
6. the stock AITER A4W4 row for M = S on the gathered, prequantized rows
   (aiter_moe.AiterMoe.run_prequant);
7. shared output + the new residual rows added into the rank's own rows of the
   routed partial, then one TP all-reduce.

With the next layer's input norm (`_post_moe`), step 7 is a reduce-scatter of the routed partial instead. With
VLLM_M3_SP_SCATTER_RS=1 and a stock row whose stage 2 ends in AITER's bf16 scatter epilogue (padded M >= 4096),
the routed MoE stops before its top-4 combine (`mxfp4_moe_scatter_reduce`) and one launch of scatter_rs_gluon.py
(VLLM_M3_SP_SCATTER_RS_IMPL=gluon, default) or scatter_rs.py (=flydsl; same protocol and bits) does
the combine and the reduce-scatter together: each rank pushes the combined rows of every token straight to the
token's owner (no [M, 6144] combined output, no F.pad copy). Bitwise the old combine + AITER custom RS.

The all-reduced result already holds the new residual, so the layer returns
(result, zero residual): the next layer's fused add (or the final norm) adds a
zero. The zero residual is a stride-0 view of one zero row (the fused-add Triton
kernels take the residual's row stride).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .shared import SharedWeights, shared_full

TOPK_GROUPS = 4  # biased_grouped_topk(.., 4, 4, ..): every group kept, a flat top-k


class SpCarry:
    """What an SP layer hands the next one instead of all-reduced rows: the next layer's attention input as MXFP8
    rows of every token (gathered), and this rank's rows of the residual (the residual stays sharded)."""

    __slots__ = ("x_q", "x_s", "res_local", "rows_total")

    def __init__(self, x_q, x_s, res_local, rows_total):
        self.x_q, self.x_s, self.res_local, self.rows_total = (
            x_q,
            x_s,
            res_local,
            rows_total,
        )

    @property
    def shape(self):
        return self.x_q.shape


class SpPrefill:
    def __init__(
        self,
        group,
        min_rows: int,
        moe_impl: str = "explicit",
        scatter_rs: bool = False,
        scatter_rs_max_rows: int = 16384,
        scatter_rs_impl: str | None = None,
    ) -> None:
        self.group = group
        self.rank = group.rank_in_group
        self.world = group.world_size
        self.min_rows = max(min_rows, self.world)
        self.moe_impl = moe_impl
        self._zero: torch.Tensor | None = None
        self.srs = None  # ScatterRs / GluonScatterRs (fused top-4 combine + reduce-scatter), once set up
        self._mark = None  # bench hook: called with a phase name between the post-MoE launches (None: off)
        self._split = False  # bench: stop stage 2 before its combine, then run that combine + pad + RS as before
        self._capture = None  # tests: a dict that receives each run's gathered top-k ids / weights (None: off)
        if scatter_rs:
            if scatter_rs_impl is None:
                from .envs import SP_SCATTER_RS_IMPL as scatter_rs_impl
            self._setup_scatter_rs(scatter_rs_max_rows, scatter_rs_impl)

    def _setup_scatter_rs(self, max_rows: int, impl: str = "gluon") -> None:
        """Collective (every TP rank, same point: SpPrefill is built in M3Glue._setup). The peer buffer
        (uncached IPC, ~153 MB at 16384 rows) is allocated here, at the first eager forward (vLLM's profile run),
        so the memory profile sees it; the kernel is compiled here too (impl gluon: a Triton JIT warmup, ~1-2 s; M,
        rows and the rank are runtime arguments, so serving never compiles another variant)."""
        import inspect
        import time

        from vllm.logger import init_logger
        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import tp_agree

        log = init_logger(__name__)
        cpu = self.group.cpu_group
        ok, err = True, None
        try:
            from aiter import fused_moe as fm

            for f in ("_flydsl_v2_stage2_wrapper", "_mxfp4_a4w4_stage2_fw"):
                if (
                    "scatter_intermediate"
                    not in inspect.signature(getattr(fm, f)).parameters
                ):
                    raise RuntimeError(
                        f"aiter.fused_moe.{f} has no scatter_intermediate (unpatched AITER)"
                    )
            # import before the collective constructor: an import error on one rank must not leave the others in
            # the peer-buffer handshake
            if impl == "gluon":
                from .scatter_rs_gluon_runtime import GluonScatterRs as _Impl
            else:
                from .scatter_rs_runtime import ScatterRs as _Impl
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
        if not tp_agree(ok, cpu):
            log.warning(
                "MiniMax-M3 SP scatter-RS off: %s", err or "refused on another rank"
            )
            return
        srs = None
        t0 = time.perf_counter()
        try:
            device = torch.device("cuda", torch.cuda.current_device())
            srs = _Impl(
                cpu, self.rank, device, max_rows=max_rows
            )  # collective handshake + compile inside
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
        if not tp_agree(ok, cpu):
            log.warning(
                "MiniMax-M3 SP scatter-RS off: setup failed on %s rank: %s",
                "this" if err else "another",
                err,
            )
            if srs is not None:
                srs.close()
            return
        self.srs = srs
        if self.rank == 0:
            log.info(
                "MiniMax-M3 SP scatter-RS ready (impl %s): up to %d rows, peer buffer %.1f MB/rank, setup + "
                "compile %.1f s (%s)",
                impl,
                srs.max_rows,
                srs.peer_nbytes / 1e6,
                time.perf_counter() - t0,
                getattr(srs, "describe", srs.compiled.__class__.__name__),
            )

    def assert_healthy(self) -> None:
        if self.srs is not None:
            self.srs.assert_healthy()

    def close(self) -> None:
        if self.srs is not None:
            self.srs.close()
            self.srs = None

    def zero_residual(self, rows: int, like: torch.Tensor) -> torch.Tensor:
        if (
            self._zero is None
            or self._zero.shape[1] != like.shape[1]
            or self._zero.dtype != like.dtype
        ):
            self._zero = torch.zeros(
                1, like.shape[1], dtype=like.dtype, device=like.device
            )
        return self._zero.expand(rows, like.shape[1])

    # ------------------------------------------------------------ collectives
    def _aiter_comm(self):
        ca = getattr(self.group.device_communicator, "aiter_ar_comm", None)
        ca = getattr(ca, "_impl", ca)
        return None if ca is None or getattr(ca, "disabled", True) else ca

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_reduce_scatter

        ca = self._aiter_comm()
        if ca is not None and ca.should_custom_rs(x, 0):
            out = torch.empty(
                (x.shape[0] // self.world, x.shape[1]), dtype=x.dtype, device=x.device
            )
            ca.custom_reduce_scatter(x, out, dim=0)
            return out
        return tensor_model_parallel_reduce_scatter(x, dim=0)

    def _all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Rows of a byte tensor from every rank, in rank order (bf16 carrier:
        AITER's IPC gather takes 16/32-bit floats only)."""
        from vllm.distributed import tensor_model_parallel_all_gather

        x16 = x.contiguous().view(torch.bfloat16)
        ca = self._aiter_comm()
        if ca is not None and ca.should_custom_ag(x16):
            return ca.custom_all_gather(x16, dim=0).view(torch.uint8)
        return tensor_model_parallel_all_gather(x16, dim=0).view(torch.uint8)

    # ------------------------------------------------------------ per layer
    def run(
        self,
        layer,
        part: torch.Tensor,
        residual: torch.Tensor,
        sw: SharedWeights,
        aiter_moe,
        nxt_norm=None,
    ):
        """`part`: this rank's unreduced o_proj output [S, H]; `residual`: [S, H], or this rank's padded rows
        [R, H] (an SpCarry's). Without `nxt_norm`: (all-reduced MoE output + new residual, zero residual). With
        the next layer's input norm: the MoE partial reduce-scattered, + residual and shared rows, the next layer's
        fused add + Gemma RMSNorm + MXFP8 quant on this rank's rows, the codes and scales all-gathered ->
        (SpCarry, this rank's new residual rows)."""
        import aiter
        from aiter import dtypes
        from aiter.ops.quant import per_1x32_mx_quant_hip

        from vllm.distributed import tensor_model_parallel_all_reduce
        from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (
            gemma_fused_add_rmsnorm_mxfp8_and_bf16,
        )

        moe, norm = layer.block_sparse_moe, layer.post_attention_layernorm
        s, h = part.shape
        rows = -(-s // self.world)
        if rows * self.world != s:
            part = F.pad(part, (0, 0, 0, rows * self.world - s))
        hs = self._reduce_scatter(part.contiguous())
        lo = self.rank * rows
        n = max(
            0, min(rows, s - lo)
        )  # this rank's real rows (the last may hold padding)
        if (
            residual.shape[0] == rows and s != rows
        ):  # this rank's rows already (sharded residual)
            res = residual
        else:
            res = residual[lo : lo + n]
            if n < rows:
                res = F.pad(res, (0, 0, 0, rows - n))
        x_q, x_s, normed, res_new = gemma_fused_add_rmsnorm_mxfp8_and_bf16(
            hs, res, norm.weight, norm.variance_epsilon
        )
        logits, _ = moe.gate(normed)
        sh = shared_full(sw, moe.shared_experts, x_q, x_s)
        topk = aiter_moe.topk
        w = torch.empty((rows, topk), dtype=torch.float32, device=part.device)
        ids = torch.empty((rows, topk), dtype=torch.int32, device=part.device)
        aiter.biased_grouped_topk(
            logits,
            moe.e_score_correction_bias,
            w,
            ids,
            TOPK_GROUPS,
            TOPK_GROUPS,
            True,
            float(moe.routed_scaling_factor),
        )
        fp4, sc = per_1x32_mx_quant_hip(
            normed, scale=None, quant_dtype=dtypes.fp4x2, shuffle=False
        )
        small = torch.cat(
            [sc.view(torch.uint8), ids.view(torch.uint8), w.view(torch.uint8)], dim=1
        )
        fp4_all = self._all_gather(fp4.view(torch.uint8))  # [S_pad, H / 2]
        small_all = self._all_gather(small)  # [S_pad, H / 32 + 8 topk]
        g = h // 32
        sc_all = small_all[:s, :g].contiguous()
        ids_all = small_all[:s, g : g + 4 * topk].contiguous().view(torch.int32)
        w_all = small_all[:s, g + 4 * topk :].contiguous().view(torch.float32)
        if self._capture is not None:
            self._capture.update(ids=ids_all, w=w_all)
        defer = None  # stage 2 stops before its top-4 combine: the intermediate goes to the fused combine + RS
        if (
            nxt_norm is not None
            and self.moe_impl != "fused"
            and (self.srs is not None or self._split)
            and (self.srs is None or self.srs.fits(s))
            and aiter_moe.scatter_deferrable(s)
        ):
            defer = {}
        if self.moe_impl == "fused":
            out = aiter_moe.run_fused_prequant(fp4_all[:s], sc_all, ids_all, w_all)
        else:  # routed partial, x routed scale (not written when ``defer`` comes back filled)
            out = aiter_moe.run_prequant(
                fp4_all[:s], sc_all, ids_all, w_all, scatter=defer
            )
        if nxt_norm is not None:
            return self._post_moe(out, s, rows, res_new, sh, nxt_norm, defer)
        if n:
            mine = out[lo : lo + n]
            mine += sh[:n]
            mine += res_new[:n]
        return tensor_model_parallel_all_reduce(out), self.zero_residual(s, out)

    def _post_moe(self, out, s, rows, res_new, sh, nxt_norm, defer=None):
        """The MoE partial reduce-scattered (bf16), + this rank's residual and shared rows, then the next layer's
        input norm with its MXFP8 quant on these rows; codes and scales all-gathered (~0.52x a bf16 gather).
        `defer` (filled by stage 2): the uncombined GEMM2 intermediate; the top-4 combine then runs fused into
        the reduce-scatter (scatter_rs.py), or as before (bench `_split`)."""
        from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (
            gemma_fused_add_rmsnorm_mxfp8,
        )

        mark = self._mark
        if mark is not None:
            mark("stage2")
        if defer and self.srs is not None and not self._split:
            # [R, H]: this rank's rows of the combined, reduce-scattered MoE output (x routed scale)
            rs = self.srs.launch(
                defer["inter"], defer["reverse_sorted"], defer["sorted_weights"], s
            )
        else:
            if (
                defer
            ):  # bench split: the combine stage 2 skipped, exactly as AITER runs it
                import aiter

                out = torch.empty(
                    (s, defer["model_dim"]),
                    dtype=torch.bfloat16,
                    device=defer["inter"].device,
                )
                aiter.mxfp4_moe_scatter_reduce(
                    flat_out=defer["inter"],
                    reverse_sorted=defer["reverse_sorted"],
                    sorted_weights=defer["sorted_weights"],
                    out=out,
                    NE=0,
                    TOPK=defer["topk"],
                    D_HIDDEN=defer["model_dim"],
                    MB=defer["MB"],
                )
            if rows * self.world != s:
                out = F.pad(out, (0, 0, 0, rows * self.world - s))
            rs = self._reduce_scatter(
                out.contiguous()
            )  # [R, H]: this rank's rows of the MoE output
        if mark is not None:
            mark("rs")
        res_in = res_new + sh[:rows]
        x_q, x_s, res_out = gemma_fused_add_rmsnorm_mxfp8(
            rs, res_in, nxt_norm.weight, nxt_norm.variance_epsilon
        )
        q_all = self._all_gather(x_q.view(torch.uint8))[:s].view(x_q.dtype)
        s_all = self._all_gather(x_s.view(torch.uint8))[:s].view(x_s.dtype)
        return SpCarry(q_all, s_all, res_out, s), res_out
