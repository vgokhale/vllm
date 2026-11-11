# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model runtime of the decode glue and SP prefill paths.

Lifecycle (every phase that can fail on one rank ends in a TP agreement, so all
ranks keep or drop a path together):

* `create` (model __init__): config checks only; registers on the MoE layers.
* `after_load_weights` (end of MiniMaxM3Model.load_weights, all ranks,
  outside capture): derived shared-expert weights (SP: all-gathers), the decode
  peer buffer, per-width buffers and compiled kernels.
* `begin_forward` / `end_forward` (MiniMaxM3Model.forward): one plan a
  forward, from rank-uniform data only (forward context, attention metadata,
  row count, capture state); kept thread-locally for the decoder layers.
* `moe_half` (MiniMaxM3DecoderLayer.forward, after attention): the layer's
  MoE half on the planned path, or None for stock.
* `output_step` (MiniMaxM3ModelState.postprocess_state, every sampled step,
  eager or replayed): the periodic collective epoch reset.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import torch

from vllm.logger import init_logger

from .envs import SP_POST_MOE, SP_SCATTER_RS, SP_SCATTER_RS_MAX_ROWS, envs

logger = init_logger(__name__)

DECODE, SP = "decode", "sp"


class StepPlan(NamedTuple):
    mode: str
    rows: int


def _tp_agree(ok: bool, group) -> bool:
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import tp_agree

    return tp_agree(ok, group.cpu_group)


class M3Glue:
    @classmethod
    def create(cls, model, vllm_config) -> M3Glue | None:
        if not (envs.DECODE_GLUE or envs.SP_PREFILL):
            return None
        reasons = cls._refusals(model, vllm_config)
        if reasons:
            logger.warning("MiniMax-M3 glue paths off: %s", "; ".join(reasons))
            return None
        return cls(model, vllm_config)

    @staticmethod
    def _refusals(model, vllm_config) -> list[str]:
        from vllm.platforms import current_platform

        pc, cfg = vllm_config.parallel_config, model.config
        reasons = []
        if not current_platform.is_rocm():
            reasons.append("requires ROCm")
        else:
            from vllm.platforms.rocm import on_gfx950

            if not on_gfx950():
                reasons.append("requires gfx950")
        if pc.tensor_parallel_size != 4:
            reasons.append("requires TP 4")
        if pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1:
            reasons.append("requires PP 1 and DP 1")
        if pc.enable_expert_parallel or pc.enable_eplb:
            reasons.append("expert parallelism / EPLB")
        if getattr(vllm_config, "lora_config", None) is not None:
            reasons.append("LoRA")
        aux = getattr(vllm_config, "aux_output_config", None)
        if getattr(aux, "enable_return_routed_experts", False):
            reasons.append("routed-expert capture")
        expect = {
            "hidden_size": 6144,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "routed_scaling_factor": 2.0,
            "rms_norm_eps": 1e-6,
            "scoring_func": "sigmoid",
        }
        for k, v in expect.items():
            if getattr(cfg, k, None) != v:
                reasons.append(f"{k}={getattr(cfg, k, None)} (kernel built for {v})")
        if not getattr(cfg, "use_routing_bias", False):
            reasons.append("no routing bias")
        layers = [
            layer for layer in model.layers if getattr(layer, "is_moe_layer", False)
        ]
        if not layers or any(
            layer.block_sparse_moe.shared_experts is None for layer in layers
        ):
            reasons.append("no MoE layers or fused shared experts")
        return reasons

    def __init__(self, model, vllm_config) -> None:
        from vllm.distributed import get_tp_group

        self.group = get_tp_group()
        self.model = model
        self.layers = [
            layer for layer in model.layers if getattr(layer, "is_moe_layer", False)
        ]
        for layer in self.layers:
            layer._m3_glue = self
        self.meta_key = getattr(self.layers[0].self_attn, "layer_name", None)
        self.want_decode = envs.DECODE_GLUE
        if self.want_decode and not vllm_config.use_v2_model_runner:
            logger.warning(
                "MiniMax-M3 decode glue off: its epoch reset needs Model Runner V2"
            )
            self.want_decode = False
        bad = [s for s in envs.GLUE_WIDTHS if s % 4 or not 17 <= s <= 192]
        if self.want_decode and bad:
            raise ValueError(
                f"VLLM_M3_GLUE_WIDTHS {bad}: multiples of 4 in [17, 192] only"
            )
        self.want_sp = envs.SP_PREFILL
        self.decode = None  # DecodeGlue, once set up
        self.sp = None  # SpPrefill, once set up
        self.shared: dict[int, object] = {}  # layer id -> SharedWeights
        self._tls = threading.local()
        self._moe: dict[int, object] = {}  # layer id -> AiterMoe (lazy, host only)
        self._decode_ok: dict[
            int, tuple | None
        ] = {}  # width -> (md, q_dtype_a) or None
        self._sp_ok: dict[int, bool | None] = {}  # padded M -> True, or None if refused
        self._since_reset = 0
        self.resets = 0
        self.counts = {DECODE: 0, SP: 0, "stock": 0}
        self._forwards = 0
        self._seen: set = set()
        self._loaded = False  # load_weights finished at least once
        self._ready = False  # the setup ran (first eager forward)
        logger.info(
            "MiniMax-M3 glue: decode %s (widths %s), SP prefill %s (min rows %d, MoE %s)",
            self.want_decode,
            envs.GLUE_WIDTHS,
            self.want_sp,
            envs.SP_MIN_ROWS,
            envs.SP_MOE,
        )

    # ------------------------------------------------------------ setup
    def after_load_weights(self) -> None:
        """End of a load_weights call. The loader can call load_weights several times (one group of checkpoint
        tensors at a time) and post-processes the parameters afterwards, so nothing is derived here: the setup
        runs at the first eager, uncaptured forward (the profile run), after every weight is final."""
        if self._ready:
            logger.warning(
                "MiniMax-M3 glue: weights reloaded after setup; derived weights not rebuilt"
            )
        self._loaded = True

    def _maybe_setup(self) -> None:
        """Collective, every TP rank: the first forward is the same on all of them (the profile run)."""
        if self._ready or not self._loaded or torch.cuda.is_current_stream_capturing():
            return
        self._ready = True
        self._setup()

    def _setup(self) -> None:
        from .shared import build_shared_weights

        if not (self.want_decode or self.want_sp):
            return
        ok, err = True, None
        try:
            for layer in self.layers:
                moe, lid = layer.block_sparse_moe, layer.layer_id
                mlp = moe.shared_experts
                for lin in (mlp.gate_up_proj, mlp.down_proj):
                    if lin.weight.element_size() != 1 or not hasattr(
                        lin, "weight_scale"
                    ):
                        raise ValueError(f"layer {lid}: shared expert is not MXFP8")
                # what the glue kernel reads through raw pointers
                checks = (
                    (layer.post_attention_layernorm.weight, torch.bfloat16, (6144,)),
                    (moe.gate.weight, torch.float32, (128, 6144)),
                    (moe.e_score_correction_bias, torch.float32, (128,)),
                )
                for t, dtype, shape in checks:
                    if (
                        t.dtype != dtype
                        or tuple(t.shape) != shape
                        or not t.is_contiguous()
                    ):
                        raise ValueError(
                            f"layer {lid}: {tuple(t.shape)} {t.dtype}, want {shape} {dtype}"
                        )
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
        if not _tp_agree(ok, self.group):
            logger.warning(
                "MiniMax-M3 glue paths off: %s", err or "refused on another rank"
            )
            self.want_decode = self.want_sp = False
            return
        from .shared import fused_shared_supported

        # decode: the shared expert inside the a16w4 MoE launches (bf16 weights) when
        # the model's SwiGLU is the kernels' (alpha 1.702, beta 1, clamped)
        self.shared_fused = (
            self.want_decode
            and envs.GLUE_SHARED_FUSED
            and all(
                fused_shared_supported(layer.block_sparse_moe.shared_experts)
                for layer in self.layers
            )
        )
        for layer in self.layers:  # SP: all-gathers in layer order on every rank
            self.shared[layer.layer_id] = build_shared_weights(
                layer.block_sparse_moe.shared_experts,
                self.want_decode,
                self.want_sp,
                self.group,
                fused=self.shared_fused,
            )
        if self.want_sp:
            from .sp import SpPrefill

            self.sp = SpPrefill(
                self.group,
                envs.SP_MIN_ROWS,
                envs.SP_MOE,
                scatter_rs=SP_SCATTER_RS and SP_POST_MOE,
                scatter_rs_max_rows=SP_SCATTER_RS_MAX_ROWS,
            )
        if self.want_decode:
            self._setup_decode()
        torch.accelerator.synchronize()

    def _setup_decode(self) -> None:
        from .decode import DecodeGlue

        device = torch.device("cuda", torch.accelerator.current_device_index())
        inter_tp = self.layers[
            0
        ].block_sparse_moe.shared_experts.down_proj.weight.shape[1]
        glue = DecodeGlue(
            envs.GLUE_WIDTHS,
            self.group,
            device,
            envs.DECODE_POST and not envs.DECODE_POSTAR,
            shared_inter=inter_tp if self.shared_fused else 0,
            postar=envs.DECODE_POSTAR,
        )
        ok, err = True, None
        try:
            glue.setup()
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
        if not _tp_agree(ok, self.group):
            logger.warning(
                "MiniMax-M3 decode glue off: setup failed on %s rank: %s",
                "this" if err else "another",
                err,
            )
            glue.close()
            return
        self.decode = glue
        logger.info(
            "MiniMax-M3 decode glue ready: widths %s, shared expert %s",
            glue.widths,
            "fused into the a16w4 MoE launches (bf16)"
            if self.shared_fused
            else "separate (MXFP8 GEMMs)",
        )

    # ------------------------------------------------------------ per forward
    def begin_forward(
        self, positions: torch.Tensor, intermediate_tensors
    ) -> StepPlan | None:
        self._maybe_setup()
        plan = self._decide(positions.shape[-1], intermediate_tensors)
        self._tls.plan = plan
        self.counts[plan.mode if plan else "stock"] += 1
        self._forwards += 1
        if (
            envs.LOG_EVERY
            and self._forwards % envs.LOG_EVERY == 0
            and self.group.rank_in_group == 0
        ):
            logger.info(
                "MiniMax-M3 glue: forwards (eager/capture) by path %s", self.counts
            )
        return plan

    def end_forward(self, plan: StepPlan | None) -> None:
        self._tls.plan = None
        if plan is not None and plan.mode == DECODE:
            self.decode.assert_healthy()
        elif plan is not None and plan.mode == SP:
            self.sp.assert_healthy()  # the fused combine + reduce-scatter's error record (no-op without it)

    def _decide(self, rows: int, intermediate_tensors) -> StepPlan | None:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
        from vllm.v1.worker.ubatching import dbo_enabled

        if (
            intermediate_tensors is not None
            or not is_forward_context_available()
            or dbo_enabled()
        ):
            return None
        ctx = get_forward_context()
        if ctx.ubatch_slices is not None:
            return None
        mode = ctx.cudagraph_runtime_mode
        # FULL capture runs the forward with mode NONE; PIECEWISE graphs are keyed by
        # token count alone and replay mixed steps, so they stay stock
        if (
            self.decode is not None
            and rows in self.decode.bufs
            and mode
            in (
                CUDAGraphMode.NONE,
                CUDAGraphMode.FULL,
            )
        ):
            if self._pure_decode(ctx, rows) and self._decode_row(rows) is not None:
                return StepPlan(DECODE, rows)
        if (
            self.sp is not None
            and rows >= self.sp.min_rows
            and mode == CUDAGraphMode.NONE
            and not torch.cuda.is_current_stream_capturing()
            and self._sp_row(rows) is not None
        ):
            return StepPlan(SP, rows)
        return None

    def _pure_decode(self, ctx, rows: int) -> bool:
        from vllm.models.minimax_m3.common.sparse_attention import (
            MiniMaxM3SparseMetadata,
        )

        attn = ctx.attn_metadata
        md = attn.get(self.meta_key) if isinstance(attn, dict) else None
        if not isinstance(md, MiniMaxM3SparseMetadata):
            return False
        return (
            md.num_prefills == 0
            and md.num_prefill_tokens == 0
            and md.num_decodes > 0
            and 0 < md.num_decode_tokens <= rows
        )

    # ------------------------------------------------------------ tuned rows (host only)
    def aiter_moe(self, layer):
        moe = self._moe.get(layer.layer_id)
        if moe is None:
            from .aiter_moe import AiterMoe

            mod = layer.block_sparse_moe
            moe = getattr(mod, "_m3_aiter_moe", None) or AiterMoe.from_moe_module(mod)
            self._moe[layer.layer_id] = moe
        return moe

    def _decode_row(self, rows: int):
        """Every MoE layer's tuned A16W4 row at `rows` takes the glue's sort
        (one answer for all layers and ranks: same shapes, same CSV)."""
        if rows not in self._decode_ok:
            try:
                moe = [self.aiter_moe(layer) for layer in self.layers][
                    0
                ]  # every layer derives
                why = moe.decode_refusal(rows, envs.GLUE_MOE_EXACT)
                info = why or moe.describe(rows, envs.GLUE_MOE_EXACT)
            except Exception as e:  # noqa: BLE001 - a refusal, not a crash
                why = info = f"{type(e).__name__}: {e}"
            self._decode_ok[rows] = (
                None if why else moe.metadata(rows, envs.GLUE_MOE_EXACT)
            )
            self._note(DECODE, rows, ("refused: " if why else "") + info)
        return self._decode_ok[rows]

    def _sp_row(self, rows: int):
        from aiter.fused_moe import get_padded_M

        key = get_padded_M(rows)
        if key not in self._sp_ok:
            try:
                moe = [self.aiter_moe(layer) for layer in self.layers][
                    0
                ]  # every layer derives
                why = moe.sp_refusal(rows)
                info = why or moe.describe(rows)
            except Exception as e:  # noqa: BLE001 - a refusal, not a crash
                why = info = f"{type(e).__name__}: {e}"
            self._sp_ok[key] = None if why else True
            self._note(SP, key, ("refused: " if why else "") + info)
        return self._sp_ok[key]

    def _note(self, mode: str, rows: int, what: str) -> None:
        if self.group.rank_in_group == 0 and (mode, rows) not in self._seen:
            self._seen.add((mode, rows))
            logger.info("MiniMax-M3 glue %s rows %d (tuned row): %s", mode, rows, what)

    # ------------------------------------------------------------ per layer
    def current(self) -> StepPlan | None:
        return getattr(self._tls, "plan", None)

    def moe_half(self, layer, hidden_states: torch.Tensor, residual: torch.Tensor):
        """The layer's MoE half on this forward's path: (hidden, residual), or
        None for the stock path."""
        plan = self.current()
        if plan is None or plan.rows != hidden_states.shape[0]:
            return None
        sw = self.shared[layer.layer_id]
        moe = self.aiter_moe(layer)
        if plan.mode == DECODE:
            md, q = self._decode_row(plan.rows)  # same tuned row for every layer
            nxt = self._next_norm(layer, envs.DECODE_POST or bool(envs.DECODE_POSTAR))
            return self.decode.run(layer, hidden_states, residual, sw, moe, md, q, nxt)
        return self.sp.run(
            layer, hidden_states, residual, sw, moe, self._next_norm(layer, SP_POST_MOE)
        )

    def _next_norm(self, layer, enabled: bool):
        """The next layer's input norm when the all-reduce after this MoE can be an SP reduce-scatter + MXFP8
        gather: the next layer is a MoE layer taking MXFP8 attention input, and no drafter hidden state is taken
        between them (those need full rows); else None (all-reduce)."""
        if not enabled:
            return None
        nid = layer.layer_id + 1
        layers = self.model.layers
        if nid >= len(layers) or nid in getattr(
            self.model, "aux_hidden_state_layers", ()
        ):
            return None
        nxt = layers[nid]
        if not getattr(nxt, "is_moe_layer", False) or not getattr(
            nxt, "fuse_input_norm_mxfp8", False
        ):
            return None
        return nxt.input_layernorm

    # ------------------------------------------------------------ epoch reset
    def output_step(self) -> None:
        """Every rank, every sampled step: reset the decode epoch before the
        launches since the last reset (counted as if every step ran every MoE
        layer on the glue) could reach the kernel's tag limit."""
        if self.decode is None:
            return
        self._since_reset += 1
        # two launches a MoE layer with the post-MoE glue (it shares the epoch)
        # (postar launches keep their own device colours: they never touch the glue epoch)
        per_step = len(self.layers) * (
            2 if envs.DECODE_POST and not envs.DECODE_POSTAR else 1
        )
        if self._since_reset * per_step >= envs.GLUE_RESET_LAUNCHES:
            self.reset_epoch()

    def reset_epoch(self) -> None:
        """Collective, between steps: drain, clear, agree (a phase failing on any
        rank fails the reset on every rank)."""
        for phase, fn in (
            ("drain", torch.accelerator.synchronize),
            ("clear", self.decode.reset),
        ):
            ok, err = True, None
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                ok, err = False, e
            if not _tp_agree(ok, self.group):
                raise RuntimeError(
                    f"MiniMax-M3 decode glue epoch reset ({phase}) failed on "
                    f"{'this' if err else 'another'} rank"
                ) from err
        self._since_reset = 0
        self.resets += 1
        logger.info("MiniMax-M3 decode glue: epoch reset %d", self.resets)
