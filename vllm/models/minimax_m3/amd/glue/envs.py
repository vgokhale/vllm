# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment switches of the MiniMax-M3.1 decode glue and SP prefill paths (read
once at import)."""

import os


def _flag(name: str) -> bool:
    return os.getenv(name, "0") in ("1", "true", "True")


def _widths(spec: str) -> tuple[int, ...]:
    """"24-192:8" (range, step) or "24,32,48" -> sorted row counts."""
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            rng, _, step = part.partition(":")
            lo, hi = (int(v) for v in rng.split("-"))
            out.update(range(lo, hi + 1, int(step or 8)))
        else:
            out.add(int(part))
    return tuple(sorted(out))


class _Envs:
    # decode glue kernel for pure-decode MoE layers at the widths below
    DECODE_GLUE = _flag("VLLM_M3_DECODE_GLUE")
    # row counts to compile (multiples of 4 in [17, 192]); all compiled at load
    GLUE_WIDTHS = _widths(os.getenv("VLLM_M3_GLUE_WIDTHS", "24-192:8"))
    # epoch reset every this many (conservatively counted) launches; must stay
    # below the kernel's EPOCH_LIMIT minus the uncounted startup launches
    GLUE_RESET_LAUNCHES = int(os.getenv("VLLM_M3_GLUE_RESET_LAUNCHES", str((1 << 17) - (1 << 14))))
    # SP (sequence-parallel) MoE half for eager steps of at least SP_MIN_ROWS rows
    SP_PREFILL = _flag("VLLM_M3_SP_PREFILL")
    SP_MIN_ROWS = int(os.getenv("VLLM_M3_SP_MIN_ROWS", "1024"))
    # routed MoE of the SP path: "explicit" (sort + scale sort + fused_moe_2stages
    # with the stock tuned row) or "fused" (aiter fused_moe on the prequantized input)
    SP_MOE = os.getenv("VLLM_M3_SP_MOE", "explicit")
    # decode: the all-reduce after the MoE as the post-MoE glue (reduce-scatter -> the next layer's add + norm + MXFP8
    # quant on the owner rows -> MXFP8 all-gather; the residual stays sharded by owner rows between layers)
    DECODE_POST = os.getenv("VLLM_M3_DECODE_POST", "0") in ("1", "true", "True")
    # decode: the all-reduce after the MoE + the next layer's add + Gemma RMSNorm + MXFP8 quant as ONE postar launch
    # (postar.py; Lamport inboxes, no tags): "rsag8" = bf16 reduce-scatter -> owner add/norm/quant -> MXFP8 all-gather
    # (the residual stays SHARDED: the next pre-MoE glue runs mode 2); "rsag16" = bf16 reduce-scatter -> bf16
    # all-gather of the sum -> every rank adds/norms/quantizes (residual replicated, stock numerics). Overrides
    # DECODE_POST. "" = off.
    DECODE_POSTAR = os.getenv("VLLM_M3_DECODE_POSTAR", "")
    # decode glue: the routed MoE's tuned row for exactly the step's rows when the fmoe table has one (else padded M)
    GLUE_MOE_EXACT = os.getenv("VLLM_M3_GLUE_MOE_EXACT", "1") in ("1", "true", "True")
    # decode: the shared expert inside the routed MoE launches (AITER a16w4 dense
    # shared-expert prologue, exact bf16 weights) instead of separate MXFP8 GEMMs
    GLUE_SHARED_FUSED = os.getenv("VLLM_M3_GLUE_SHARED_FUSED", "1") in ("1", "true", "True")
    # decode glue cross-GPU transport: "lamport" (reduce-scatter, and from 44 rows the all-gather too, as a plain bf16
    # payload into sentinel inboxes: no tag bytes) or "tagged" (the glue3 tagged triples)
    GLUE_RS = os.getenv("VLLM_M3_GLUE_RS", "lamport")
    # rank 0 logs path counters every N model forwards (0: off)
    LOG_EVERY = int(os.getenv("VLLM_M3_GLUE_LOG_EVERY", "0"))


envs = _Envs()
if envs.GLUE_RS not in ("lamport", "tagged"):
    raise ValueError(f"VLLM_M3_GLUE_RS={envs.GLUE_RS!r}: lamport or tagged")

# SP: the all-reduce after the MoE too (reduce-scatter -> the next layer's norm + MXFP8 quant -> MXFP8 all-gather)
SP_POST_MOE = os.environ.get("VLLM_M3_SP_POST_MOE", "1") == "1"
# SP post-MoE: the routed MoE's top-4 combine (AITER mxfp4_moe_scatter_reduce) fused into that reduce-scatter
# (scatter_rs.py; needs the AITER fused_moe.py with ``scatter_intermediate``). Only rows whose stage 2 is the bf16
# scatter epilogue (padded M >= 4096) and M <= SP_SCATTER_RS_MAX_ROWS; everything else runs the old path.
SP_SCATTER_RS = os.environ.get("VLLM_M3_SP_SCATTER_RS", "0") in ("1", "true", "True")
# rows the fused path takes at most (its uncached receive buffer: 3 x ceil(rows / 4) x 12 KB a rank, 151 MB at 16384)
SP_SCATTER_RS_MAX_ROWS = int(os.environ.get("VLLM_M3_SP_SCATTER_RS_MAX_ROWS", "16384"))
# its implementation: "gluon" (scatter_rs_gluon.py, 1024-thread CTAs: 3-5% faster in the kernel bench) or "flydsl"
# (scatter_rs.py). Same protocol, peer-buffer layout and bits (the outputs are bitwise equal).
SP_SCATTER_RS_IMPL = os.environ.get("VLLM_M3_SP_SCATTER_RS_IMPL", "gluon")
if SP_SCATTER_RS_IMPL not in ("gluon", "flydsl"):
    raise ValueError(f"VLLM_M3_SP_SCATTER_RS_IMPL={SP_SCATTER_RS_IMPL!r}: gluon or flydsl")
