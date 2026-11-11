# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MiniMax-M3.1 TP4 SP prefill: the routed MoE's top-4 combine fused into the post-MoE reduce-scatter (gfx950).

Replaces, per MoE layer of an SP step (M rows, R = ceil(M / 4) owner rows a rank), on every rank:
  AITER mxfp4_moe_scatter_reduce (weighted 4-way sum of GEMM2's per-(token, k) rows -> [M, 6144] bf16)
  + F.pad to 4R rows + the bf16 reduce-scatter (AITER custom RS, else RCCL) -> [R, 6144] bf16.
The [M, 6144] combined output is never written: each rank reads its GEMM2 intermediate and pushes the combined rows
of every token straight to the token's owner rank.

Arithmetic (both bitwise the old path's where the old path is deterministic):
  P_r(t) = bf16(fma(x_3, w_3, fma(x_2, w_2, fma(x_1, w_1, fma(x_0, w_0, 0)))))   per rank r, sorted positions
           pos_k = reverse_sorted[4 t + k], x_k = inter[pos_k] (bf16 -> f32), w_k = sorted_weights[pos_k]; a
           position outside [0, max_sorted) is skipped: mxfp4_moe_scatter_reduce's loop, op for op;
  out(t)   = bf16(f32(P_0) + f32(P_1) + f32(P_2) + f32(P_3)), ranks in order: AITER's reduce_scatter_split_first_dim
           (packed_reduce) order. Padding rows (t >= M) are written as zeros (the old path reduce-scatters F.pad zeros).

Work split: 256 CTAs x 768 threads (a thread = one 16-byte vector of a 6144-column row). CTA b owns local rows
r = b, b + 256, ... < R. Iteration i: push row r_i (the combined rows of tokens o R + r_i for the 3 peers o) into
each peer's inbox, then reduce this rank's own row r_{i-1} (its own combined row + the 3 peers' inbox rows). Every
rank walks the same rows in the same order, so a row's three pushes land at about the time its owner wants them.

Transport: payload + flag. A push is plain 16-byte stores into the owner's inbox slot (src, r); after the CTA's
stores drained (s_waitcnt vmcnt(0) + barrier) one lane per peer stores the row flag (system scope) = this launch's
colour + 1. The owner polls the 3 flags of its row (local uncached memory), then reads the inbox rows. No sentinel,
no re-arm write. Flags carry the launch colour, so they never need clearing (only after a fault, reset()).

Single-buffered inbox (3 x R_MAX rows a rank): before its first push a CTA waits until every CTA of each owner has
finished the previous launch (`done` words, written by the owner's CTAs into every peer's buffer at their end,
after their last inbox read). Within a model step the AG between two SP layers already orders them; the check
makes back-to-back launches (tests, other callers) safe too.

Launch colour: one int32 a CTA (`colors[bid]`), read at entry, bumped at exit; every launch runs the full grid on
every rank, so the colours move in lockstep. Every wait is bounded by `ticks` (s_memrealtime, 100 MHz, a kernel
argument); a timed-out lane writes [1, rank, colour, site] to every rank's error record (peer buffer bytes 0..15) and
every waiter that sees it gives up (garbage output, host-visible via `assert_healthy`; `reset` restores).
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.deepseek_v41.amd.mono.common.ops import (
    memrealtime,
    peer_bases,
    rsrc,
    traced,
)
from vllm.models.deepseek_v41.amd.mono.common.plan import sel

_STREAM = fx.Stream(None)

HIDDEN = 6144
TP = 4
TOPK = 4
BLOCKS = 256
THREADS = 768  # a thread = one 16-byte vector (8 bf16) of a row
VECS = HIDDEN // 8
ROW_BYTES = HIDDEN * 2
ROW_DW = HIDDEN // 2  # dwords a bf16 row
ERROR_BYTES = 256
CM_SYS = 17  # sc0 | sc1: system scope
CM_NT = 2
SITE_DONE, SITE_FLAG = 1, 2
TL_POINTS = 8
DEFAULT_MAX_ROWS = (
    16384  # SP steps run below vLLM's DBO threshold (16384 tokens in the dspark cell)
)

assert VECS == THREADS


def r_max(max_rows: int) -> int:
    return -(-max_rows // TP)


def layout(max_rows: int) -> dict:
    """Peer buffer regions of one rank: (byte offset, bytes)."""
    rm = r_max(max_rows)
    flags = ERROR_BYTES
    nflag = (TP - 1) * rm * 4
    done = flags + nflag
    ndone = TP * BLOCKS * 4
    inbox = -(-(done + ndone) // 65536) * 65536
    ninbox = (TP - 1) * rm * ROW_BYTES
    return {
        "error": (0, ERROR_BYTES),
        "flags": (flags, nflag),
        "done": (done, ndone),
        "inbox": (inbox, ninbox),
    }


def peer_bytes(max_rows: int) -> int:
    end = max(o + n for o, n in layout(max_rows).values())
    return -(-end // (2 << 20)) * (2 << 20)


_SOURCES = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class SrsBuild:
    max_rows: int = DEFAULT_MAX_ROWS
    timeline: bool = False
    nt_inter: bool = (
        True  # GEMM2 intermediate loads non-temporal (each row is read once)
    )
    push_cm: int = 0  # cache bits of the inbox pushes (0 plain; the inbox is uncached memory anyway)
    lag: int = 2  # iterations between a row's push and its reduce (v1 >= 1, v2 >= 2)
    proto: str = "v2"  # v1: push, drain, flag, reduce a row at a time; v2: pipelined (see run_v2)


def kernel_name(key: SrsBuild) -> str:
    return (
        f"m31_sp_scatter_rs_{key.proto}_r{r_max(key.max_rows)}"
        + ("_tl" if key.timeline else "")
        + ("" if key.nt_inter else "_cinter")
        + (f"_cm{key.push_cm}" if key.push_cm else "")
        + f"_lag{key.lag}"
    )


# ---------------------------------------------------------------- small helpers
def _gptr(addr, ty, align):
    return fx.inttoptr(
        fx.PointerType.get(ty, address_space=fx.AddressSpace.Global, alignment=align),
        fx.Int64(addr),
    )


def load32_sys(addr):
    """A relaxed system-scope atomic load (never hoisted out of a poll loop, never cached)."""
    return fx.generic_load(
        _gptr(addr, T.i32, 4), dtype=fx.Int32, memory_order=fx.AtomicOrdering.Monotonic
    )


def store32_sys(addr, v):
    fx.generic_store(
        _gptr(addr, T.i32, 4), fx.Int32(v), memory_order=fx.AtomicOrdering.Monotonic
    )


def store64_sys(addr, v):
    fx.generic_store(
        _gptr(addr, T.i64, 8), fx.Int64(v), memory_order=fx.AtomicOrdering.Monotonic
    )


def sync_acquire():
    """System-scope acquire that orders (compiler + hardware waits) but makes nothing visible: the inbox is uncached
    memory, so no cache invalidate is needed (m3_sync's av-none fence)."""
    fence = _llvm.FenceOp(_llvm.AtomicOrdering.acquire)
    fence.attributes["llvm.mmra"] = ir.Attribute.parse(
        '#llvm.mmra_tag<"amdgcn-av":"none">'
    )


def bf_lo(w):
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def unpack8(vec):
    """A 16-byte vector of 4 dwords (bf16 pairs) -> 8 f32 in column order."""
    out = []
    for e in range(4):
        w = fx.Int32(vec[e])
        out += [bf_lo(w), bf_hi(w)]
    return out


def pack8(vals):
    """8 f32 -> 4 dwords of bf16 pairs (RNE), a 16-byte vector."""
    v = fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16)
    return v.bitcast(fx.Int32)


@traced
def stamp(c, k, cond=True):
    """Timeline point k of this CTA (thread 0), when `cond`. Points: 0 entry, 1 done-guard passed, 2 first push
    stores issued, 3 first push drained, 4 first reduce's flags seen, 5 first reduce stored, 6 walk done, 7 exit."""
    if const_expr(c["timeline"]):
        pred = c["tid"] == 0
        if const_expr(not isinstance(cond, bool)):
            pred = pred & cond
        if pred:
            bo.buffer_store(
                memrealtime(), rsrc(c["tl"]), fx.block_idx.x * TL_POINTS + k
            )


@traced
def fail(c, site):
    """[1, rank, colour, site] -> every rank's error record (lo word last: it is what waiters read)."""
    lo = (fx.Int64(fx.Int32(c["rank"])) << 32) | fx.Int64(1)
    hi = (fx.Int64(fx.Int32(site)) << 32) | fx.Int64(fx.Uint32(fx.Int32(c["colour"])))
    for b in c["bases"]:
        store64_sys(b + fx.Int64(8), hi)
    for b in c["bases"]:
        store64_sys(b, lo)


def peer_base(c, g):
    b = c["bases"][0]
    for q in range_constexpr(1, TP):
        b = sel(g == q, c["bases"][q], b)
    return b


# ---------------------------------------------------------------- the per-rank combine (scatter_reduce's math)
def token_loads(c, t):
    """Token t (uniform, in range): its 4 sorted positions (scalar loads), their weights and this thread's 16-byte
    vector of each intermediate row. -> [(valid, w, vec)] in k order (nothing waited on yet)."""
    tid = c["tid"]
    rv = fx.Vector(
        bo.buffer_load(rsrc(c["rev"]), t * TOPK, vec_width=4, is_scalar=True)
    )
    out = []
    for k in range_constexpr(TOPK):
        p = fx.Int32(rv[k])
        ok = (p >= fx.Int32(0)) & (p < c["max_sorted"])
        pc = ok.select(p, fx.Int32(0))
        w = fx.Int32(
            bo.buffer_load(rsrc(c["sw"]), pc, vec_width=1, is_scalar=True)
        ).bitcast(fx.Float32)
        x = fx.Vector(
            bo.buffer_load(
                rsrc(c["inter"]),
                pc * ROW_DW + tid * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT if c["nt_inter"] else 0,
            )
        )
        out.append((ok, w, x))
    return out


def combine(ks):
    """Acc = 0; acc = fma(x_k, w_k, acc) for the valid k in order -> 8 f32 (not yet rounded)."""
    acc = [fx.Float32(0.0)] * 8
    for ok, w, x in ks:
        xs = unpack8(x)
        acc = [
            ok.select(fx.Float32(fmath.fma(xs[e], w, acc[e])), acc[e]) for e in range(8)
        ]
    return acc


# ---------------------------------------------------------------- waits
@traced
def wait_flags(c, r):
    """Until the 3 peers' flags of owner row r (this rank's flag region, uncached) hold this launch's value.
    Every wave polls for itself (uniform addresses: one request a flag a wave)."""
    base = c["sym"] + fx.Int64(c["flag0"]) + fx.Int64(r) * 4
    stride = fx.Int64(c["rmax"] * 4)
    want = c["want"]

    def pending():
        p = fx.Int32(0)
        for s in range_constexpr(TP - 1):
            v = load32_sys(base + stride * s)
            p = p | ((v - want) < fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
        return p

    t0 = memrealtime()
    late = fx.Int64(0)
    err = fx.Int32(0)
    p = pending()
    if p != 0:
        err = load32_sys(c["err"])
        while (p != 0) & (err == 0) & (late < c["ticks"]):
            rocdl.s_sleep(1)
            p = pending()
            err = load32_sys(c["err"])
            late = memrealtime() - t0
        if (p != 0) & (err == 0):
            if c["lane"] == 0:
                fail(c, SITE_FLAG)
    sync_acquire()


@traced
def wait_done(c):
    """Single-buffer guard, once a pushing CTA: every CTA of each peer owner finished the previous launch (its
    `done` word in this rank's buffer >= this launch's colour). Thread j checks owner (rank + 1 + j // 256) % 4,
    CTA j % 256."""
    tid = c["tid"]
    o = (c["rank"] + 1 + tid // BLOCKS) % TP
    addr = c["sym"] + fx.Int64(c["done0"]) + fx.Int64(o * BLOCKS + tid % BLOCKS) * 4
    col = c["colour"]
    t0 = memrealtime()
    late = fx.Int64(0)
    err = fx.Int32(0)
    v = load32_sys(addr)
    if (v - col) < fx.Int32(0):
        err = load32_sys(c["err"])
        while ((v - col) < fx.Int32(0)) & (err == 0) & (late < c["ticks"]):
            rocdl.s_sleep(1)
            v = load32_sys(addr)
            err = load32_sys(c["err"])
            late = memrealtime() - t0
        if ((v - col) < fx.Int32(0)) & (err == 0):
            fail(c, SITE_DONE)
    sync_acquire()
    gpu.barrier()


# ---------------------------------------------------------------- push / reduce
@traced
def push_row(c, r, first=False):
    """Owner-local row r: the combined rows of tokens o R + r (o = the 3 peers) -> each peer's inbox slot
    (this rank as source, r); then, once the stores drained, the 3 row flags."""
    tid, rank, rows, m = c["tid"], c["rank"], c["rows"], c["m"]
    plan = []
    for d in range_constexpr(1, TP):
        o = (rank + d) % TP
        t = o * rows + r
        tc = fx.min(t, m - 1)
        plan.append((d, o, t, token_loads(c, tc)))
    for d, o, t, ks in plan:
        acc = combine(ks)
        slot = TP - 1 - d  # the owner's slot of this rank: (rank - o - 1) % 4
        base = (
            peer_base(c, o)
            + fx.Int64(c["inbox0"])
            + (fx.Int64(slot * c["rmax"]) + fx.Int64(r)) * ROW_BYTES
        )
        if t < m:
            bo.buffer_store(
                pack8(acc), rsrc(base), tid * 4, cache_modifier=c["push_cm"]
            )
    stamp(c, 2, first)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    stamp(c, 3, first)
    if tid < TP - 1:
        d = tid + 1
        o = (rank + d) % TP
        t = o * rows + r
        if t < m:
            fa = (
                peer_base(c, o)
                + fx.Int64(c["flag0"])
                + (fx.Int64((TP - 1 - d) * c["rmax"]) + fx.Int64(r)) * 4
            )
            store32_sys(fa, c["want"])


@traced
def reduce_row(c, r, first=False):
    """This rank's row r (token rank R + r): its own combined row + the 3 peers' inbox rows, summed in rank order
    -> out[r]; a padding row (token >= M) -> zeros."""
    tid, rank, rows, m = c["tid"], c["rank"], c["rows"], c["m"]
    t = rank * rows + r
    obase = rsrc(c["out"])
    if t < m:
        ks = token_loads(c, t)
        wait_flags(c, r)
        stamp(c, 4, first)
        ins = []
        for s in range_constexpr(TP - 1):
            ib = (
                c["sym"]
                + fx.Int64(c["inbox0"])
                + (fx.Int64(s * c["rmax"]) + fx.Int64(r)) * ROW_BYTES
            )
            ins.append(
                fx.Vector(bo.buffer_load(rsrc(ib), tid * 4, vec_width=4, dtype=T.i32))
            )
        own = pack8(
            combine(ks)
        )  # bf16-rounded: the old path reduce-scatters scatter_reduce's bf16 rows
        own_f = unpack8(own)
        in_f = [unpack8(v) for v in ins]
        acc = None
        for src in range_constexpr(TP):
            # absolute rank src: own if src == rank, else inbox slot (src - rank - 1) % 4
            sl = (fx.Int32(src) - rank - 1 + TP) % TP
            v = own_f
            for s in range_constexpr(TP - 1):
                v = [
                    sel((sl == s) & (fx.Int32(src) != rank), in_f[s][e], v[e])
                    for e in range(8)
                ]
            acc = v if acc is None else [acc[e] + v[e] for e in range(8)]
        bo.buffer_store(pack8(acc), obase, r * ROW_DW + tid * 4)
        stamp(c, 5, first)
    else:
        bo.buffer_store(
            fx.Vector.filled((4,), 0, fx.Int32), obase, r * ROW_DW + tid * 4
        )


# ---------------------------------------------------------------- walk
@traced
def run(c, key):
    bid = fx.block_idx.x
    rows = c["rows"]
    stamp(c, 0)
    n_it = (bid < rows).select((rows - bid + (BLOCKS - 1)) // BLOCKS, fx.Int32(0))
    if n_it > 0:
        wait_done(c)
    stamp(c, 1)
    lag = key.lag
    for i in range(n_it + lag):
        if i < n_it:
            push_row(c, bid + i * BLOCKS, i == 0)
        if i >= lag:
            reduce_row(c, bid + (i - lag) * BLOCKS, i == lag)
    stamp(c, 6)


# ---------------------------------------------------------------- v2: pipelined walk
NPK = (TP - 1) * TOPK  # 12 (token, k) slots a push row


def push_loads(c, r, valid):
    """Prefetch of push row r (`valid`: r is one of this CTA's rows): for each peer d = 1..3 the token o R + r,
    its 4 positions and weights (scalar loads, token clamped into range) and this thread's 16-byte vector of each
    intermediate row (masked: no traffic when the row or token is invalid). -> (positions, weights, vectors)."""
    tid, rank, rows, m = c["tid"], c["rank"], c["rows"], c["m"]
    ps, ws, xs = [], [], []
    for d in range_constexpr(1, TP):
        o = (rank + d) % TP
        t = o * rows + r
        tv = valid & (t < m)
        rv = fx.Vector(
            bo.buffer_load(
                rsrc(c["rev"]), fx.min(t, m - 1) * TOPK, vec_width=4, is_scalar=True
            )
        )
        for k in range_constexpr(TOPK):
            p = fx.Int32(rv[k])
            ok = (p >= fx.Int32(0)) & (p < c["max_sorted"])
            pc = ok.select(p, fx.Int32(0))
            ws.append(
                fx.Int32(
                    bo.buffer_load(rsrc(c["sw"]), pc, vec_width=1, is_scalar=True)
                ).bitcast(fx.Float32)
            )
            xs.append(
                fx.Vector(
                    bo.buffer_load(
                        c["inter_rs"],
                        pc * ROW_DW + tid * 4,
                        vec_width=4,
                        dtype=T.i32,
                        mask=tv & ok,
                        cache_modifier=CM_NT if c["nt_inter"] else 0,
                    )
                )
            )
            ps.append(p)
    return ps, ws, xs


def own_loads(c, t, valid):
    """Token t's 4 positions / weights / this thread's vectors (t clamped; vector loads masked by `valid`)."""
    tid, m = c["tid"], c["m"]
    rv = fx.Vector(
        bo.buffer_load(
            rsrc(c["rev"]), fx.min(t, m - 1) * TOPK, vec_width=4, is_scalar=True
        )
    )
    out = []
    for k in range_constexpr(TOPK):
        p = fx.Int32(rv[k])
        ok = (p >= fx.Int32(0)) & (p < c["max_sorted"])
        pc = ok.select(p, fx.Int32(0))
        w = fx.Int32(
            bo.buffer_load(rsrc(c["sw"]), pc, vec_width=1, is_scalar=True)
        ).bitcast(fx.Float32)
        x = fx.Vector(
            bo.buffer_load(
                c["inter_rs"],
                pc * ROW_DW + tid * 4,
                vec_width=4,
                dtype=T.i32,
                mask=valid & ok,
                cache_modifier=CM_NT if c["nt_inter"] else 0,
            )
        )
        out.append((ok, w, x))
    return out


@traced
def wait_flags_if(c, r, valid):
    """wait_flags for owner row r, only when `valid` (no load, no wait otherwise)."""
    base = c["sym"] + fx.Int64(c["flag0"]) + fx.Int64(r) * 4
    stride = fx.Int64(c["rmax"] * 4)
    want = c["want"]

    def pending():
        p = fx.Int32(0)
        for s_ in range_constexpr(TP - 1):
            v = load32_sys(base + stride * s_)
            p = p | ((v - want) < fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
        return p

    if valid:
        t0 = memrealtime()
        late = fx.Int64(0)
        err = fx.Int32(0)
        p = pending()
        if p != 0:
            err = load32_sys(c["err"])
            while (p != 0) & (err == 0) & (late < c["ticks"]):
                rocdl.s_sleep(1)
                p = pending()
                err = load32_sys(c["err"])
                late = memrealtime() - t0
            if (p != 0) & (err == 0):
                if c["lane"] == 0:
                    fail(c, SITE_FLAG)
    sync_acquire()


def push_stores(c, r, valid, ps, ws, xs):
    """Row r's combined rows (prefetched data) -> the 3 peers' inbox slots; masked stores (always issued)."""
    tid, rank, rows, m = c["tid"], c["rank"], c["rows"], c["m"]
    for d in range_constexpr(1, TP):
        o = (rank + d) % TP
        t = o * rows + r
        tv = valid & (t < m)
        ks = []
        for k in range_constexpr(TOPK):
            q = (d - 1) * TOPK + k
            p = ps[q]
            ok = (p >= fx.Int32(0)) & (p < c["max_sorted"])
            ks.append((ok, ws[q], xs[q]))
        acc = combine(ks)
        slot = TP - 1 - d  # the owner's slot of this rank: (rank - o - 1) % 4
        base = (
            peer_base(c, o)
            + fx.Int64(c["inbox0"])
            + (fx.Int64(slot * c["rmax"]) + fx.Int64(r)) * ROW_BYTES
        )
        bo.buffer_store(
            pack8(acc),
            rsrc(base, ROW_BYTES),
            tid * 4,
            mask=tv,
            cache_modifier=c["push_cm"],
        )


def rank_sum(c, own_f, in_f):
    """f32(P_0) + f32(P_1) + f32(P_2) + f32(P_3), absolute rank order (own = this rank's row)."""
    rank = c["rank"]
    acc = None
    for src in range_constexpr(TP):
        sl = (fx.Int32(src) - rank - 1 + TP) % TP
        v = own_f
        for s_ in range_constexpr(TP - 1):
            v = [
                sel((sl == s_) & (fx.Int32(src) != rank), in_f[s_][e], v[e])
                for e in range(8)
            ]
        acc = v if acc is None else [acc[e] + v[e] for e in range(8)]
    return acc


@traced
def v2_iteration(c, key, ii, n_it, cur):
    """One v2 iteration (see run_v2); `cur` = this iteration's push data (prefetched one iteration ago).
    -> the prefetch for iteration ii + 1. An iteration past the walk is a no-op (masked memory ops, no waits)."""
    bid = fx.block_idx.x
    tid, rank, rows, m = c["tid"], c["rank"], c["rows"], c["m"]
    lag = key.lag
    ps, ws, xs = cur
    # a1
    nxt = push_loads(c, bid + (ii + 1) * BLOCKS, (ii + 1) < n_it)
    j = ii - lag
    vred = (j >= fx.Int32(0)) & (j < n_it)
    rr = vred.select(bid + j * BLOCKS, bid)
    t_red = rank * rows + rr
    tok = vred & (t_red < m)
    own = own_loads(c, t_red, tok)
    # a2
    wait_flags_if(c, rr, tok)
    stamp(c, 4, ii == lag)
    # w: every wave's S(ii - 1) drained -> row ii - 1's flags
    jp = ii - 1
    vp = (jp >= fx.Int32(0)) & (jp < n_it)
    if vp:
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        stamp(c, 3, ii == 1)
        if tid < TP - 1:
            d = tid + 1
            o = (rank + d) % TP
            rp = bid + jp * BLOCKS
            if o * rows + rp < m:
                fa = (
                    peer_base(c, o)
                    + fx.Int64(c["flag0"])
                    + (fx.Int64((TP - 1 - d) * c["rmax"]) + fx.Int64(rp)) * 4
                )
                store32_sys(fa, c["want"])
    # a3
    ins = []
    for s_ in range_constexpr(TP - 1):
        ib = (
            c["sym"]
            + fx.Int64(c["inbox0"])
            + (fx.Int64(s_ * c["rmax"]) + fx.Int64(rr)) * ROW_BYTES
        )
        ins.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(ib, ROW_BYTES), tid * 4, vec_width=4, dtype=T.i32, mask=tok
                )
            )
        )
    # b
    push_stores(c, bid + ii * BLOCKS, ii < n_it, ps, ws, xs)
    stamp(c, 2, ii == 0)
    # c
    own_f = unpack8(pack8(combine(own)))  # bf16-rounded, as scatter_reduce writes it
    acc = rank_sum(c, own_f, [unpack8(v) for v in ins])
    val = tok.select(pack8(acc), fx.Vector.filled((4,), 0, fx.Int32))
    bo.buffer_store(val, c["out_rs"], rr * ROW_DW + tid * 4, mask=vred)
    stamp(c, 5, ii == lag)
    return nxt


@traced
def run_v2(c, key):
    """Iteration i (over this CTA's rows r_i = bid + 256 i, plus LAG drain iterations):
      a1  prefetch push row i+1 (next iteration's data) + own vectors of reduce row j = i - LAG
      a2  poll row j's 3 flags (a load issued after S(i-1): it returns once S(i-1) drained, in order)
      w   s_waitcnt vmcnt(0) + barrier: every wave's S(i-1) drained -> row i-1's flags to the 3 peers
      a3  row j's 3 inbox vectors
      b   push row i (S(i): its data was prefetched one iteration ago, so no load waits on it)
      c   reduce row j: own combine (bf16), + inbox rows in rank order -> out (zeros for padding tokens)
    Only a2/w wait on the link (for S(i-1)); a row's flags are a round old when its owner polls them (LAG 2).
    The runtime loop runs iteration pairs: the prefetch alternates between two register sets, so the loop-carried
    vectors need no copy at the back edge (a copy would wait for every load, hence, in order, for S(i))."""
    bid = fx.block_idx.x
    rows = c["rows"]
    lag = key.lag
    stamp(c, 0)
    n_it = (bid < rows).select((rows - bid + (BLOCKS - 1)) // BLOCKS, fx.Int32(0))
    ps0, ws0, xs0 = push_loads(c, bid, n_it > 0)
    if n_it > 0:
        wait_done(c)
    stamp(c, 1)
    total = n_it + lag
    for t, st in range(
        fx.Int32(0),
        (total + 1) // 2,
        fx.Int32(1),
        init=list(ps0) + list(ws0) + list(xs0),
    ):
        ii = fx.Int32(t) * 2
        a = (
            [fx.Int32(st[q]) for q in range(NPK)],
            [fx.Float32(st[NPK + q]) for q in range(NPK)],
            [fx.Vector(st[2 * NPK + q]) for q in range(NPK)],
        )
        b = v2_iteration(c, key, ii, n_it, a)
        a2 = v2_iteration(c, key, ii + 1, n_it, b)
        _ = yield list(a2[0]) + list(a2[1]) + list(a2[2])
    stamp(c, 6)


@traced
def run_floor(c, key):
    """Bench only (proto "floor"): this CTA's pushes alone (same loads, combine and stores, prefetched as v2), no
    flags, no reduce, no output: the push transport's floor for the M at hand."""
    bid = fx.block_idx.x
    rows = c["rows"]
    n_it = (bid < rows).select((rows - bid + (BLOCKS - 1)) // BLOCKS, fx.Int32(0))
    ps0, ws0, xs0 = push_loads(c, bid, n_it > 0)
    for t, st in range(
        fx.Int32(0),
        (n_it + 1) // 2,
        fx.Int32(1),
        init=list(ps0) + list(ws0) + list(xs0),
    ):
        ii = fx.Int32(t) * 2
        ps = [fx.Int32(st[q]) for q in range(NPK)]
        ws = [fx.Float32(st[NPK + q]) for q in range(NPK)]
        xs = [fx.Vector(st[2 * NPK + q]) for q in range(NPK)]
        b = push_loads(c, bid + (ii + 1) * BLOCKS, (ii + 1) < n_it)
        push_stores(c, bid + ii * BLOCKS, ii < n_it, ps, ws, xs)
        a2 = push_loads(c, bid + (ii + 2) * BLOCKS, (ii + 2) < n_it)
        push_stores(c, bid + (ii + 1) * BLOCKS, (ii + 1) < n_it, b[0], b[1], b[2])
        _ = yield list(a2[0]) + list(a2[1]) + list(a2[2])


def build_scatter_rs(key: SrsBuild):
    lay = layout(key.max_rows)
    rm = r_max(key.max_rows)
    keyed = (key, _SOURCES)

    @flyc.kernel(name=kernel_name(key), known_block_size=[THREADS, 1, 1])
    def ksrs(
        inter: Int64,
        rev: Int64,
        sw: Int64,
        out: Int64,
        colors: Int64,
        tl: Int64,
        sym: Int64,
        peers: Int64,
        ticks: Int64,
        m: Int32,
        rows: Int32,
        max_sorted: Int32,
        rank: Int32,
        mode_arg: Int32,
    ):
        _ = keyed
        if mode_arg != 0:
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
            colour = fx.Int32(
                bo.buffer_load(rsrc(colors), bid, vec_width=1, dtype=T.i32)
            )
            bases = peer_bases(peers, TP)
            c = {
                "tid": tid,
                "lane": tid % 64,
                "rank": rank,
                "colour": colour,
                "want": colour + 1,
                "bases": bases,
                "sym": sym,
                "err": sym,
                "ticks": ticks,
                "m": m,
                "rows": rows,
                "max_sorted": max_sorted,
                "rmax": rm,
                "flag0": lay["flags"][0],
                "done0": lay["done"][0],
                "inbox0": lay["inbox"][0],
                "inter": inter,
                "rev": rev,
                "sw": sw,
                "out": out,
                "tl": tl,
                "timeline": key.timeline,
                "nt_inter": key.nt_inter,
                "push_cm": key.push_cm,
                # sized resources: a masked access (offset 0x7FFFFFFF) must fall out of bounds
                "inter_rs": rsrc(inter, fx.Int64(max_sorted) * ROW_BYTES),
                "out_rs": rsrc(out, fx.Int64(rows) * ROW_BYTES),
            }
            if const_expr(key.proto == "v2"):
                assert key.lag >= 2, (
                    "v2 needs LAG >= 2 (row flags are published one iteration late)"
                )
                run_v2(c, key)
            elif const_expr(key.proto == "floor"):
                run_floor(c, key)
            else:
                run(c, key)
            # done: every inbox read of this CTA has returned (its values were used); then tell every peer
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if tid < TP - 1:
                o = (rank + 1 + tid) % TP
                store32_sys(
                    peer_base(c, o)
                    + fx.Int64(lay["done"][0])
                    + fx.Int64(rank * BLOCKS + bid) * 4,
                    colour + 1,
                )
            if tid == 0:
                bo.buffer_store(colour + 1, rsrc(colors), bid)
            stamp(c, 7)

    @flyc.jit
    def launch(
        inter: Int64,
        rev: Int64,
        sw: Int64,
        out: Int64,
        colors: Int64,
        tl: Int64,
        sym: Int64,
        peers: Int64,
        ticks: Int64,
        m: Int32,
        rows: Int32,
        max_sorted: Int32,
        rank: Int32,
        mode_arg: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        ksrs(
            inter,
            rev,
            sw,
            out,
            colors,
            tl,
            sym,
            peers,
            ticks,
            m,
            rows,
            max_sorted,
            rank,
            mode_arg,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
