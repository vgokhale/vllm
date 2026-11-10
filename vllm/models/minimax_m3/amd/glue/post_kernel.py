"""MiniMax-M3.1 decode post-MoE glue (TP4, gfx950, 17 .. 192 rows): one persistent FlyDSL launch from this rank's
routed MoE output + shared-expert partial to the next layer's MXFP8 attention input, with the residual stream left
sharded by owner rows.

Replaces: ``routed + shared`` (bf16 add) -> TP all-reduce (bf16) -> the next layer's Gemma fused add + RMSNorm + MXFP8
quant. The all-gather moves MXFP8 codes + E8M0 scales (1.03 B a column) instead of bf16 (2 B a column).

Stages (stage walk, BLOCKS x THREADS, S rows, G = S / 4 rows a TP owner, owner g holds rows g G .. g G + G - 1):

    push   (every CTA, 8 triples = 24 columns a task): bf16(moe + shared) of every row -> its owner's RS slot
           [this rank], tagged triple pairs (3 bf16 + 16-bit tag in one 64-bit store), peers' rows first
    owner  (CTAs 0 .. G Q - 1: Q parts of 6144 / Q columns a row of this rank's G rows, Q = 16 / 8 / 4 for G <= 16 /
           32 / 48; 6 contiguous columns a thread): the 4 ranks' triples summed in rank order (f32) -> bf16 h;
           s = h + residual (f32) -> residual out (bf16, this rank's rows only); the part's sum of squares -> the row's
           Q parts (tagged pairs in scratch, summed in part order by every part); Gemma RMSNorm of s with the next
           layer's input norm -> bf16 -> MXFP8 (vLLM's fused-norm math: per 32 columns
           sb = clamp(ceil(log2(amax / 448)) + 127, 0, 254), q = y 2^(127 - sb) as e4m3) -> a code triple (6 codes) a
           thread + the part's scale triples (4 scale bytes each) to every rank's AG slot [this rank]
    recv   (CTAs 64 .. 64 + S - 1, a CTA a row; after its owner part, if any): the row's codes and scales -> x_q [S, H] e4m3, x_s [S, H / 32] u8

The residual out is the stock fused norm's ``residual_out`` on this rank's rows; the next layer's pre-MoE glue
(glue.py, mode 2) reads only those rows: its owners add them in the reduce-scatter and gather the sum.

Sync, tags, the device-side epoch and fail-stop are the pre-MoE glue's (and the same peer buffer + epoch): every
launch of either kernel moves the epoch by one, so the two kernels alternate parity halves like consecutive launches
of one kernel.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.deepseek_v41.amd.mono.common.ops import (
    bf16_pair,
    bf16_round,
    fp8_pack4,
    peer_bases,
    rsrc,
    traced,
)
from vllm.models.deepseek_v41.amd.mono.common.plan import BLOCKS, THREADS, WAVES
from vllm.models.deepseek_v41.amd.mono.common.sync import Addr, Space, preg, sreg
from vllm.models.minimax_m3.amd.specialized_decode.mono.sync import WAIT_TICKS, load32

try:  # in the vLLM overlay: glue/kernel.py; standalone: glue.py next to this file
    from . import kernel as G
except ImportError:
    import glue as G  # noqa: F401

_STREAM = fx.Stream(None)
HIDDEN, TP, GMAX, TRIPLES = G.HIDDEN, G.TP, G.GMAX, G.TRIPLES
MIN_ROWS, MAX_ROWS = G.MIN_ROWS, G.MAX_ROWS
PUSH_TRIPLES = 8  # a push task's triples (24 columns)
PUSH_TASKS = TRIPLES // PUSH_TRIPLES  # 256: one a CTA
PUSH_LANES = THREADS // PUSH_TRIPLES  # 64 rows a pass
COLS = 6  # contiguous columns a thread in the owner stage (2 bf16 triples -> 1 code triple)
MAX_PARTS = 16
GROUPS = HIDDEN // 32  # 192 MXFP8 groups a row
CODE_TRIPLES = HIDDEN // 6  # 1024
SCALE_DW = GROUPS // 4  # 48 scale dwords a row, a triple each
AGQ_ROW = CODE_TRIPLES + SCALE_DW  # 1072 pairs a row
PAG0 = G.RS_PAIRS  # the AG region's first pair in a parity half (after the RS region, as glue.py's AG)
RECV0 = 64
SITE_PRS, SITE_PAG, SITE_PSSQ = 6, 7, 8
FP8_MAX = 448.0
F32_TINY = 1.1754943508222875e-38
TL_POINTS = G.TL_POINTS

assert PUSH_TASKS == BLOCKS and GMAX <= PUSH_LANES and GMAX <= RECV0 and RECV0 + MAX_ROWS <= BLOCKS

def owner_parts(g_rows: int) -> int:
    """Q: column parts a owner row (one CTA each), G Q <= BLOCKS."""
    return 16 if g_rows <= 16 else (8 if g_rows <= 32 else 4)


assert all(g * owner_parts(g) <= BLOCKS and HIDDEN // owner_parts(g) // COLS <= THREADS for g in range(1, GMAX + 1))
assert (PAG0 + TP * GMAX * AGQ_ROW) * 8 <= G.HALF_BYTES

_SOURCES = hashlib.sha256(Path(__file__).read_bytes() + Path(G.__file__).read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class PostBuild:
    rows: int
    timeline: bool = False  # s_memrealtime at stage boundaries -> tl [BLOCKS, TL_POINTS] i64 (points 0 .. 3, 8, 9)
    wait_ticks: int = WAIT_TICKS


def kernel_name(key: PostBuild) -> str:
    return f"m31_decode_post_s{key.rows}" + ("_tl" if key.timeline else "")


# ---------------------------------------------------------------- helpers
def cols6(buf, t, col):
    """Row t's bf16 columns col .. col + 5 (col even) of a [*, HIDDEN] buffer: 3 dwords."""
    dw = t * (HIDDEN // 2) + col // 2
    a = fx.Vector(bo.buffer_load(rsrc(buf), dw, vec_width=2, dtype=T.i32))
    b = fx.Int32(bo.buffer_load(rsrc(buf), dw + 2, vec_width=1, dtype=T.i32))
    return [fx.Int32(a[0]), fx.Int32(a[1]), b]


def dword_values(dws):
    """Dwords of bf16 pairs -> f32s, column order."""
    out = []
    for d in dws:
        out.append((d << 16).bitcast(fx.Float32))
        out.append((d & fx.Int32(-65536)).bitcast(fx.Float32))
    return out


def scale_of(amax_bits):
    """A group's E8M0 scale byte and the e4m3 multiplier 2^(127 - sb), from max |y| (f32 bits): the stock fused
    norm's clamp(ceil(log2(max(amax, tiny) / 448)) + 127, 0, 254), ceil(log2) from the bits; amax / 448 subnormal
    (an all-zero group) -> 0."""
    amax = fx.max(amax_bits.bitcast(fx.Float32), fx.Float32(F32_TINY))
    bits = (amax / fx.Float32(FP8_MAX)).bitcast(fx.Int32)
    ex = (bits >> 23) & 0xFF
    k_exp = ex - 127 + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
    sb = fx.min(fx.max(k_exp + 127, fx.Int32(0)), fx.Int32(254))
    sb = (ex == 0).select(fx.Int32(0), sb)
    return sb, ((254 - sb) << 23).bitcast(fx.Float32)


def u64(x):
    return fx.Int64(fx.Uint32(fx.Int32(x)))


def agq_pair(owner, r, q):
    return PAG0 + (owner * GMAX + r) * AGQ_ROW + q


# ---------------------------------------------------------------- push
@traced
def stage_push(c, task):
    """Triples 8 task ..: bf16(moe + shared) of every row -> its owner's RS slot [this rank], peers' rows first."""
    g_rows, rank, tid, a = c["G"], c["rank"], c["tid"], c["args"]
    tr = task * PUSH_TRIPLES + tid % PUSH_TRIPLES
    r = tid // PUSH_TRIPLES
    rr = fx.min(r, g_rows - 1)
    loads = []  # every owner's loads in flight before the first store
    for d in range_constexpr(1, TP + 1):
        g = (rank + d) % TP
        t = g * g_rows + rr
        loads.append((g, G.triple_bits(a["moe"], t, tr), G.triple_bits(a["sh"], t, tr)))
    for g, m, sh in loads:
        bits = fx.Int64(0)
        for k in range_constexpr(3):
            v = G.bf16_bits_f32(fx.Int32(m >> (16 * k))) + G.bf16_bits_f32(fx.Int32(sh >> (16 * k)))
            bits = bits | (G.bf16_bits(v) << (16 * k))
        if r < g_rows:
            c["mb"].put_triple(Addr("peer", Space.PEER, G._peer_base(c, g)), G.rs_pair(rank, r, tr), bits)


# ---------------------------------------------------------------- owner: sum + residual + next norm + MXFP8
@traced
def owner_loads(c, bid):
    """An owner part CTA's residual and norm weight (its 6 columns a thread), issued at kernel entry."""
    tid, a, qn = c["tid"], c["args"], c["Q"]
    pc = HIDDEN // qn
    z = fx.Int32(0)
    r0, r1, r2 = z, z, z
    w0, w1, w2 = z, z, z
    if bid < c["G"] * qn:
        col = (bid % qn) * pc + COLS * fx.min(tid, pc // COLS - 1)
        r0, r1, r2 = cols6(a["res"], c["rank"] * c["G"] + bid // qn, col)
        w0, w1, w2 = cols6(a["norm_w"], 0, col)
    return [r0, r1, r2], [w0, w1, w2]


@traced
def stage_owner(c, u, res, nw):
    """Owner unit u: part q = u % Q of row r = u // Q of this rank's rows (row rank G + r). Threads past the part's
    columns repeat its last thread's work (same loads, same maxima) and store nothing."""
    g_rows, rank, tid, a, mb, gmax, qn = c["G"], c["rank"], c["tid"], c["args"], c["mb"], c["gmax"], c["Q"]
    pc = HIDDEN // qn  # the part's columns
    act = pc // COLS  # its active threads
    r = u // qn
    q = u % qn
    t = rank * g_rows + r
    if tid < pc // 32:
        fx.ptr_store(fx.Int32(0), gmax + tid)
    i = fx.min(tid, act - 1)
    lc = COLS * i  # column in the part
    col = q * pc + lc
    got = mb.poll([(c["own"], G.rs_pair(src, r, col // 3), 2) for src in range(TP)], site=SITE_PRS, triples=True)
    G.stamp(c, 4)
    rv = dword_values(res)
    wv = dword_values(nw)
    s = []
    sq = fx.Float32(0.0)
    for j in range_constexpr(2):  # triple col / 3 + j = got[src][j]
        for k in range_constexpr(3):
            acc = G.triple_value(got[0][j][0], got[0][j][1], k)
            for src in range_constexpr(1, TP):
                acc = acc + G.triple_value(got[src][j][0], got[src][j][1], k)
            v = bf16_round(acc) + rv[3 * j + k]
            s.append(v)
            sq = sq + v * v
    if tid < act:
        dw = t * (HIDDEN // 2) + col // 2
        bo.buffer_store(fx.Vector.from_elements([bf16_pair(s[0], s[1]).bitcast(fx.Int32),
                                                 bf16_pair(s[2], s[3]).bitcast(fx.Int32)], fx.Int32),
                        rsrc(a["res_out"]), dw)
        bo.buffer_store(bf16_pair(s[4], s[5]).bitcast(fx.Int32), rsrc(a["res_out"]), dw + 2)
    part = G.block_sum(c, (tid < act).select(sq, fx.Float32(0.0)), 0)
    # the row's sum of squares: every part publishes its own, then sums all Q in part order (same order everywhere)
    if tid == 0:
        c["mb"].put(c["ssq"], r * MAX_PARTS + q, part)
    if tid < qn:
        got2 = c["mb"].poll([(c["ssq"], r * MAX_PARTS + tid, 1)], site=SITE_PSSQ)
        fx.ptr_store(got2[0][0].bitcast(fx.Float32), c["parts"] + tid)
    gpu.barrier()
    G.stamp(c, 5)
    tot = fx.ptr_load(c["parts"])
    for j in range_constexpr(1, qn):
        tot = tot + fx.ptr_load(c["parts"] + j)
    var = tot / fx.Float32(HIDDEN)
    rstd = fx.Float32(1.0) / fx.Float32(fmath.sqrt(var + fx.Float32(G.EPS)))
    ys = [bf16_round(s[k] * rstd * (fx.Float32(1.0) + wv[k])) for k in range(COLS)]
    # group maxima (groups of the part): a thread's 6 columns touch g0 and g1 (= g0 or g0 + 1); |y| bits as ints
    g0 = lc // 32
    g1 = (lc + COLS - 1) // 32
    m0, m1 = fx.Float32(0.0), fx.Float32(0.0)
    hi = []
    for k in range_constexpr(COLS):
        in1 = ((lc + k) // 32) != g0
        hi.append(in1)
        ab = fx.Float32(fmath.absf(ys[k]))
        m0 = in1.select(m0, fx.max(m0, ab))
        m1 = in1.select(fx.max(m1, ab), m1)
    fx.llvm.atomic_max(gmax + g0, m0.bitcast(fx.Int32), syncscope="workgroup")
    fx.llvm.atomic_max(gmax + g1, m1.bitcast(fx.Int32), syncscope="workgroup")
    gpu.barrier()
    _, inv0 = scale_of(fx.ptr_load(gmax + g0))
    _, inv1 = scale_of(fx.ptr_load(gmax + g1))
    qv = [ys[k] * hi[k].select(inv1, inv0) for k in range(COLS)]
    d0 = fp8_pack4(qv[0], qv[1], qv[2], qv[3])
    d1 = fp8_pack4(qv[4], qv[5], fx.Float32(0.0), fx.Float32(0.0))
    bits = u64(d0) | ((u64(d1) & 0xFFFF) << 32)
    if tid < act:
        for p in range_constexpr(TP):
            c["mb"].put_triple(c["peer"][p], agq_pair(rank, r, col // 6), bits)
    sdw = pc // 32 // 4  # the part's scale dwords
    if tid < sdw:
        sbs = [scale_of(fx.ptr_load(gmax + 4 * tid + k))[0] for k in range(4)]
        word = sbs[0] | (sbs[1] << 8) | (sbs[2] << 16) | (sbs[3] << 24)
        for p in range_constexpr(TP):  # c["mb"], not a local: a branch would carry the local as its state
            c["mb"].put_triple(c["peer"][p], agq_pair(rank, r, CODE_TRIPLES + q * sdw + tid), u64(word))
    G.stamp(c, 6)


# ---------------------------------------------------------------- receive: a CTA a row
@traced
def stage_recv(c, t):
    """Row t's codes (12 a thread) and scales (a dword a thread < 48) from its owner's AG slot -> x_q, x_s."""
    g_rows, tid, a, mb = c["G"], c["tid"], c["args"], c["mb"]
    owner = t // g_rows
    r = t % g_rows
    got = mb.poll([(c["own"], agq_pair(owner, r, 2 * tid), 2),
                   (c["own"], agq_pair(owner, r, CODE_TRIPLES + fx.min(tid, SCALE_DW - 1)), 1)],
                  site=SITE_PAG, triples=True)
    G.stamp(c, 7)
    (lo0, hi0), (lo1, hi1) = got[0]
    d0 = lo0
    d1 = (hi0 & 0xFFFF) | (lo1 << 16)
    d2 = ((lo1 >> 16) & 0xFFFF) | (hi1 << 16)
    dw = t * (HIDDEN // 4) + 3 * tid
    bo.buffer_store(d0, rsrc(a["xq"]), dw)
    bo.buffer_store(d1, rsrc(a["xq"]), dw + 1)
    bo.buffer_store(d2, rsrc(a["xq"]), dw + 2)
    if tid < SCALE_DW:
        bo.buffer_store(got[1][0][0], rsrc(a["xs"]), t * SCALE_DW + tid)


# ---------------------------------------------------------------- walk
@traced
def run_post(c, bid):
    s, g_rows = c["S"], c["G"]
    G.stamp(c, 0)
    res, nw = owner_loads(c, bid)  # before the push: an owner's polls must not queue behind them
    stage_push(c, bid)
    G.stamp(c, 1)
    if bid < g_rows * c["Q"]:
        stage_owner(c, bid, res, nw)
    G.stamp(c, 2)
    if (bid >= RECV0) & (bid < RECV0 + s):
        stage_recv(c, bid - RECV0)
    G.stamp(c, 3)


def build_post(key: PostBuild):
    s = key.rows
    assert MIN_ROWS <= s <= MAX_ROWS and s % TP == 0, s
    g_rows = s // TP
    qn = owner_parts(g_rows)
    keyed = (s, key.timeline, key.wait_ticks, _SOURCES)

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, 4 * WAVES * 16, 16]
        gmax: fx.Array[fx.Int32, GROUPS, 16]
        parts: fx.Array[fx.Float32, MAX_PARTS, 16]

    @flyc.kernel(name=kernel_name(key), known_block_size=[THREADS, 1, 1])
    def kpost(
        moe: Int64,
        sh: Int64,
        res: Int64,
        norm_w: Int64,
        xq: Int64,
        xs: Int64,
        res_out: Int64,
        scratch: Int64,
        tl: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        mode: Int32,
    ):
        _ = keyed
        if mode != 0:
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
            lds = fx.SharedAllocator().allocate(Lds).peek()
            ep = load32(epoch, 0, "agent")
            tag = ep + 1
            raw = peer_bases(peers, TP)
            par = fx.Int64(ep & 1) * fx.Int64(G.HALF_BYTES) + fx.Int64(G.ERROR_BYTES)
            tag16 = ((ep >> 1) + 1) & 0xFFFF
            mb = G.FastMailbox(tag, sym, raw, rank, ep, key.wait_ticks, tag16)
            c = {
                "S": s,
                "G": g_rows,
                "Q": qn,
                "tid": tid,
                "lane": tid % 64,
                "wave": tid // 64,
                "rank": rank,
                "mb": mb,
                "red": lds.red.ptr,
                "gmax": lds.gmax.ptr,
                "parts": lds.parts.ptr,
                "ssq": sreg(scratch, 0, "ssq"),
                "timeline": key.timeline,
                "own": preg(sym + par, 0, "peer"),
                "peer": [preg(b + par, 0, "peer") for b in raw],
                "args": {"moe": moe, "sh": sh, "res": res, "norm_w": norm_w, "xq": xq, "xs": xs, "res_out": res_out,
                         "tl": tl},
            }
            G.refuse_wrap(c, bid, ep)
            run_post(c, bid)
            G.stamp(c, 8)
            G.finish(c, epoch, tag)
            G.stamp(c, 9)

    @flyc.jit
    def launch(
        moe: Int64,
        sh: Int64,
        res: Int64,
        norm_w: Int64,
        xq: Int64,
        xs: Int64,
        res_out: Int64,
        scratch: Int64,
        tl: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        mode: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kpost(moe, sh, res, norm_w, xq, xs, res_out, scratch, tl, sym, peers, rank, epoch, mode).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
