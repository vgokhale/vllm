"""MiniMax-M3.1 TP4 decode post-MoE collective (gfx950, 1 .. 192 rows): one FlyDSL launch from this rank's bf16 MoE
output (routed + shared, one buffer) to the next layer's MXFP8 attention input, with the residual add, the Gemma
RMSNorm and the per-32 MXFP8 quant fused in.

Replaces: AITER cross_device_reduce_2stage (bf16 all-reduce) + gemma_fused_add_rmsnorm_mxfp8.

Three exchange schemes (``PostArBuild.mode``), the same arithmetic:

  oneshot  every rank pushes its whole partial to every peer; every rank reduces every row (row = CTA): sum in rank
           order, + residual, RMSNorm, MXFP8.                       wire: 3 x 12 KB a row a rank (12 KB a row a link)
  rsag8    reduce-scatter of the bf16 partials (rows of owner g -> rank g); the owner (a CTA a row) sums, adds the
           residual, norms, quantizes and pushes MXFP8 codes + E8M0 scales to every rank (quantized all-gather).
           The residual stays SHARDED by owner rows (res_out valid on rows rank*G .. rank*G+G-1 only).
                                                                       wire: 3 KB + 1.55 KB a row a link
  rsag16   reduce-scatter as rsag8; the owner pushes the bf16 sum h (the all-reduce result) to every rank; every
           rank then adds the residual and norms/quantizes every row itself (residual REPLICATED, stock numerics).
                                                                       wire: 3 KB + 3 KB a row a link

Sum order (all modes): f32(P_0) + f32(P_1) + f32(P_2) + f32(P_3), ranks 0..3 in order, then one bf16 rounding -> h:
the order of AITER's cross_device_reduce_{1,2}stage, so h is bitwise the AITER all-reduce output and the new
residual bf16(f32(h) + f32(res)) is bitwise the stock fused norm's residual_out.

Transport: Lamport inboxes. A payload is its own flag: every inbox 16-bit half-word holds the sentinel 0xFFFF until a
peer's push lands; a sender rewrites a sentinel half-word of its own data to 0x7FFF (still a NaN in bf16 and two NaN
e4m3 codes in MXFP8; finite data never holds 0xFFFF: bf16 0xFFFF is a NaN, an e4m3 0xFF byte is a NaN and an E8M0
scale is <= 254). The reader polls its 16-byte slots until no half-word is the sentinel (one round trip after the
last byte lands, no flag hop; a wave first spins on one probe a slot, lane 63's, so idle waiters do not flood the
inbox with uncached reads), consumes them and re-arms them (writes the sentinel back) at the end of the launch. A slot
is polled and re-armed by exactly one lane (a slot two waves polled could be re-armed by the fast one before the slow
one saw it: a lost hand-off).
Inboxes are double-buffered by launch parity: a slot of parity p is written again two launches later, which a peer
can only reach after this rank finished the next launch (it needs this rank's next push), i.e. after this launch's
re-arm stores completed (kernel boundary).

Launch parity: one int32 colour per CTA (``colors[bid]``), read at entry and bumped by the CTA at exit. Every launch
runs the full grid on every rank, so all colours move in lockstep (any row count; widths may interleave).

Every wait is bounded (``wait_ticks`` of the 100 MHz s_memrealtime clock); a timed-out lane writes
[1, rank, colour, site] to every rank's error record (peer buffer bytes 0 .. 15) and every waiter that sees a set
error record gives up: a hang becomes a garbage step + a host-visible error. After a failure the host must call
``reset`` (no launch in flight on any rank).

Thread layout: THREADS = 384, a row's 768 16-byte vectors split as vectors (tid, tid + 384): thread t holds bf16
columns 8t .. 8t+7 and 3072 + 8t .. 3072 + 8t + 7; an MXFP8 group (32 columns) is 4 consecutive lanes.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.deepseek_v41.amd.mono.common.ops import (
    bf16_round,
    butterfly,
    fp8_pack4,
    lane_gather,
    memrealtime,
    peer_bases,
    rsrc,
    traced,
    xred,
)
from vllm.models.deepseek_v41.amd.mono.common.plan import sel

_STREAM = fx.Stream(None)

HIDDEN = 6144
TP = 4
EPS = 1e-6
BLOCKS = 256
THREADS = 384
WAVES = THREADS // 64
VECS = HIDDEN // 8  # 768 16-byte bf16 vectors a row
HALF = VECS // 2  # 384 = THREADS
GROUPS = HIDDEN // 32  # 192 MXFP8 groups a row
SCALE_VECS = GROUPS // 16  # 12 16-byte scale vectors a row
AG8_VECS = HALF + SCALE_VECS  # 396: codes (a vector a thread: its 16 codes) + scales
MAX_ROWS = 192
GMAX = MAX_ROWS // TP  # 48 owner rows
FP8_MAX = 448.0
F32_TINY = 1.1754943508222875e-38
ERROR_BYTES = 256
WAIT_TICKS = 200_000_000  # 2 s
TL_POINTS = 10
MODES = ("oneshot", "rsag8", "rsag16")
SITE_RS, SITE_AG = 1, 2
SENT_SIGN = -0x7FFF8000  # 0x80008000

assert HALF == THREADS and GROUPS % 16 == 0


def rs_rows(mode: str) -> int:
    """Rows a source slot region holds: every row (oneshot) or an owner's rows."""
    return MAX_ROWS if mode == "oneshot" else GMAX


def layout(mode: str) -> dict:
    """Peer buffer regions (byte offset, bytes) of one rank, both parities."""
    rs_half = TP * rs_rows(mode) * VECS * 16
    out = {"error": (0, ERROR_BYTES), "rs": (ERROR_BYTES, 2 * rs_half)}
    off = ERROR_BYTES + 2 * rs_half
    if mode == "rsag8":
        out["ag"] = (off, 2 * TP * GMAX * AG8_VECS * 16)
    elif mode == "rsag16":
        out["ag"] = (off, 2 * TP * GMAX * VECS * 16)
    return out


def peer_bytes(mode: str) -> int:
    end = max(o + n for o, n in layout(mode).values())
    return -(-end // (2 << 20)) * (2 << 20)


_SOURCES = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class PostArBuild:
    rows: int
    mode: str = "rsag8"
    timeline: bool = False
    wait_ticks: int = WAIT_TICKS
    owner_push: bool = False  # rsag*: owner CTAs push reduce-scatter units too (default: CTAs >= G only)
    store_cm: int = 0  # cache bits of the peer pushes (0 plain, 2 nt)
    probe: bool = True  # poll phase 1: one probe a wave a slot before the full read
    sleep: int = 1  # s_sleep argument (x 64 clocks) between probe rounds
    push_per: int = 1  # rsag*: reduce-scatter units a push CTA (row-major: a CTA's first unit is an early row, so
    # early owner rows land first and their all-gather overlaps the later rows' reduce-scatter)


def kernel_name(key: PostArBuild) -> str:
    return (f"m31_postar_{key.mode}_s{key.rows}" + ("_tl" if key.timeline else "") + ("_op" if key.owner_push else "")
            + (f"_cm{key.store_cm}" if key.store_cm else "") + ("" if key.probe else "_np") + f"_z{key.sleep}" + (f"_pp{key.push_per}" if key.push_per != 1 else ""))


# ---------------------------------------------------------------- small helpers
def _gptr(addr, ty, align):
    return fx.inttoptr(fx.PointerType.get(ty, address_space=fx.AddressSpace.Global, alignment=align), fx.Int64(addr))


def ld16(base, dw):
    """4 dwords at dword ``dw`` of a (cached) buffer."""
    return fx.Vector(bo.buffer_load(rsrc(base), dw, vec_width=4, dtype=T.i32))


def st16(vec, base, dw, cm=0):
    bo.buffer_store(vec, rsrc(base), dw, cache_modifier=cm)


def words(vec, n=4):
    return [fx.Int32(vec[e]) for e in range(n)]


def sent_mask(w):
    """Bit 15 of every half-word of ``w`` that is the sentinel 0xFFFF (AITER one_shot_allreduce's test)."""
    x = w ^ fx.Int32(-1)
    nz = ((x & fx.Int32(0x7FFF7FFF)) + fx.Int32(0x7FFF7FFF)) | x
    return (nz & fx.Int32(SENT_SIGN)) ^ fx.Int32(SENT_SIGN)


def canon(ws):
    """Sentinel half-words 0xFFFF -> 0x7FFF (NaN stays NaN)."""
    return [w ^ sent_mask(w) for w in ws]


def any_sent(ws):
    acc = sent_mask(ws[0])
    for w in ws[1:]:
        acc = acc | sent_mask(w)
    return acc != fx.Int32(0)


def bf_lo(w):
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def unpack8(ws):
    """4 dwords of bf16 pairs -> 8 f32, column order."""
    out = []
    for w in ws:
        out += [bf_lo(w), bf_hi(w)]
    return out


def pack_bf16(a, b):
    """bf16(a) | bf16(b) << 16 (RNE), an Int32."""
    return fx.Int32(fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0])


def scale_of(amax):
    """E8M0 byte and 2^(127 - sb) of a group with max |y| amax (f32): the stock fused norm's
    clamp(ceil(log2(max(amax, tiny) / 448)) + 127, 0, 254), ceil(log2) from the bits."""
    a = fx.max(amax, fx.Float32(F32_TINY))
    bits = (a / fx.Float32(FP8_MAX)).bitcast(fx.Int32)
    ex = (bits >> 23) & 0xFF
    k_exp = ex - 127 + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
    sb = fx.min(fx.max(k_exp + 127, fx.Int32(0)), fx.Int32(254))
    sb = (ex == 0).select(fx.Int32(0), sb)
    return sb, ((254 - sb) << 23).bitcast(fx.Float32)


def _fmax(a, b):
    return fx.max(a, b)


@traced
def stamp(c, k):
    if const_expr(c["timeline"]):
        if c["tid"] == 0:
            bo.buffer_store(memrealtime(), rsrc(c["tl"]), fx.block_idx.x * TL_POINTS + k)


def load64(addr):
    return fx.generic_load(_gptr(addr, T.i64, 8), dtype=fx.Int64, memory_order=fx.AtomicOrdering.Monotonic)


def store64(addr, v):
    fx.generic_store(_gptr(addr, T.i64, 8), fx.Int64(v), memory_order=fx.AtomicOrdering.Monotonic)


def error_word(err_base):
    return fx.generic_load(_gptr(err_base, T.i32, 4), dtype=fx.Int32, memory_order=fx.AtomicOrdering.Monotonic)


def wave_last(a):
    """Lane 63's value of the per-lane Int64 ``a`` (both halves through ds_bpermute)."""
    lo = lane_gather(fx.Int32(a), 63)
    hi = lane_gather(fx.Int32(a >> 32), 63)
    return (fx.Int64(hi) << 32) | fx.Int64(fx.Uint32(lo))


# ---------------------------------------------------------------- Lamport poll
@traced
def poll(c, addrs, site):
    """16-byte slots at per-lane byte addresses ``addrs`` (own inbox, uncached, each list entry contiguous across
    the lanes of a wave): -> their dwords once no half-word is the sentinel.

    Phase 1 spins on one probe a wave a slot (lane 63's slot; every lane loads the same address, one request), so the
    waiters do not flood the inbox with uncached reads while the pushes land; phase 2 reads every lane's slots and
    re-reads until all are clean. Loads are system-scope relaxed atomics (not hoisted, not deleted)."""
    ticks = c["ticks"]
    lane = c["lane"]
    nw = 4 * len(addrs)
    probes = [wave_last(a) + fx.Int64(8) for a in addrs] if const_expr(c["probe"]) else []

    def load_probes():
        ws = []
        for a in probes:
            w = load64(a)
            ws += [fx.Int32(w), fx.Int32(w >> 32)]
        return fx.Vector.from_elements(ws, fx.Int32)

    def load_all():
        ws = []
        for a in addrs:
            for h in range(2):
                w = load64(a + fx.Int64(8 * h))
                ws += [fx.Int32(w), fx.Int32(w >> 32)]
        return fx.Vector.from_elements(ws, fx.Int32)

    def pending(v, n):
        return any_sent([fx.Int32(v[e]) for e in range(n)])

    t0 = memrealtime()
    late = fx.Int64(0)
    err = fx.Int32(0)
    if const_expr(c["probe"]):
        pv = load_probes()
        if pending(pv, 2 * len(probes)):
            err = error_word(c["err"])
            while pending(pv, 2 * len(probes)) & (err == 0) & (late < ticks):
                rocdl.s_sleep(c["sleep"])
                pv = load_probes()
                err = error_word(c["err"])
                late = memrealtime() - t0
    if const_expr(site == SITE_RS):
        stamp(c, 2)
    v = load_all()
    if pending(v, nw):
        err = error_word(c["err"])
        while pending(v, nw) & (err == 0) & (late < ticks):
            rocdl.s_sleep(1)
            v = load_all()
            err = error_word(c["err"])
            late = memrealtime() - t0
    if pending(v, nw) & (err == 0):
        fail(c, site)
    return [fx.Int32(v[e]) for e in range(nw)]


@traced
def fail(c, site):
    """[1, rank, colour, site] -> every rank's error record (lo word last: it is what waiters read)."""
    lo = (fx.Int64(fx.Int32(c["rank"])) << 32) | fx.Int64(1)
    hi = (fx.Int64(fx.Int32(site)) << 32) | fx.Int64(fx.Uint32(fx.Int32(c["colour"])))
    for b in c["bases"]:
        store64(b + fx.Int64(8), hi)
    for b in c["bases"]:
        store64(b, lo)


# ---------------------------------------------------------------- addresses
def peer_base(c, g):
    """Rank g's (traced, uniform) peer buffer base."""
    b = c["bases"][0]
    for q in range_constexpr(1, TP):
        b = sel(g == q, c["bases"][q], b)
    return b


def rs_off(c, src, r):
    """Byte offset of slot row (this launch's parity, source src, row r) of the RS region."""
    rows = c["rs_rows"]
    return fx.Int64(c["rs0"]) + ((fx.Int64(c["par"]) * TP + fx.Int64(src)) * rows + fx.Int64(r)) * (VECS * 16)


def ag_off(c, owner, r):
    n = c["ag_vecs"]
    return fx.Int64(c["ag0"]) + ((fx.Int64(c["par"]) * TP + fx.Int64(owner)) * GMAX + fx.Int64(r)) * (n * 16)


# ---------------------------------------------------------------- reduce-scatter / one-shot push
@traced
def stage_push(c, b0, nctas, units, rows_of):
    """Push units u = b0, b0 + nctas, .. < units (u traced): unit u = (peer d = u % 3 + 1, k = u // 3: row k // 2,
    half k % 2); this rank's partial row ``rows_of(g, r)`` half -> peer g's RS slot (this rank, r). Every unit's load
    goes out before the first store."""
    tid, rank = c["tid"], c["rank"]
    per = -(-units // nctas)
    plan = []
    for i in range_constexpr(per):
        u = b0 + i * nctas
        uu = fx.min(u, units - 1)
        d = uu % 3 + 1
        g = (rank + d) % TP
        k = uu // 3
        r = k // 2
        v = (k % 2) * HALF + tid
        val = ld16(c["inp"], rows_of(g, r) * (HIDDEN // 2) + v * 4)
        plan.append((u, g, r, v, val))
    for u, g, r, v, val in plan:
        if u < units:
            ws = canon(words(val))
            st16(fx.Vector.from_elements(ws, fx.Int32), peer_base(c, g) + rs_off(c, rank, r), v * 4, c["store_cm"])


# ---------------------------------------------------------------- row math
@traced
def block_sum(c, v):
    """Sum over the CTA, waves in order (every thread gets the same value)."""
    v = butterfly(v, (32, 16, 8, 4, 2, 1))
    if c["lane"] == 0:
        fx.ptr_store(v, c["red"] + c["wave"])
    gpu.barrier()
    tot = fx.ptr_load(c["red"])
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(c["red"] + w)
    return tot


@traced
def norm_quant(c, t, hs, resw, ww, write_res, ag_r=None):
    """Row t: s = h + residual (f32) -> res_out (if ``write_res``), Gemma RMSNorm, bf16, MXFP8 -> x_q / x_s; with
    ``ag_r`` (rsag8 owner row r): the codes and scales also to every peer's AG slot (this rank, r)."""
    tid = c["tid"]
    res = unpack8(resw[0]) + unpack8(resw[1])
    w = unpack8(ww[0]) + unpack8(ww[1])
    s = [hs[e] + res[e] for e in range(16)]
    sq = s[0] * s[0]
    for e in range_constexpr(1, 16):
        sq = sq + s[e] * s[e]
    var = block_sum(c, sq) / fx.Float32(HIDDEN)
    stamp(c, 4)
    rstd = fx.Float32(1.0) / fx.Float32(fmath.sqrt(var + fx.Float32(EPS)))
    y = [bf16_round(s[e] * rstd * (fx.Float32(1.0) + w[e])) for e in range(16)]
    codes = []
    sbs = []
    for j in range_constexpr(2):
        am = fx.Float32(fmath.absf(y[8 * j]))
        for e in range_constexpr(1, 8):
            am = fx.max(am, fx.Float32(fmath.absf(y[8 * j + e])))
        am = xred(am, 1, _fmax)
        am = xred(am, 2, _fmax)
        sb, inv = scale_of(am)
        sbs.append(sb)
        q = [y[8 * j + e] * inv for e in range(8)]
        codes += [fp8_pack4(q[0], q[1], q[2], q[3]), fp8_pack4(q[4], q[5], q[6], q[7])]
    if const_expr(ag_r is not None):
        push_ag8_codes(c, ag_r, codes)
    # x_q row t: 8 codes at byte 8 tid (dwords 2 tid ..) and at byte 3072 + 8 tid
    for j in range_constexpr(2):
        bo.buffer_store(fx.Vector.from_elements(codes[2 * j:2 * j + 2], fx.Int32), rsrc(c["xq"]),
                        t * (HIDDEN // 4) + j * (HIDDEN // 8) + 2 * tid)
    if tid % 4 == 0:
        fx.ptr_store(sbs[0], c["scl"] + tid // 4)
        fx.ptr_store(sbs[1], c["scl"] + GROUPS // 2 + tid // 4)
    gpu.barrier()
    i = fx.min(tid, SCALE_VECS - 1)
    sw = []
    for q in range_constexpr(4):
        b4 = [fx.ptr_load(c["scl"] + 16 * i + 4 * q + k) for k in range(4)]
        sw.append(b4[0] | (b4[1] << 8) | (b4[2] << 16) | (b4[3] << 24))
    if tid < SCALE_VECS:
        if const_expr(ag_r is not None):
            push_ag8_scales(c, ag_r, sw)
        st16(fx.Vector.from_elements(sw, fx.Int32), c["xs"], t * (GROUPS // 4) + 4 * tid)
    # the new residual last: nothing waits on it in this launch (the all-gather pushes go first)
    if const_expr(write_res):
        for j in range_constexpr(2):
            dws = [pack_bf16(s[8 * j + 2 * q], s[8 * j + 2 * q + 1]) for q in range(4)]
            st16(fx.Vector.from_elements(dws, fx.Int32), c["res_out"], t * (HIDDEN // 2) + (j * HALF + tid) * 4)


def own_words(c, t):
    """This rank's partial, residual and norm weight (vectors tid, tid + 384) of row t: issued before any poll."""
    tid = c["tid"]
    own = [words(ld16(c["inp"], t * (HIDDEN // 2) + (j * HALF + tid) * 4)) for j in range(2)]
    return own


def res_words(c, t):
    tid = c["tid"]
    return [words(ld16(c["res"], t * (HIDDEN // 2) + (j * HALF + tid) * 4)) for j in range(2)]


def w_words(c):
    tid = c["tid"]
    return [words(ld16(c["norm_w"], (j * HALF + tid) * 4)) for j in range(2)]


@traced
def poll_rs(c, r):
    """The three peers' row-r RS slots (vectors tid, tid + 384) of this rank's inbox -> {src: [v0, v1]} with src
    the peer's rank as a *position* d = 1..3 after this rank (src = (rank + d) % 4)."""
    tid, rank = c["tid"], c["rank"]
    addrs = []
    for d in range_constexpr(1, TP):
        src = (rank + d) % TP
        base = c["sym"] + rs_off(c, src, r)
        for j in range_constexpr(2):
            addrs.append(base + fx.Int64((j * HALF + tid) * 16))
    ws = poll(c, addrs, SITE_RS)
    return {d: [ws[8 * (d - 1) + 4 * j: 8 * (d - 1) + 4 * j + 4] for j in range(2)] for d in range(1, TP)}, addrs


def by_rank(c, own, got_d):
    """{d: words} (d = position after this rank) -> per absolute source rank, selecting at runtime."""
    rank = c["rank"]
    out = {}
    for src in range(TP):
        # absolute rank src is position d = (src - rank) % 4; d == 0 is this rank
        sel_words = []
        for j in range(2):
            ws = []
            for e in range(4):
                v = own[j][e]
                for d in range(1, TP):
                    v = sel(((rank + d) % TP) == src, got_d[d][j][e], v)
                ws.append(v)
            sel_words.append(ws)
        out[src] = sel_words
    return out


@traced
def sum_ranks(c, own, got_d):
    """h (16 f32, bf16-rounded) = rank-order sum of the four partials."""
    srcs = by_rank(c, own, got_d)
    hs = []
    for j in range_constexpr(2):
        acc = unpack8(srcs[0][j])
        for src in range_constexpr(1, TP):
            v = unpack8(srcs[src][j])
            acc = [acc[e] + v[e] for e in range(8)]
        hs += [bf16_round(a) for a in acc]
    return hs


def rearm(c, addrs):
    sent = fx.Vector.filled((4,), -1, fx.Int32)
    for a in addrs:
        fx.generic_store(_gptr(a, T.i32, 16), sent)


# ---------------------------------------------------------------- stages
def push_ag8_codes(c, r, codes):
    """This thread's 16 codes of owner row r (one vector) -> every peer's AG slot (this rank, r), vector tid."""
    tid, rank = c["tid"], c["rank"]
    cv = fx.Vector.from_elements(canon(codes), fx.Int32)
    for d in range_constexpr(1, TP):
        st16(cv, peer_base(c, (rank + d) % TP) + ag_off(c, rank, r), tid * 4, c["store_cm"])


def push_ag8_scales(c, r, sw):
    """Scale vector tid (< 12) of owner row r -> every peer's AG slot (this rank, r), vector 384 + tid."""
    tid, rank = c["tid"], c["rank"]
    sv = fx.Vector.from_elements(canon(sw), fx.Int32)
    for d in range_constexpr(1, TP):
        st16(sv, peer_base(c, (rank + d) % TP) + ag_off(c, rank, r), (HALF + tid) * 4, c["store_cm"])


@traced
def recv_ag8(c, k, g_rows):
    """Receive unit k < 3G: owner g = (rank + 1 + k % 3) % 4, row r = k // 3: codes + scales -> x_q / x_s row g G + r;
    then re-arm the slots."""
    tid, rank = c["tid"], c["rank"]
    g = (rank + 1 + k % 3) % TP
    r = k // 3
    t = g * g_rows + r
    base = c["sym"] + ag_off(c, g, r)
    a0 = base + fx.Int64(tid * 16)
    # threads >= 12 poll (and re-arm) their own code slot again: a shared scale slot polled by every wave would be
    # re-armed by a fast wave before a slow one saw it (a lost hand-off; it hung at launch 10)
    a1 = sel(tid < SCALE_VECS, base + fx.Int64((HALF + fx.min(tid, SCALE_VECS - 1)) * 16), a0)
    ws = poll(c, [a0, a1], SITE_AG)
    stamp(c, 7)
    for j in range_constexpr(2):
        bo.buffer_store(fx.Vector.from_elements(ws[2 * j:2 * j + 2], fx.Int32), rsrc(c["xq"]),
                        t * (HIDDEN // 4) + j * (HIDDEN // 8) + 2 * tid)
    if tid < SCALE_VECS:
        st16(fx.Vector.from_elements(ws[4:8], fx.Int32), c["xs"], t * (GROUPS // 4) + 4 * tid)
    rearm(c, [a0, a1])


@traced
def push_ag16(c, r, hs):
    """bf16 h of owner row r (vectors tid, tid + 384) -> every peer's AG slot (this rank, r)."""
    tid, rank = c["tid"], c["rank"]
    for j in range_constexpr(2):
        dws = canon([pack_bf16(hs[8 * j + 2 * q], hs[8 * j + 2 * q + 1]) for q in range(4)])
        v = fx.Vector.from_elements(dws, fx.Int32)
        for d in range_constexpr(1, TP):
            g = (rank + d) % TP
            st16(v, peer_base(c, g) + ag_off(c, rank, r), (j * HALF + tid) * 4, c["store_cm"])


@traced
def recv_ag16(c, k, g_rows, ww):
    """Receive unit k < 3G (owner g, row r as recv_ag8): poll h, + residual, norm, quant -> res_out / x_q / x_s."""
    tid, rank = c["tid"], c["rank"]
    g = (rank + 1 + k % 3) % TP
    r = k // 3
    t = g * g_rows + r
    resw = res_words(c, t)
    base = c["sym"] + ag_off(c, g, r)
    addrs = [base + fx.Int64((j * HALF + tid) * 16) for j in range(2)]
    ws = poll(c, addrs, SITE_AG)
    stamp(c, 7)
    hs = unpack8(ws[0:4]) + unpack8(ws[4:8])
    norm_quant(c, t, hs, resw, ww, True)
    rearm(c, addrs)


# ---------------------------------------------------------------- walk
@traced
def run(c, bid, key):
    s = key.rows
    mode = key.mode
    rank = c["rank"]
    stamp(c, 0)
    if const_expr(mode == "oneshot"):
        # owner CTA b < S: row b. Every CTA pushes: units over all rows (6 S half-rows), owners' loads first.
        is_owner = bid < s
        tt = fx.min(bid, s - 1)
        own = own_words(c, tt)
        resw = res_words(c, tt)
        ww = w_words(c)
        stage_push(c, bid, BLOCKS, 6 * s, lambda g, r: r)
        stamp(c, 1)
        if is_owner:
            got, addrs = poll_rs(c, bid)
            stamp(c, 3)
            hs = sum_ranks(c, own, got)
            norm_quant(c, bid, hs, resw, ww, True)
            stamp(c, 5)
            rearm(c, addrs)
    else:
        g_rows = s // TP
        is_owner = bid < g_rows
        tt = rank * g_rows + fx.min(bid, g_rows - 1)
        own = own_words(c, tt)
        resw = res_words(c, tt)
        ww = w_words(c)
        if const_expr(key.owner_push):
            stage_push(c, bid, BLOCKS, 6 * g_rows, lambda g, r: g * g_rows + r)
        else:
            n_push = min(BLOCKS - g_rows, -(-6 * g_rows // key.push_per))
            if (bid >= g_rows) & (bid < g_rows + n_push):
                stage_push(c, bid - g_rows, n_push, 6 * g_rows, lambda g, r: g * g_rows + r)
        stamp(c, 1)
        if is_owner:
            got, addrs = poll_rs(c, bid)
            stamp(c, 3)
            hs = sum_ranks(c, own, got)
            if const_expr(mode == "rsag8"):
                norm_quant(c, tt, hs, resw, ww, True, bid)
            else:
                push_ag16(c, bid, hs)
                norm_quant(c, tt, hs, resw, ww, True)
            stamp(c, 5)
            rearm(c, addrs)
        k = bid - g_rows
        if (bid >= g_rows) & (k < 3 * g_rows):
            stamp(c, 6)
            if const_expr(mode == "rsag8"):
                recv_ag8(c, k, g_rows)
            else:
                recv_ag16(c, k, g_rows, ww)
    stamp(c, 8)


def build_postar(key: PostArBuild):
    s, mode = key.rows, key.mode
    assert mode in MODES, mode
    assert 1 <= s <= MAX_ROWS, s
    if mode != "oneshot":
        assert s % TP == 0 and s // TP * 4 <= BLOCKS, s
    lay = layout(mode)
    keyed = (s, mode, key.timeline, key.wait_ticks, key.owner_push, key.store_cm, key.probe, key.sleep, key.push_per, _SOURCES)

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, 16, 16]
        scl: fx.Array[fx.Int32, GROUPS, 16]

    @flyc.kernel(name=kernel_name(key), known_block_size=[THREADS, 1, 1])
    def kpostar(
        inp: Int64,
        res: Int64,
        norm_w: Int64,
        xq: Int64,
        xs: Int64,
        res_out: Int64,
        colors: Int64,
        tl: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        mode_arg: Int32,
    ):
        _ = keyed
        if mode_arg != 0:
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
            lds = fx.SharedAllocator().allocate(Lds).peek()
            colour = fx.Int32(bo.buffer_load(rsrc(colors), bid, vec_width=1, dtype=T.i32))
            bases = peer_bases(peers, TP)
            c = {
                "tid": tid,
                "lane": tid % 64,
                "wave": tid // 64,
                "rank": rank,
                "colour": colour,
                "par": colour & 1,
                "bases": bases,
                "sym": sym,
                "err": sym,
                "ticks": fx.Int64(key.wait_ticks),
                "rs0": lay["rs"][0],
                "rs_rows": rs_rows(mode),
                "ag0": lay["ag"][0] if "ag" in lay else 0,
                "ag_vecs": AG8_VECS if mode == "rsag8" else VECS,
                "red": lds.red.ptr,
                "scl": lds.scl.ptr,
                "timeline": key.timeline,
                "store_cm": key.store_cm,
                "probe": key.probe,
                "sleep": key.sleep,
                "inp": inp,
                "res": res,
                "norm_w": norm_w,
                "xq": xq,
                "xs": xs,
                "res_out": res_out,
                "tl": tl,
            }
            run(c, bid, key)
            gpu.barrier()
            if tid == 0:
                bo.buffer_store(colour + 1, rsrc(colors), bid)
            stamp(c, 9)

    @flyc.jit
    def launch(
        inp: Int64,
        res: Int64,
        norm_w: Int64,
        xq: Int64,
        xs: Int64,
        res_out: Int64,
        colors: Int64,
        tl: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        mode_arg: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kpostar(inp, res, norm_w, xq, xs, res_out, colors, tl, sym, peers, rank, mode_arg).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
