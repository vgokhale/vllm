# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gluon port of overlay/scatter_rs.py (protocol v2): the MoE top-k combine fused into the post-MoE reduce-scatter.

Same semantics, buffer layout, flags, colours, done words and error record as the FlyDSL kernel (see its docstring
and ../README.md), so the two run on the same peer buffer and can be launched interleaved:

  P_r(t) = bf16(fma chain over k = 0..TOPK-1 of x_k * w_k, skipping positions outside [0, max_sorted))  per rank r
  out(t) = bf16(f32(P_0) + f32(P_1) + f32(P_2) + f32(P_3)), ranks in order; padding rows (t >= M) zeros.

Differences forced by Gluon (none changes a bit of the result or the memory protocol):
  * a CTA is W warps (a power of two; FlyDSL: 12). Each thread holds ceil(H / 2 / (64 W)) dwords of a row, split
    into power-of-two chunks (H = 6144: W = 16 (tuned) a dwordx2 + a dword a thread; W = 8 x4 + x2; W = 4 3 x x4,
    which spills); a chunk past the row end gets a column mask (any H % 8 == 0);
  * rev / sorted_weights scalar loads are inline-asm s_load (a plain gl.load after a store becomes a vector load,
    which waits behind the peer stores: vmcnt counts both and completes in order);
  * polls, flag / done / error stores, the drain and the bf16 pack (v_cvt_pk_bf16_f32) are inline asm;
  * peer buffers are 4 pointer params, rotated on the host (peer d = rank + d), so no in-kernel select;
  * the bounded-wait budget is an i32 in microseconds (an i64 Python int that fits in i32 would re-specialize).
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

TP: gl.constexpr = gl.constexpr(4)
SITE_DONE: gl.constexpr = gl.constexpr(1)
SITE_FLAG: gl.constexpr = gl.constexpr(2)


# ---------------------------------------------------------------- geometry (constexpr)
@gluon.constexpr_function
def _widths(D, NT):
    """A thread's ceil(D / NT) dwords of a row as power-of-two chunks, widest first."""
    P = -(-D // NT)
    return [4] * (P // 4) + ([2] if P % 4 >= 2 else []) + ([1] if P % 2 else [])


@gluon.constexpr_function
def n_chunks(D, NT):
    return len(_widths(D, NT))


@gluon.constexpr_function
def chunk_w(D, NT, c):
    return _widths(D, NT)[c]


@gluon.constexpr_function
def chunk_off(D, NT, c):
    return NT * sum(_widths(D, NT)[:c])


@gluon.constexpr_function
def chunk_tail(D, NT, c):
    """Chunk c runs past the row (H / 2 not a multiple of the thread count): its ops get a column mask."""
    return NT * sum(_widths(D, NT)[:c + 1]) > D


# ---------------------------------------------------------------- inline asm (strings built once, host side)
def _sload_imm(nb, per):
    """nb bases (s u64) x per dwords at immediate offsets 0, 4, .. -> nb * per SGPR outputs, one lgkm wait."""
    n = nb * per
    lines = [f"s_load_dword ${b * per + k}, ${n + b}, {4 * k:#x}" for b in range(nb) for k in range(per)]
    lines.append("s_waitcnt lgkmcnt(0)")
    return "\n".join(lines), ",".join(["=&s"] * n + ["s"] * nb + ["~{memory}"])


def _sload_soff(n):
    """One base (s u64) + n SGPR byte offsets -> n SGPR outputs, one lgkm wait."""
    lines = [f"s_load_dword ${i}, ${n}, ${n + 1 + i}" for i in range(n)]
    lines.append("s_waitcnt lgkmcnt(0)")
    return "\n".join(lines), ",".join(["=&s"] * n + ["s"] * (n + 1) + ["~{memory}"])


def _poll(n):
    """Bounded poll, when ``act`` (s, uniform) != 0: until (v_i - want) >= 0 (signed) for the n words of every
    active lane (system-scope loads). Outputs: failed (=&s: 0 ok, 1 timed out, 2 a peer's error record seen),
    n + 1 scratch (=&v), now (=&s u64), pending (=&s u64), deadline (=&v u64). Inputs: act (v, uniform), n addresses (v),
    error-word address (v), want (v), ticks (v u64). The deadline is taken before the first check (FlyDSL's t0)."""
    fail, sc0, err, now, pend, dl = 0, 1, 1 + n, 2 + n, 3 + n, 4 + n
    act = 5 + n
    a0 = act + 1
    aerr, want, ticks = a0 + n, a0 + n + 1, a0 + n + 2
    lines = [f"s_mov_b32 ${fail}, 0", f"v_cmp_ne_u32 vcc, 0, ${act}", "s_cmp_eq_u64 vcc, 0",
             "s_cbranch_scc1 .Lpdone${:uid}",
             f"s_memrealtime ${now}", "s_waitcnt lgkmcnt(0)", f"v_lshl_add_u64 ${dl}, ${now}, 0, ${ticks}",
             ".Lpoll${:uid}:"]
    lines += [f"global_load_dword ${sc0 + i}, ${a0 + i}, off sc0 sc1" for i in range(n)]
    lines += [f"global_load_dword ${err}, ${aerr}, off sc0 sc1", "s_waitcnt vmcnt(0)", f"s_mov_b64 ${pend}, 0"]
    for i in range(n):
        lines += [f"v_sub_u32 ${sc0 + i}, ${sc0 + i}, ${want}", f"v_cmp_gt_i32 vcc, 0, ${sc0 + i}",
                  f"s_or_b64 ${pend}, ${pend}, vcc"]
    lines += [
        f"s_cmp_eq_u64 ${pend}, 0", "s_cbranch_scc1 .Lpdone${:uid}",
        f"v_cmp_ne_u32 vcc, 0, ${err}", "s_cmp_lg_u64 vcc, 0", "s_cbranch_scc1 .Lpgive${:uid}",
        f"s_memrealtime ${now}", "s_waitcnt lgkmcnt(0)",
        f"v_cmp_ge_u64 vcc, ${now}, ${dl}", "s_cmp_lg_u64 vcc, 0", "s_cbranch_scc1 .Lpout${:uid}",
        "s_sleep 1", "s_branch .Lpoll${:uid}",
        ".Lpgive${:uid}:", f"s_mov_b32 ${fail}, 2", "s_branch .Lpdone${:uid}",
        ".Lpout${:uid}:", f"s_mov_b32 ${fail}, 1",
        ".Lpdone${:uid}:",
    ]
    cons = (["=&s"] + ["=&v"] * (n + 1) + ["=&s", "=&s", "=&v"] + ["v"] + ["v"] * (n + 3)
            + ["~{memory}", "~{vcc}", "~{scc}"])
    return "\n".join(lines), ",".join(cons)


# s_waitcnt vmcnt(0) + s_barrier when the (uniform) input is non-zero: the publish of a CTA's drained stores, kept
# out of an scf.if (an if region with side effects in the walk loop made LLVM build a divergent-style loop exit
# whose dead readfirstlane of a loop-carried vector cost a vmcnt(0) at every loop header)
_DRAIN_BAR = ("\n".join(["v_cmp_ne_u32 vcc, 0, $1", "s_cmp_eq_u64 vcc, 0", "s_cbranch_scc1 .Ldb${:uid}",
                          "s_waitcnt vmcnt(0)", "s_barrier", ".Ldb${:uid}:", "s_mov_b32 $0, 0"]),
              "=s,v,~{memory},~{vcc},~{scc}")
_STORE32 = ("\n".join(["v_cmp_ne_u32 vcc, 0, $2", "s_and_saveexec_b64 $1, vcc", "s_cbranch_execz .Lst${:uid}",
                       "global_store_dword $3, $4, off sc0 sc1", ".Lst${:uid}:", "s_mov_b64 exec, $1",
                       "v_mov_b32 $0, 0"]),
            "=v,=&s,v,v,v,~{memory},~{vcc},~{scc}")
# the error record [1, rank, colour, site] into 4 buffers: the high pair first, then the low pair (the word waiters
# read), system scope, from the active lanes
_ERREC = ("\n".join(["v_cmp_ne_u32 vcc, 0, $2", "s_and_saveexec_b64 $1, vcc", "s_cbranch_execz .Ler${:uid}"]
                    + [f"global_store_dwordx2 ${3 + b}, $7, off offset:8 sc0 sc1" for b in range(4)]
                    + [f"global_store_dwordx2 ${3 + b}, $8, off sc0 sc1" for b in range(4)]
                    + [".Ler${:uid}:", "s_mov_b64 exec, $1", "v_mov_b32 $0, 0"]),
          "=v,=&s,v,v,v,v,v,v,v,~{memory},~{vcc},~{scc}")
_SLOAD_IMM = {(nb, per): _sload_imm(nb, per) for nb in (1, 2, 3, 4) for per in (1, 2, 4, 6, 8)}
_SLOAD_SOFF = {n: _sload_soff(n) for n in range(1, 33)}
_POLL = {n: _poll(n) for n in (1, 2, 3, 4)}


@gluon.constexpr_function
def sload_imm_asm(nb, per):
    return _SLOAD_IMM[(int(nb), int(per))][0]


@gluon.constexpr_function
def sload_imm_cons(nb, per):
    return _SLOAD_IMM[(int(nb), int(per))][1]


@gluon.constexpr_function
def sload_soff_asm(n):
    return _SLOAD_SOFF[int(n)][0]


@gluon.constexpr_function
def sload_soff_cons(n):
    return _SLOAD_SOFF[int(n)][1]


@gluon.constexpr_function
def poll_asm(n):
    return _POLL[int(n)][0]


@gluon.constexpr_function
def poll_cons(n):
    return _POLL[int(n)][1]


@gluon.constexpr_function
def drain_bar_asm():
    return _DRAIN_BAR[0]


@gluon.constexpr_function
def drain_bar_cons():
    return _DRAIN_BAR[1]


@gluon.constexpr_function
def store32_asm():
    return _STORE32[0]


@gluon.constexpr_function
def store32_cons():
    return _STORE32[1]


@gluon.constexpr_function
def errec_asm():
    return _ERREC[0]


@gluon.constexpr_function
def errec_cons():
    return _ERREC[1]


# ---------------------------------------------------------------- small device helpers
@gluon.jit
def thread_ids(W: gl.constexpr):
    """Thread index, one element a thread."""
    L: gl.constexpr = gl.BlockedLayout([1], [64], [W], [0])
    return gl.arange(0, W * 64, layout=L)


@gluon.jit
def cols(c: gl.constexpr, D: gl.constexpr, W: gl.constexpr):
    """Dword offsets within a row of chunk c: [64 W, w], w contiguous dwords a thread."""
    w: gl.constexpr = chunk_w(D, W * 64, c)
    off: gl.constexpr = chunk_off(D, W * 64, c)
    L: gl.constexpr = gl.BlockedLayout([1, w], [64, 1], [W, 1], [1, 0])
    r = gl.arange(0, W * 64, layout=gl.SliceLayout(1, L))
    k = gl.arange(0, w, layout=gl.SliceLayout(0, L))
    return off + r[:, None] * w + k[None, :]


@gluon.jit
def cmask(c: gl.constexpr, D: gl.constexpr, W: gl.constexpr, cond):
    """``cond`` (uniform), and the column mask of a chunk that runs past the row."""
    if chunk_tail(D, W * 64, c):
        return cond & (cols(c, D, W) < D)
    else:
        return cond


@gluon.jit
def memrealtime():
    return gl.inline_asm_elementwise("s_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s,~{memory}", [], gl.int64,
                                     False, 1)


@gluon.jit
def drain():
    """This wave's outstanding vector memory ops completed (vmcnt 0); a compiler barrier too."""
    gl.inline_asm_elementwise("s_waitcnt vmcnt(0)\ns_mov_b32 $0, 0", "=s,~{memory}", [], gl.int32, False, 1)


@gluon.jit
def store32_sys(addr, value, active):
    """System-scope 32-bit store (sc0 sc1) of ``value`` at ``addr`` where ``active`` (per-lane tensors)."""
    return gl.inline_asm_elementwise(store32_asm(), store32_cons(), [active.to(gl.int32), addr, value],
                                     (gl.int32, gl.int64), False, 1)


@gluon.jit
def fail_record(own, pr1, pr2, pr3, rank, colour, site: gl.constexpr, active):
    """[1, rank, colour, site] -> every rank's error record, from the lanes where ``active``."""
    lo = (rank.to(gl.int64) << 32) | 1
    hi = (gl.full(active.shape, site, gl.int64, active.type.layout) << 32) | colour.to(gl.uint32).to(gl.int64)
    lo = gl.full(active.shape, 0, gl.int64, active.type.layout) + lo
    gl.inline_asm_elementwise(errec_asm(), errec_cons(),
                              [active.to(gl.int32), own.to(gl.pointer_type(gl.int64)), pr1.to(gl.pointer_type(gl.int64)),
                               pr2.to(gl.pointer_type(gl.int64)), pr3.to(gl.pointer_type(gl.int64)), hi, lo],
                              (gl.int32, gl.int64), False, 1)


@gluon.jit
def fence_data(xs):
    """Each tensor of a tuple of tuples through an empty side-effecting asm (output tied to the input, memory
    clobber): the math on the data stays after this point (FENCE). At W = 8 LLVM otherwise hoisted the push
    combine's vmcnt waits above the last prefetch loads (+13..20 us); at W = 16 the hoisted combine overlaps the
    poll instead and the fence costs 5..11 us, so FENCE is off there."""
    out = ()
    for q in gl.static_range(len(xs)):
        x = ()
        for c in gl.static_range(len(xs[q])):
            x = x + (gl.inline_asm_elementwise("", "=v,0,~{memory}", [xs[q][c]], gl.int32, False, 1),)
        out = out + (x,)
    return out


@gluon.jit
def pack_bf16(lo, hi):
    """Two f32 -> one dword of bf16 (RNE), lo in bits 0..15: v_cvt_pk_bf16_f32 (the FlyDSL kernel's instruction)."""
    return gl.inline_asm_elementwise("v_cvt_pk_bf16_f32 $0, $1, $2", "=v,v,v", [lo, hi], gl.int32, True, 1)


@gluon.jit
def bf_lo(x):
    return (x << 16).to(gl.float32, bitcast=True)


@gluon.jit
def bf_hi(x):
    return (x & -65536).to(gl.float32, bitcast=True)


# ---------------------------------------------------------------- loads
@gluon.jit
def row_loads(inter, rev, sw, toks, valids, max_sorted, m,
              D: gl.constexpr, W: gl.constexpr, TOPK: gl.constexpr, NTL: gl.constexpr):
    """For each token (toks: tuple of 4 uniform tokens, valids: load the vectors or not): TOPK positions and weights
    (s_load: rev, then the weights after the vector loads went out) and this thread's dwords of each intermediate
    row (masked: no traffic when invalid). -> (positions, weights, vectors), flattened (token, k) order."""
    NCH: gl.constexpr = n_chunks(D, W * 64)
    NB: gl.constexpr = len(toks)
    bases = ()
    for b in gl.static_range(NB):
        bases = bases + (rev + gl.minimum(toks[b], m - 1) * TOPK,)
    ps = gl.inline_asm_elementwise(sload_imm_asm(NB, TOPK), sload_imm_cons(NB, TOPK), bases,
                                   (gl.int32,) * (NB * TOPK), False, 1)
    xs = ()
    offs = ()
    for q in gl.static_range(NB * TOPK):
        p = ps[q]
        ok = (p >= 0) & (p < max_sorted)
        pc = gl.where(ok, p, 0)
        offs = offs + (pc * 4,)
        v = valids[q // TOPK] & ok
        row = inter + pc.to(gl.int64) * D
        x = ()
        for c in gl.static_range(NCH):
            if NTL == 1:
                x = x + (gl.amd.cdna4.buffer_load(row, cols(c, D, W), mask=cmask(c, D, W, v), cache=".cs"),)
            else:
                x = x + (gl.amd.cdna4.buffer_load(row, cols(c, D, W), mask=cmask(c, D, W, v)),)
        xs = xs + (x,)
    wbits = gl.inline_asm_elementwise(sload_soff_asm(NB * TOPK), sload_soff_cons(NB * TOPK), (sw,) + offs,
                                      (gl.int32,) * (NB * TOPK), False, 1)
    ws = ()
    for q in gl.static_range(NB * TOPK):
        ws = ws + (wbits[q].to(gl.float32, bitcast=True),)
    return ps, ws, xs


@gluon.jit
def combine(ps, ws, xs, q0: gl.constexpr, max_sorted, D: gl.constexpr, W: gl.constexpr, TOPK: gl.constexpr):
    """acc = 0; acc = fma(x_k, w_k, acc) for the valid k in order (mxfp4_moe_scatter_reduce's loop), rounded to
    bf16 pairs: a tuple over chunks of dword tensors."""
    NCH: gl.constexpr = n_chunks(D, W * 64)
    out = ()
    for c in gl.static_range(NCH):
        alo = gl.zeros_like(xs[q0][c]).to(gl.float32)
        ahi = gl.zeros_like(xs[q0][c]).to(gl.float32)
        for k in gl.static_range(TOPK):
            p = ps[q0 + k]
            ok = (p >= 0) & (p < max_sorted)
            x = xs[q0 + k][c]
            alo = gl.where(ok, gl.fma(bf_lo(x), ws[q0 + k], alo), alo)
            ahi = gl.where(ok, gl.fma(bf_hi(x), ws[q0 + k], ahi), ahi)
        out = out + (pack_bf16(alo, ahi),)
    return out


@gluon.jit
def poll_flags(own, r, want, err_addr, ticks, active, F0: gl.constexpr, RMAX: gl.constexpr, W: gl.constexpr):
    """When ``active``: until the 3 peers' flags of owner row r (this rank's flag region) hold ``want``; every
    wave polls for itself (uniform addresses). -> failed (0 ok, 1 timed out, 2 a peer's error record seen)."""
    tid = thread_ids(W)
    z = gl.zeros_like(tid)
    f = own + F0 + r + z
    out = gl.inline_asm_elementwise(poll_asm(3), poll_cons(3),
                                    [active.to(gl.int32), f, f + RMAX, f + 2 * RMAX, err_addr + z, want + z,
                                     ticks + z.to(gl.int64)],
                                    (gl.int32, gl.int32, gl.int32, gl.int32, gl.int32, gl.int64, gl.int64, gl.int64),
                                    False, 1)
    return out[0]


@gluon.jit
def drain_barrier_if(cond):
    """When ``cond`` (uniform): every wave's vector memory ops completed, then a CTA barrier."""
    gl.inline_asm_elementwise(drain_bar_asm(), drain_bar_cons(), [cond.to(gl.int32)], gl.int32, False, 1)


# ---------------------------------------------------------------- the kernel
@gluon.jit
def iteration(inter, rev, sw, out, own, pr1, pr2, pr3, cur, ii, n_it, bid, m, rows, max_sorted, rank, want, colour,
              wait_ticks,
              D: gl.constexpr, W: gl.constexpr, TOPK: gl.constexpr, BLOCKS: gl.constexpr, RMAX: gl.constexpr,
              F0: gl.constexpr, I0: gl.constexpr, LAG: gl.constexpr, NTL: gl.constexpr, FENCE: gl.constexpr,
              DBG: gl.constexpr):
    """One v2 iteration (FlyDSL v2_iteration): a1 prefetch push row ii+1 + own vectors of reduce row j = ii - LAG;
    a2 poll row j's flags; w drain + barrier + row ii-1's flags; a3 row j's inbox; b push row ii (prefetched data);
    c reduce row j -> out. -> the prefetch for iteration ii + 1."""
    NCH: gl.constexpr = n_chunks(D, W * 64)
    NPK: gl.constexpr = (TP - 1) * TOPK
    tid = thread_ids(W)
    # ---- a1
    rn = bid + (ii + 1) * BLOCKS
    vn = (ii + 1) < n_it
    j = ii - LAG
    vred = (j >= 0) & (j < n_it)
    rr = gl.where(vred, bid + j * BLOCKS, bid)
    t_red = rank * rows + rr
    tok = vred & (t_red < m)
    t1 = ((rank + 1) % TP) * rows + rn
    t2 = ((rank + 2) % TP) * rows + rn
    t3 = ((rank + 3) % TP) * rows + rn
    ps, ws, xs = row_loads(inter, rev, sw, (t1, t2, t3, t_red), (vn & (t1 < m), vn & (t2 < m), vn & (t3 < m), tok),
                           max_sorted, m, D, W, TOPK, NTL)
    # ---- a2
    # (no scf.if in the walk: the conditions go into the asm, see _DRAIN_BAR; DBG & 32 = the scf.if form)
    if (DBG & 1) == 0:
        if (DBG & 32) != 0:
            if tok:
                failed = poll_flags(own, rr, want, own, wait_ticks, tok, F0, RMAX, W)
                fail_record(own, pr1, pr2, pr3, rank, colour, SITE_FLAG, ((tid % 64) == 0) & (failed == 1))
        else:
            failed = poll_flags(own, rr, want, own, wait_ticks, tok, F0, RMAX, W)
            fail_record(own, pr1, pr2, pr3, rank, colour, SITE_FLAG, ((tid % 64) == 0) & (failed == 1))
    # ---- w: every wave's S(ii - 1) drained -> row ii - 1's flags to the 3 peers
    jp = ii - 1
    if (DBG & 2) == 0:
        vp = (jp >= 0) & (jp < n_it)
        if (DBG & 32) != 0:
            if vp:
                drain()
                gl.barrier()
        else:
            drain_barrier_if(vp)
        rp = bid + jp * BLOCKS
        d = tid + 1
        o = (rank + d) % TP
        dst = gl.where(tid == 0, pr1, gl.where(tid == 1, pr2, pr3))
        fa = dst + F0 + (TP - 1 - d) * RMAX + rp
        store32_sys(fa, want + gl.zeros_like(tid), (tid < TP - 1) & (o * rows + rp < m) & vp)
    # ---- a3: row j's inbox rows, by absolute source rank (own rank: masked, no load)
    ins = ()
    for src in gl.static_range(TP):
        sl = (src - rank - 1 + TP) % TP
        slc = gl.where(src == rank, 0, sl)
        base = own + I0 + (slc * RMAX + rr).to(gl.int64) * D
        v = tok & (src != rank)
        x = ()
        for c in gl.static_range(NCH):
            x = x + (gl.amd.cdna4.buffer_load(base, cols(c, D, W), mask=cmask(c, D, W, v)),)
        ins = ins + (x,)
    # ---- b: push row ii from the data prefetched one iteration ago
    cps, cws, cxs = cur
    if FENCE:
        cxs = fence_data(cxs)
    r_push = bid + ii * BLOCKS
    vpush = ii < n_it
    for d in gl.static_range(1, TP if (DBG & 8) == 0 else 1):
        tp_ = ((rank + d) % TP) * rows + r_push
        pk = combine(cps, cws, cxs, (d - 1) * TOPK, max_sorted, D, W, TOPK)
        if d == 1:
            dst = pr1
        elif d == 2:
            dst = pr2
        else:
            dst = pr3
        base = dst + I0 + ((TP - 1 - d) * RMAX + r_push).to(gl.int64) * D
        for c in gl.static_range(NCH):
            gl.amd.cdna4.buffer_store(pk[c], base, cols(c, D, W), mask=cmask(c, D, W, vpush & (tp_ < m)))
    # ---- c: reduce row j: own combine (bf16) + inbox rows, absolute rank order -> out
    own_pk = combine(ps, ws, fence_data(xs) if FENCE else xs, NPK, max_sorted, D, W, TOPK)
    obase = out + rr.to(gl.int64) * D
    for c in gl.static_range(NCH if (DBG & 4) == 0 else 0):
        olo = bf_lo(own_pk[c])
        ohi = bf_hi(own_pk[c])
        alo = gl.where(rank == 0, olo, bf_lo(ins[0][c]))
        ahi = gl.where(rank == 0, ohi, bf_hi(ins[0][c]))
        for src in gl.static_range(1, TP):
            alo = alo + gl.where(rank == src, olo, bf_lo(ins[src][c]))
            ahi = ahi + gl.where(rank == src, ohi, bf_hi(ins[src][c]))
        val = gl.where(tok, pack_bf16(alo, ahi), 0)
        gl.amd.cdna4.buffer_store(val, obase, cols(c, D, W), mask=cmask(c, D, W, vred))
    # the next iteration's push data: the 3 peers' tokens of row ii + 1
    nps = ()
    nws = ()
    nxs = ()
    for q in gl.static_range(NPK):
        nps = nps + (ps[q],)
        nws = nws + (ws[q],)
        nxs = nxs + (xs[q],)
    return nps, nws, nxs


@gluon.jit(do_not_specialize=["wait_us", "m", "rows", "max_sorted", "rank"])
def scatter_rs_kernel(inter, rev, sw, out, colors, own, pr1, pr2, pr3, wait_us, m, rows, max_sorted, rank,
                      H: gl.constexpr, TOPK: gl.constexpr, W: gl.constexpr, BLOCKS: gl.constexpr,
                      RMAX: gl.constexpr, FLAG0: gl.constexpr, DONE0: gl.constexpr, INBOX0: gl.constexpr,
                      LAG: gl.constexpr, NTL: gl.constexpr, UNROLL: gl.constexpr = 2, FENCE: gl.constexpr = 0,
                      DBG: gl.constexpr = 0):
    """inter: [max_sorted, H] bf16 as dwords; rev: [M TOPK] i32; sw: f32; out: [R, H] bf16 as dwords; colors:
    [BLOCKS] i32; own / pr1..3: the peer buffers (as i32) of ranks rank, rank+1, rank+2, rank+3; FLAG0 / DONE0 /
    INBOX0: byte offsets of the regions (scatter_rs.layout)."""
    D: gl.constexpr = H // 2
    F0: gl.constexpr = FLAG0 // 4
    DN0: gl.constexpr = DONE0 // 4
    I0: gl.constexpr = INBOX0 // 4
    NDONE: gl.constexpr = (TP - 1) * BLOCKS
    NQ: gl.constexpr = (NDONE + W * 64 - 1) // (W * 64)
    gl.static_assert(H % 8 == 0, "H: a multiple of 8 (whole 16-byte vectors)")
    gl.static_assert(LAG >= 2, "v2 needs LAG >= 2")
    gl.static_assert(NQ <= 4, "done-guard poll: <= 4 words a thread")
    bid = gl.program_id(0)
    tid = thread_ids(W)
    colour = gl.load(colors + bid)
    want = colour + 1
    wait_ticks = wait_us.to(gl.int64) * 100  # s_memrealtime: 100 MHz
    n_it = gl.where(bid < rows, (rows - bid + (BLOCKS - 1)) // BLOCKS, 0)
    # prefetch of the first push row (local data), then the single-buffer guard
    rn = bid
    t1 = ((rank + 1) % TP) * rows + rn
    t2 = ((rank + 2) % TP) * rows + rn
    t3 = ((rank + 3) % TP) * rows + rn
    v0 = n_it > 0
    cur = row_loads(inter, rev, sw, (t1, t2, t3), (v0 & (t1 < m), v0 & (t2 < m), v0 & (t3 < m)), max_sorted, m,
                    D, W, TOPK, NTL)
    if n_it > 0:
        # every CTA of each peer owner finished the previous launch: done word (owner rank+1+i//BLOCKS, CTA
        # i % BLOCKS) >= colour, for i = tid, tid + 64 W, .. < 3 BLOCKS (words past the end re-poll word tid)
        addrs = ()
        for q in gl.static_range(NQ):
            i = (q * W * 64 + tid) % NDONE
            o = (rank + 1 + i // BLOCKS) % TP
            addrs = addrs + (own + DN0 + o * BLOCKS + i % BLOCKS,)
        z = gl.zeros_like(tid)
        res = gl.inline_asm_elementwise(poll_asm(NQ), poll_cons(NQ),
                                        (n_it,) + addrs + (own + z, colour + z, wait_ticks + z.to(gl.int64)),
                                        (gl.int32,) * (NQ + 2) + (gl.int64, gl.int64, gl.int64), False, 1)
        fail_record(own, pr1, pr2, pr3, rank, colour, SITE_DONE, ((tid % 64) == 0) & (res[0] == 1))
        gl.barrier()
    total = n_it + LAG
    if UNROLL == 2:
        # iteration pairs: the prefetch alternates between two register sets, so the loop-carried vectors need no
        # copy at the back edge (a copy would wait for them, and in order for the peer stores before them)
        for t in range(0, (total + 1) // 2):
            ii = t * 2
            cur = iteration(inter, rev, sw, out, own, pr1, pr2, pr3, cur, ii, n_it, bid, m, rows, max_sorted, rank,
                            want, colour, wait_ticks, D, W, TOPK, BLOCKS, RMAX, F0, I0, LAG, NTL, FENCE, DBG)
            cur = iteration(inter, rev, sw, out, own, pr1, pr2, pr3, cur, ii + 1, n_it, bid, m, rows, max_sorted,
                            rank, want, colour, wait_ticks, D, W, TOPK, BLOCKS, RMAX, F0, I0, LAG, NTL, FENCE, DBG)
    else:
        for ii in range(0, total):
            cur = iteration(inter, rev, sw, out, own, pr1, pr2, pr3, cur, ii, n_it, bid, m, rows, max_sorted, rank,
                            want, colour, wait_ticks, D, W, TOPK, BLOCKS, RMAX, F0, I0, LAG, NTL, FENCE, DBG)
    # done: every inbox read of this CTA returned; tell every peer, bump the colour
    drain()
    gl.barrier()
    dst = gl.where(tid == 0, pr1, gl.where(tid == 1, pr2, pr3))
    store32_sys(dst + DN0 + rank * BLOCKS + bid, want + gl.zeros_like(tid), tid < TP - 1)
    gl.store(colors + bid + gl.zeros_like(tid), want + gl.zeros_like(tid), mask=tid == 0)
