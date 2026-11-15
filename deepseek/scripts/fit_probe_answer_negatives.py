# Graft v4 (Nov 14 07:30): graft2 without stray </think> tokens.  graft2's generations contain extra </think> tokens
# after the real one (round_table 57, coin 3, digit/code/dice 2; base runs exactly 1): once the model re-checks inside its
# visible answer, that text looks like verifying and the probe fires again.  Fix inside the same one-row edit: the head
# inputs over graft2's own outputs (delete/hx3, data/answers.ids.json, chunked) after their first </think> join the
# negatives.  Reports, per beta: the decode-path behaviour on the calibration traces (as graft_refit.py) and the expected
# number of stray </think> per answer (sum of P(</think>) over each answer phase) for graft2 and v4.
# usage: graft_v4.py [beta]   (writes delete/head-row-graft4-BETA.q5k)
import os, sys, json, numpy as np, ctypes
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "graft_refit.py")).read().split("both = [")[0])
segsA = json.load(open(D / 'data/answers.ids.json'))['segments']
xa = np.memmap(D / 'delete/hx3/xn.f16', np.float16, 'r').reshape(-1, DIM); tot = xa.shape[0]
lens = [len(s['ids']) - 1 for s in segsA] if tot <= sum(len(s['ids']) - 1 for s in segsA) else [len(s['ids']) for s in segsA]
off = np.cumsum([0] + lens); ans_rows = []; ans_seg = []
for i, s in enumerate(segsA):
    c0 = s['ids'].index(CLOSE, s['prompt'])   # the first </think>: rows from there on are the answer phase
    r = np.arange(off[i] + c0, off[i] + lens[i]); ans_rows.append(r); ans_seg += [i] * len(r)
ans_rows = np.concatenate(ans_rows); ans_seg = np.array(ans_seg); XA = xa[ans_rows].astype(np.float32)
print(f'answer-phase rows {len(XA)} from {len(segsA)} outputs ({[s["name"] for s in segsA]})', flush=True)
cache = D / 'delete/hx3/base_head_ans.npz'
if cache.exists(): z = np.load(cache); lseA, lcA = z['lse'], z['lc']
else:
    V = len(HB) // RB; n = len(XA); mx = np.full(n, -np.inf); se = np.zeros(n); lcA = np.zeros(n)
    for v0 in range(0, V, 4096):
        kk = min(4096, V - v0); w = quants.dequantize(np.frombuffer(HB[v0 * RB:(v0 + kk) * RB], np.uint8).reshape(kk, RB), Q.Q5_K).astype(np.float32)
        lg = (XA @ w.T).astype(np.float64); m2 = np.maximum(mx, lg.max(1)); se = se * np.exp(mx - m2) + np.exp(lg - m2[:, None]).sum(1); mx = m2
        if v0 <= CLOSE < v0 + kk: lcA = lg[:, CLOSE - v0]
    lseA = mx + np.log(se); np.savez(cache, lse=lseA, lc=lcA)
def fit(extra_neg, shrink=0.2):
    P, N = [], []
    for S in (C, W):
        tr = (S['lab'] == 0) | (S['lab'] == 1); pm = tr & (S['lab'] == 1) & S['para']
        P.append(S['X'][pm]); N.append(S['X'][tr & ~pm][::3])
    if extra_neg is not None: N.append(extra_neg)
    P = np.concatenate(P).astype(np.float64); N = np.concatenate(N).astype(np.float64)
    mp, mn = P.mean(0), N.mean(0); Xc = N - mn; Sg = Xc.T @ Xc / len(Xc); Sg = (1 - shrink) * Sg + shrink * np.trace(Sg) / DIM * np.eye(DIM)
    w = np.linalg.solve(Sg, mp - mn); a, b = float(np.median(N @ w)), float(np.median(P @ w)); return w / (b - a)
def strays(g, beta):
    l2 = lcA + beta * (XA.astype(np.float64) @ g); p = np.exp(l2 - np.logaddexp(lseA + np.log1p(-np.exp(np.minimum(lcA - lseA, -1e-9))), l2))
    return [round(float(p[ans_seg == i].sum()), 2) for i in range(len(segsA))], round(float(p.max()), 3)
g2 = np.load(D / 'delete/graft2-g.npy')
print('graft2 beta 14: expected stray </think> per answer', *strays(g2, 14))
print('no edit:        expected stray </think> per answer', *strays(np.zeros(DIM), 0))
# (half the answer rows fit, the other half test - interleaved by position - so the stray count is not in-sample)
fit_rows = (np.arange(len(XA)) % 2 == 0)
g4 = fit(XA[fit_rows].astype(np.float64))
XA_all, lcA_all, lseA_all, seg_all = XA, lcA, lseA, ans_seg
XA, lcA, lseA, ans_seg = XA_all[~fit_rows], lcA_all[~fit_rows], lseA_all[~fit_rows], seg_all[~fit_rows]
for beta in (14, 16, 18):
    print(f'v4 beta {beta}: expected stray </think> per answer (held-out half)', *strays(g4, beta))
    sim(W, g4, beta, 'v4 decode rows')
print('graft2 on the same held-out half:', *strays(g2, 14))
l2 = lcA + 14 * (XA.astype(np.float64) @ g4); p = np.exp(l2 - np.logaddexp(lseA + np.log1p(-np.exp(np.minimum(lcA - lseA, -1e-9))), l2))
for j in np.argsort(-p)[:4]:
    s = segsA[ans_seg[j]]; rr = ans_rows[~fit_rows][j] - off[ans_seg[j]]
    print(f'  v4 high P {p[j]:.3f} in {s["name"]} at {rr}: next {TOK.decode([s["ids"][rr + 1]])!r} ...{TOK.decode(s["ids"][max(0, rr - 30):rr + 1])[-120:]!r}')
XA, lcA, lseA, ans_seg = XA_all, lcA_all, lseA_all, seg_all
g4 = fit(XA.astype(np.float64))   # (the shipped fit uses all answer rows)
beta = float(sys.argv[1]) if len(sys.argv) > 1 else None
if beta:
    lib = ctypes.CDLL('~/omnivoice.cpp/build/libggml-base.so.0.10.1')
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    orig = quants.dequantize(np.frombuffer(HB[CLOSE * RB:(CLOSE + 1) * RB], np.uint8).reshape(1, -1), Q.Q5_K)[0].astype(np.float64)
    new = np.ascontiguousarray(orig + beta * g4, np.float32); dst = np.zeros(RB, np.uint8)
    assert lib.ggml_quantize_chunk(13, new.ctypes.data, dst.ctypes.data, 0, 1, DIM, None) == RB
    eff = (quants.dequantize(dst.reshape(1, -1), Q.Q5_K)[0].astype(np.float64) - orig) / beta
    sim(W, eff, beta, 'v4 Q5_K row as written, decode rows'); print('Q5_K row strays', *strays(eff, beta))
    dst.tofile(D / f'delete/head-row-graft4-{TAGV}{int(beta)}.q5k'); np.save(D / f'delete/graft4-{TAGV}g.npy', g4); print('row written', beta)
