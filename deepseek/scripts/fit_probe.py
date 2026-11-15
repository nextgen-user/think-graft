# Refit of the head graft for the path it runs in (Nov 14 02:30).  The first probe was fitted on chunk-path head inputs;
# in one-token decode (production flags) P(</think>) at the same paragraph ends came out ~1-3 nats lower (delete/diag.py).
# Here: head inputs from BOTH paths (delete/hx: chunked, full traces; delete/hx1: one-token decode, traces cut ~2.5k
# tokens after the answer), the shrinkage-LDA boundary probe fitted on the union, beta calibrated by the exact simulation
# on the DECODE rows (cross-trace too), then the Q5_K row and a new overlay.  usage: graft_refit.py [beta]
import sys, os, json, math, pathlib, ctypes, struct, numpy as np
D = pathlib.Path(__file__).resolve().parent.parent
R = pathlib.Path('~/mycodesandexp/Strata-DeepSeek')
sys.path.insert(0, str(R / 'third_party/llama.cpp/gguf-py')); sys.path.insert(0, str(R / 'tools/dsv41')); sys.path.insert(0, str(R / 'tools'))
from gguf import quants, GGMLQuantizationType as Q
from smoke import tokenizer
PK = pathlib.Path('~/models/DeepSeek-V4.1-Flash-IQ2_XXS/strata-pack'); TOK = tokenizer(PK)
DIM, CLOSE = 5120, 128822
files = {}; mdir = None; tens = {}
for l in list(open(PK / 'ds4.txt')) + list(open(PK / 'tensors.txt')):
    f = l.split()
    if not f: continue
    if f[0] == 'model_dir': mdir = f[1]
    if f[0] == 'file': files[int(f[1])] = f[2]
    if f[0] == 't' and f[1] in ('head.weight', 'norm.weight'): tens[f[1]] = (int(f[3]), int(f[4]), int(f[5]))
def rd(name):
    fi, off, nb = tens[name]
    with open(pathlib.Path(mdir) / files[fi], 'rb') as fh: fh.seek(off); return fh.read(nb)
norm_w = np.frombuffer(rd('norm.weight'), np.float32).copy(); HB = rd('head.weight'); RB = DIM // 256 * 176
info = json.load(open(D / 'delete/cand/neurons.json'))['info']
def load(hxdir, idsf):
    segs = json.load(open(D / 'data' / idsf))['segments']; H = D / 'delete' / hxdir
    if (H / 'xn.f16').exists():   # (kf52 DS4_DUMP_XN: the head's exact normed input, one row per scored position in order)
        xn = np.memmap(H / 'xn.f16', np.float16, 'r').reshape(-1, DIM); tot = xn.shape[0]
        lens = [len(s['ids']) - 1 for s in segs] if tot <= sum(len(s['ids']) - 1 for s in segs) else [len(s['ids']) for s in segs]
        seg = np.concatenate([np.full(L, i) for i, L in enumerate(lens)]); pos = np.concatenate([np.arange(L) for L in lens])
        n = min(len(seg), tot); seg, pos = seg[:n], pos[:n]; X = xn[:n].astype(np.float32)
        print(f'{hxdir}: {tot} head rows for segment lengths {lens}', flush=True)
    else:
        hx = np.memmap(H / 'hx.f16', np.float16, 'r').reshape(-1, DIM)
        rows = [tuple(int(x) for x in l.split()) for l in open(H / 'rows.tsv') if l.split() and int(l.split()[0]) == 39]
        seg, pos, k, last = [], [], -1, None
        for (il, p0, T) in rows:
            if last is None or p0 < last: k += 1
            last = p0; seg += [k] * T; pos += list(range(p0, p0 + T))
        n = min(len(seg), hx.shape[0]); seg = np.array(seg[:n]); pos = np.array(pos[:n])
        X = hx[:n].astype(np.float64); X = (X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-20) * norm_w).astype(np.float32)
    lab = np.full(n, -1); para = np.zeros(n, bool); names = [s['name'] for s in segs]
    for i, s in enumerate(segs):
        I = info[s['name']]; m = seg == i; p = pos[m]
        lab[m] = np.where(p < I['prompt'], -1, np.where(p < I['answer'], 0, np.where(p < I['close'] - 1, 1, 2)))
        para[m] = np.array(['\n\n' in TOK.decode([s['ids'][q]]) for q in p])
    cache = H / 'base_head.npz'
    if cache.exists(): z = np.load(cache); lse, lc = z['lse'], z['lc']
    else:
        V = len(HB) // RB; mx = np.full(n, -np.inf); se = np.zeros(n); lc = np.zeros(n)
        for v0 in range(0, V, 4096):
            kk = min(4096, V - v0); w = quants.dequantize(np.frombuffer(HB[v0 * RB:(v0 + kk) * RB], np.uint8).reshape(kk, RB), Q.Q5_K).astype(np.float32)
            for r0 in range(0, n, 4096):
                lg = (X[r0:r0 + 4096] @ w.T).astype(np.float64); sl = slice(r0, r0 + 4096)
                m2 = np.maximum(mx[sl], lg.max(1)); se[sl] = se[sl] * np.exp(mx[sl] - m2) + np.exp(lg - m2[:, None]).sum(1); mx[sl] = m2
                if v0 <= CLOSE < v0 + kk: lc[sl] = lg[:, CLOSE - v0]
        lse = mx + np.log(se); np.savez(cache, lse=lse, lc=lc)
    print(f'{hxdir}: {n} rows, traces {names}, verify paragraph ends {int(((lab == 1) & para).sum())}', flush=True)
    return dict(X=X, lab=lab, para=para, seg=seg, pos=pos, names=names, lse=lse, lc=lc)
DEC = os.environ.get('GRAFT_DEC', 'hx1'); TAGV = {'hx1': '', 'hx2': 'defer-'}[DEC]   # (hx2: one-token decode under the kf45 deferral flags)
C = load('hx', 'screen.ids.json'); W = load(DEC, 'screen-trunc.ids.json')
# alignment check: the decode rows' own log P(</think>) against the engine's score TSV (where </think> is in its top 64)
tsv = [l.rstrip('\n').split('\t') for l in open(D / ('runs/hx-w1.tsv' if DEC == 'hx1' else 'runs/hx-w1-defer.tsv'))]
lp_t = []
for r in tsv[:len(W['lc'])]:
    top = dict((int(a), float(b)) for a, b in (q.split(':') for q in r[8:72])); lp_t.append(top.get(CLOSE, np.nan))
lp_t = np.array(lp_t); lp_x = (W['lc'] - W['lse'])[:len(lp_t)]; ok = ~np.isnan(lp_t)
print(f'alignment: {int(ok.sum())} rows with </think> in the top 64; |log P from rows - log P in TSV| median {np.median(np.abs(lp_x[ok] - lp_t[ok])):.4f}, max {np.abs(lp_x[ok] - lp_t[ok]).max():.4f}', flush=True)
def lda(sets, train_names, shrink=0.2):
    P, N = [], []
    for S in sets:
        tr = np.isin(np.array(S['names'])[S['seg']], train_names) & ((S['lab'] == 0) | (S['lab'] == 1))
        pm = tr & (S['lab'] == 1) & S['para']; P.append(S['X'][pm]); N.append(S['X'][tr & ~pm][::3])
    P = np.concatenate(P).astype(np.float64); N = np.concatenate(N).astype(np.float64)
    mp, mn = P.mean(0), N.mean(0); Xc = N - mn; Sg = Xc.T @ Xc / len(Xc); Sg = (1 - shrink) * Sg + shrink * np.trace(Sg) / DIM * np.eye(DIM)
    w = np.linalg.solve(Sg, mp - mn); a, b = float(np.median(N @ w)), float(np.median(P @ w)); return w / (b - a)
def sim(S, g, beta, label):
    pr = S['X'].astype(np.float64) @ g; l2 = S['lc'] + beta * pr
    p = np.exp(l2 - np.logaddexp(S['lse'] + np.log1p(-np.exp(np.minimum(S['lc'] - S['lse'], -1e-9))), l2)); out = []
    for i, nm in enumerate(S['names']):
        I = info[nm]; m = S['seg'] == i; sv = m & (S['lab'] == 0); vf = m & (S['lab'] == 1)
        ps = 1 - np.exp(np.log1p(-np.minimum(p[sv], 1 - 1e-12)).sum()); pv = p[vf]; posv = S['pos'][vf]
        surv = np.exp(np.concatenate([[0], np.cumsum(np.log1p(-np.minimum(pv, 1 - 1e-12)))[:-1]])); pst = surv * pv
        end = posv.max() + 1; e_stop = (pst * posv).sum() + (1 - pst.sum()) * end
        out.append(f"{nm[:5]}: P(close while solving) {ps:.3f}  P(closed by the cut at {end}) {pst.sum():.2f}  expected close {e_stop:.0f} (answer {I['answer']}, base close {I['close']})  "
                   f"at paragraph ends {float((pst * S['para'][vf]).sum() / max(pst.sum(), 1e-9)):.2f}")
    print(f'  {label} beta {beta}: ' + '\n' + ' ' * 6 + ('\n' + ' ' * 6).join(out), flush=True)
both = ['digit_sum_double', 'coin_pattern_race']
for tr in (['coin_pattern_race'], ['digit_sum_double']):
    g = lda([C, W], tr)
    for beta in (14, 18): sim(W, g, beta, f'decode rows, fit {tr[0][:5]} (both paths)')
g = lda([C, W], both)
for beta in (12, 14, 16, 18, 20): sim(W, g, beta, 'decode rows, fit both (both paths)')
for beta in (14, 18): sim(C, g, beta, 'chunk rows, fit both (both paths)')
beta = float(sys.argv[1]) if len(sys.argv) > 1 else None
if beta:
    lib = ctypes.CDLL('~/omnivoice.cpp/build/libggml-base.so.0.10.1')
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    orig = quants.dequantize(np.frombuffer(HB[CLOSE * RB:(CLOSE + 1) * RB], np.uint8).reshape(1, -1), Q.Q5_K)[0].astype(np.float64)
    new = np.ascontiguousarray(orig + beta * g, np.float32); dst = np.zeros(RB, np.uint8)
    assert lib.ggml_quantize_chunk(13, new.ctypes.data, dst.ctypes.data, 0, 1, DIM, None) == RB
    eff = (quants.dequantize(dst.reshape(1, -1), Q.Q5_K)[0].astype(np.float64) - orig) / beta
    sim(W, eff, beta, 'decode rows, Q5_K row as written')
    dst.tofile(D / f'delete/head-row-graft2-{TAGV}{int(beta)}.q5k'); np.save(D / f'delete/graft2-{TAGV}g.npy', g); print('row written', beta)
