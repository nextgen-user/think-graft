# The beta dial, computed exactly offline (Nov 14): the graft changes one logit, so its effect at every recorded position
# is closed-form.  For the v5 probe g at beta 0..24: (a) maths traces on the production (kf45 deferral) decode path: chance
# of closing while still solving, expected close, checking tokens kept, share of closes at paragraph ends; (b) the 22
# agentic prompts' first 48 tokens: chance the thinking closes there; (c) graft2's answer phases: expected stray </think>.
# usage: beta_sweep.py -> table + delete/beta-sweep.json
import os, sys, json, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graft_refit.py')).read().split("C = load('hx'")[0])
W = load('hx2', 'screen-trunc.ids.json'); g = np.load(D / 'delete/graft5w8-defer-g.npy')
def pc(X, lse, lc, beta):
    l2 = lc + beta * (X.astype(np.float64) @ g); return np.exp(l2 - np.logaddexp(lse + np.log1p(-np.exp(np.minimum(lc - lse, -1e-9))), l2))
# agentic rows (as graft_v5: rows before the original's own close; prompts it closes within 4 tokens left out)
meta = json.load(open(D / 'delete/ag5/meta.json')); b = open(D / 'delete/ag5/xn.bin', 'rb').read(); XG, pid = [], []
for m in meta:
    c = m['toks'].index(CLOSE) if CLOSE in m['toks'] else len(m['toks'])
    if c < 4: continue
    o, cum = m['off0'], 0
    while o < m['off1'] and cum < c - 2:
        n = int(np.frombuffer(b, np.int32, 1, o)[0]); XG.append(np.frombuffer(b, np.float32, n * DIM, o + 4).reshape(n, DIM)); pid += [m['i']] * n; o += 4 + n * DIM * 4; cum += n
XG = np.concatenate(XG); pid = np.array(pid); V = len(HB) // RB; mx = np.full(len(XG), -np.inf); se = np.zeros(len(XG)); lcG = None
for v0 in range(0, V, 4096):
    kk = min(4096, V - v0); w = quants.dequantize(np.frombuffer(HB[v0 * RB:(v0 + kk) * RB], np.uint8).reshape(kk, RB), Q.Q5_K).astype(np.float32)
    lg = (XG @ w.T).astype(np.float64); m2 = np.maximum(mx, lg.max(1)); se = se * np.exp(mx - m2) + np.exp(lg - m2[:, None]).sum(1); mx = m2
    if v0 <= CLOSE < v0 + kk: lcG = lg[:, CLOSE - v0]
lseG = mx + np.log(se)
# answer phases of graft2's outputs (as graft_v4)
segsA = json.load(open(D / 'data/answers.ids.json'))['segments']; xa = np.memmap(D / 'delete/hx3/xn.f16', np.float16, 'r').reshape(-1, DIM)
lens = [len(s['ids']) - 1 for s in segsA] if xa.shape[0] <= sum(len(s['ids']) - 1 for s in segsA) else [len(s['ids']) for s in segsA]
off = np.cumsum([0] + lens); ar = np.concatenate([np.arange(off[i] + s['ids'].index(CLOSE, s['prompt']), off[i] + lens[i]) for i, s in enumerate(segsA)])
XA = xa[ar].astype(np.float32); z = np.load(D / 'delete/hx3/base_head_ans.npz'); lseA, lcA = z['lse'], z['lc']
res = []
for beta in (0, 4, 8, 10, 12, 14, 16, 18, 20, 24):
    p = pc(W['X'], W['lse'], W['lc'], beta); row = dict(beta=beta)
    for i, nm in enumerate(W['names']):
        I = info[nm]; m = W['seg'] == i; sv = m & (W['lab'] == 0); vf = m & (W['lab'] == 1)
        ps = 1 - np.exp(np.log1p(-np.minimum(p[sv], 1 - 1e-12)).sum()); pv = p[vf]; posv = W['pos'][vf]
        surv = np.exp(np.concatenate([[0], np.cumsum(np.log1p(-np.minimum(pv, 1 - 1e-12)))[:-1]])); pst = surv * pv
        end = posv.max() + 1; e = (pst * posv).sum() + (1 - pst.sum()) * end
        # checking kept: expected close - answer position; a trace not closed by the recorded cut counts the base's own close
        e_full = (pst * posv).sum() + (1 - pst.sum()) * I['close']
        row[nm[:5]] = dict(p_close_solving=round(float(ps), 4), check_kept=round(float(e_full - I['answer'])), check_base=I['close'] - I['answer'],
                           closed_by_cut=round(float(pst.sum()), 3), at_para=round(float((pst * W['para'][vf]).sum() / max(pst.sum(), 1e-9)), 3))
    pg = pc(XG, lseG, lcG, beta); row['agentic_p_close_48'] = round(float(np.mean([1 - np.prod(1 - pg[pid == i]) for i in sorted(set(pid))])), 4)
    pa = pc(XA, lseA, lcA, beta); row['strays_per_answer'] = round(float(pa.sum() / len(segsA)), 3)
    res.append(row)
    print(f"beta {beta:2d} | digit: solving-close {row['digit']['p_close_solving']:.3f}, checking kept {row['digit']['check_kept']:5d} of {row['digit']['check_base']} | "
          f"coin_: solving-close {row['coin_']['p_close_solving']:.3f}, checking kept {row['coin_']['check_kept']:5d} of {row['coin_']['check_base']} | "
          f"agentic close in 48 tok {row['agentic_p_close_48']:.3f} | strays/answer {row['strays_per_answer']:.2f}", flush=True)
json.dump(res, open(D / 'delete/beta-sweep.json', 'w'), indent=1)
