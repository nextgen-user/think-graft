# Graft v7s (Nov 14 22:10): the survival likelihood - the fit done right.  P(</think>) at a row is exactly
# sigmoid(o + x.dw) (o = the base model's log-odds of </think>).  A run stops at its FIRST close, so per episode (a maths
# trace; an agentic step) the likelihood of the wanted behaviour is
#     prod_{rows before the window} (1 - p)  x  (1 - prod_{window rows} (1 - p))
# window = where stopping is right: maths - from the row where the answer is settled through the 3rd paragraph end after
# it (or the trace's own close if sooner); agentic - from the first paragraph end before/inside drafting through the
# next one, or the step's own natural close.  Rows after the window are left out (unknown).  Every other generated row
# (answers, tool-call content, step starts, steps without a window) is a pure "no close": -log(1-p) = softplus(z), weight
# 1 per row (subsampled groups count the rows they stand for).  v7 (one label per paragraph end) counted ~300 rewards
# and paid early-close risk on every solving row for them.  L2 on dw (standardised features), L-BFGS.
# Held out as v7.  usage: graft_v7s.py [CONFIG] ; with CONFIG writes delete/head-row-graft7.q5k
import os, sys, json, time, numpy as np
__file__ = os.path.abspath(__file__); _A7S = sys.argv[1:]; sys.argv = sys.argv[:1]
exec(open(os.path.join(os.path.dirname(__file__), 'graft_v7.py')).read().rsplit("if __name__ == '__main__':", 1)[0])
NPARA = 3
# rows in the last 20 tokens before the regex-marked answer: the value is usually already computed there ("-> out85."
# then "So result 85") - neither "no close" nor counted as an early close in the reports
for _i, _s in enumerate(segs6):
    if _s['kind'] == 'maths':
        _m = (seg6 == _i) & (lab6 == 'solve') & (pos6 >= _s['solve_end'] - 20); lab6[_m] = 'near'
def episodes():
    """window row lists (indices into X6 / old C, W arrays) and pure-negative masks for TRAINING rows"""
    t = ~hold6; win6 = []; neg6 = np.zeros(len(R6), bool); drop6 = np.zeros(len(R6), bool)
    for i, s in enumerate(segs6):
        m = np.nonzero((seg6 == i) & t)[0]
        if not len(m): continue
        if s['kind'] == 'maths':
            ps = pos6[m]; L = lab6[m]; se = s['solve_end']
            para_after = [k for k in m[ps >= se] if lab6[k] == 'pos']
            if not para_after: neg6[m[(L == 'solve') | (L == 'ans')]] = True; drop6[m[(L == 'check') | (L == 'pos')]] = True; continue
            wend = pos6[para_after[min(NPARA, len(para_after)) - 1]]
            win6.append(m[(ps >= se) & (ps <= wend) & np.isin(L, ['check', 'pos'])])
            neg6[m[(L == 'solve') | (L == 'ans')]] = True; drop6[m[(ps > wend) & np.isin(L, ['check', 'pos'])]] = True
        else:
            for k, st in enumerate(s['steps']):
                mm = m[(pos6[m] >= st['a']) & (pos6[m] < st['b'])]
                if not len(mm): continue
                pp = [r for r in mm if lab6[r] == 'apos']
                if pp:
                    p0 = pos6[pp[0]]; p1 = pos6[pp[min(1, len(pp) - 1)]]
                    win6.append(mm[(pos6[mm] >= p0) & (pos6[mm] <= p1) & (lab6[mm] != 'acap')])
                    neg6[mm[(pos6[mm] < p0) & (lab6[mm] != 'acap')]] = True; drop6[mm[pos6[mm] > p1]] = True
                else: neg6[mm[lab6[mm] != 'acap']] = True
    return win6, neg6
def old_eps(S):
    """the calibration traces: lab 0 = solving (no close), window = first NPARA paragraph ends of lab 1 (verifying)"""
    nm = np.array(S['names'])[S['seg']]; W_, N_ = [], np.zeros(len(S['lab']), bool)
    for i, n in enumerate(S['names']):
        if n in HOLD: continue
        m = np.nonzero(S['seg'] == i)[0]; L = S['lab'][m]; v = m[L == 1]; pe = v[S['para'][v]]
        N_[m[L == 0]] = True
        if len(pe): W_.append(v[v <= pe[min(NPARA, len(pe)) - 1]])
    return W_, N_
def fit_s(cfg, iters=400, verbose=False):
    win6, neg6 = episodes(); WC, NC = old_eps(C); WW, NW = old_eps(W)
    negs = [(X6[neg6 & (lab6 == 'solve')], o6[neg6 & (lab6 == 'solve')], 1.0),   # (every solving row: v7s-1 skipped 2 of 3 and closed between them) (X6[neg6 & (lab6 != 'solve') & (lab6 != 'astart')], o6[neg6 & (lab6 != 'solve') & (lab6 != 'astart')], 1.0),
            (X6[neg6 & (lab6 == 'astart')], o6[neg6 & (lab6 == 'astart')], cfg['wstart']), (XA[~oddA], oA[~oddA], cfg['wans']), (XG[~holdG], oG[~holdG], cfg['wstart']),
            (XC[~hold_c], oC_[~hold_c], cfg['wcall']), (C['X'][NC][::2], offs(C['lse'], C['lc'])[NC][::2], 2.0), (W['X'][NW][::2], offs(W['lse'], W['lc'])[NW][::2], 2.0)]
    wins = [(X6[w], o6[w]) for w in win6] + [(C['X'][w], offs(C['lse'], C['lc'])[w]) for w in WC] + [(W['X'][w], offs(W['lse'], W['lc'])[w]) for w in WW]
    Xs = np.concatenate([n[0] for n in negs] + [w[0] for w in wins]).astype(np.float32); o = np.concatenate([n[1] for n in negs] + [w[1] for w in wins])
    wn = np.concatenate([np.full(len(n[0]), n[2]) for n in negs]); nN = len(wn)
    ep = np.concatenate([np.full(len(w[0]), k) for k, w in enumerate(wins)]); nE = len(wins); we = cfg['pw']
    sd = (Xs.std(0, dtype=np.float64) + 1e-6).astype(np.float32); Xs /= sd; lam = cfg['lam']; norm = wn.sum() + we * nE
    def f(th):
        z = o + (Xs @ th.astype(np.float32)).astype(np.float64); sp = np.logaddexp(0, z); p = 1 / (1 + np.exp(-z))
        A = np.bincount(ep, weights=sp[nN:], minlength=nE)                      # -log S_e
        lw = -np.log(-np.expm1(-np.maximum(A, 1e-300)))                          # -log(1 - S_e)
        loss = (float((wn * sp[:nN]).sum()) + we * float(lw.sum())) / norm + 0.5 * lam * float(th @ th)
        r = np.empty_like(z); r[:nN] = wn * p[:nN]
        ratio = np.exp(-A) / (-np.expm1(-np.maximum(A, 1e-300)))                 # S/(1-S)
        r[nN:] = -we * p[nN:] * ratio[ep]
        return loss, (Xs.T @ (r / norm).astype(np.float32)).astype(np.float64) + lam * th
    th = np.zeros(DIM); fx, gx = f(th); S_, Y_ = [], []; t0 = time.time()
    for it in range(iters):
        q = gx.copy(); al = []
        for s_, y_ in reversed(list(zip(S_, Y_))): a_ = (s_ @ q) / (y_ @ s_); al.append(a_); q -= a_ * y_
        if S_: q *= (S_[-1] @ Y_[-1]) / (Y_[-1] @ Y_[-1])
        for (s_, y_), a_ in zip(zip(S_, Y_), reversed(al)): b_ = (y_ @ q) / (y_ @ s_); q += s_ * (a_ - b_)
        d = -q; step = 1.0
        while True:
            fn, gn = f(th + step * d)
            if fn <= fx + 1e-4 * step * (gx @ d) or step < 1e-10: break
            step *= 0.5
        s_ = step * d; y_ = gn - gx; th += s_
        if s_ @ y_ > 1e-15: S_.append(s_); Y_.append(y_)
        if len(S_) > 20: S_.pop(0); Y_.pop(0)
        if abs(fx - fn) < 1e-11 * max(1, abs(fx)): fx, gx = fn, gn; break
        fx, gx = fn, gn
    if verbose: print(f'   fit: {it + 1} iterations, loss {fx:.6f}, {time.time() - t0:.0f} s; {nN} no-close rows, {nE} episodes with a window ({len(ep)} window rows)', flush=True)
    return th / sd.astype(np.float64)
BASE = dict(pw=1.0, wans=1.0, wstart=1.0, wcall=1.0, lam=1e-6)
if __name__ == '__main__':
    if _A7S:
        cfg = dict(BASE, **{k: float(v) for k, v in (x.split('=') for x in _A7S[0].split(','))}); dw = fit_s(cfg, verbose=True); report(dw, f'== chosen {cfg}')
        np.save(D / 'delete/graft7-dw.npy', dw)
        lib = ctypes.CDLL('~/omnivoice.cpp/build/libggml-base.so.0.10.1'); lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        orig = quants.dequantize(np.frombuffer(HB[CLOSE * RB:(CLOSE + 1) * RB], np.uint8).reshape(1, -1), Q.Q5_K)[0].astype(np.float64)
        new = np.ascontiguousarray(orig + dw, np.float32); dst = np.zeros(RB, np.uint8)
        assert lib.ggml_quantize_chunk(13, new.ctypes.data, dst.ctypes.data, 0, 1, DIM, None) == RB
        eff = quants.dequantize(dst.reshape(1, -1), Q.Q5_K)[0].astype(np.float64) - orig
        print('Q5_K row: |written - wanted| / |wanted| =', float(np.linalg.norm(eff - dw) / np.linalg.norm(dw)))
        report(eff, '== Q5_K row as written'); dst.tofile(D / 'delete/head-row-graft7.q5k'); print('row written')
        # decode path (one token at a time, as generation runs): the calibration traces captured both ways
        for nm_, S_ in (('chunk', C), ('decode', W)):
            for i_, n_ in enumerate(S_['names']):
                m_ = S_['seg'] == i_; pv = P(S_['X'][m_], offs(S_['lse'], S_['lc'])[m_], eff); lb = S_['lab'][m_]; ps_ = S_['pos'][m_]; I_ = info[n_]
                sv = (lb == 0) & (ps_ < I_['answer'] - 20); psolve = 1 - np.exp(np.log1p(-np.minimum(pv[sv], 1 - 1e-12)).sum())
                vf = lb >= 1; pq = pv[vf]; pp = ps_[vf]; surv = np.exp(np.concatenate([[0], np.cumsum(np.log1p(-np.minimum(pq, 1 - 1e-12)))[:-1]])); pst = surv * pq
                ec = (pst * pp).sum() + (1 - pst.sum()) * I_['close']
                print(f'   {nm_:6s} {n_:18s}{" (held out)" if n_ in HOLD else ""}: P(close while solving) {psolve:.3f}; expected close {ec:.0f} (answer at {I_["answer"]}, own close {I_["close"]}); closed within the capture {pst.sum():.2f}', flush=True)
    else:
        for ch in (dict(), dict(lam=1e-5), dict(lam=1e-7), dict(pw=3.0), dict(wans=3.0, wstart=3.0, wcall=3.0), dict(pw=3.0, lam=1e-5)):
            cfg = dict(BASE, **ch); dw = fit_s(cfg, verbose=True); report(dw, f'== v7s {ch or "base config"}')
