# Does the graft keep thinking after a surprise? (Nov 14 22:50, the user: graft2 "will think if there's a surprise")
# The original's agentic prompts (delete/ag5-prompts.jsonl) and the head's exact input at each prompt's first thinking
# token (delete/ag5/xn.bin, first row of each capture): P(</think>) right away = P(no thinking at all on that step),
# exactly, for the unedited model, graft2, v5 and v7; also P(closed within the next rows of the original's thinking).
import os, sys, json, numpy as np
__file__ = os.path.abspath(__file__); sys.argv = sys.argv[:1]
exec(open(os.path.join(os.path.dirname(__file__), 'graft_refit.py')).read().split("C = load('hx'")[0])
P_ = [json.loads(l) for l in open(D / 'delete/ag5-prompts.jsonl')]; P_ = [p for p in P_ if p.get('thinking') is not False]
meta = json.load(open(D / 'delete/ag5/meta.json')); bb = open(D / 'delete/ag5/xn.bin', 'rb').read()
z = np.load(D / 'delete/ag5/head_stats.npz') if (D / 'delete/ag5/head_stats.npz').exists() else None
rows, first_idx, pid = [], [], []
for m in meta:
    o, k = m['off0'], 0
    while o < m['off1']:
        n = int(np.frombuffer(bb, np.int32, 1, o)[0]); x = np.frombuffer(bb, np.float32, n * DIM, o + 4).reshape(n, DIM)
        if k == 0: first_idx.append(len(rows))
        rows += list(x); pid += [m['i']] * n; o += 4 + n * DIM * 4; k += n
X = np.array(rows, np.float32); pid = np.array(pid)
HF = np.load(D / 'delete/head-f16.npy', mmap_mode='r'); V = HF.shape[0]
mx = np.full(len(X), -np.inf); se = np.zeros(len(X)); lc = np.zeros(len(X))
for v0 in range(0, V, 8192):
    w = np.asarray(HF[v0:v0 + 8192], np.float32); lg = X @ w.T; m2 = np.maximum(mx, lg.max(1))
    se = se * np.exp(mx - m2) + np.exp((lg - m2[:, None]).astype(np.float64)).sum(1); mx = m2
    if v0 <= CLOSE < v0 + len(w): lc = lg[:, CLOSE - v0].astype(np.float64)
lse = mx + np.log(se); o_ = lc - (lse + np.log1p(-np.exp(np.minimum(lc - lse, -1e-9))))
EDITS = {'unedited': np.zeros(DIM), 'graft2': 14 * np.load(D / 'delete/graft2-g.npy'), 'v5': 14 * np.load(D / 'delete/graft5w8-g.npy'), 'v7': np.load(D / 'delete/graft7-dw.npy')}
KIND = {1: 'routine: script printed the answer', 2: 'routine: answer confirmed', 4: 'SURPRISE: tests fail (..EFEF., traceback)', 5: 'new info: test file contents read',
        6: 'routine: file updated', 7: 'routine: Ran 7 tests OK', 9: 'routine: empty dir listing', 14: 'routine: file created', 15: 'routine: wc 487 lines',
        16: 'routine: tail of file', 17: 'routine: (no output)', 19: 'routine: node version', 20: 'routine: no mismatches', 21: 'SURPRISE: node ENOENT error',
        22: 'routine: syntax OK', 24: 'SURPRISE: firefox failed to launch', 0: 'task start', 3: 'task start', 8: 'task start'}
print(f'{"prompt":6s} {"what the last tool result was":44s} ' + ' '.join(f'{k:>9s}' for k in EDITS) + '   (P no thinking at all = P(</think>) at the first token)')
for j, m in enumerate(meta):
    i = m['i']; r = first_idx[j]; xs = X[r].astype(np.float64)
    ps = {k: 1 / (1 + np.exp(-(o_[r] + xs @ e))) for k, e in EDITS.items()}
    print(f'{i:6d} {KIND.get(i, "routine: other"):44s} ' + ' '.join(f'{ps[k]:9.3f}' for k in EDITS) + f'   original thought {"<4" if 128822 in m["toks"][:4] else (m["toks"].index(128822) if 128822 in m["toks"] else ">=48")} tok')
