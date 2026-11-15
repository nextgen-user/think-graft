# The one-row </think> edit for Qwen3.5-4B (Nov 15 2026); same method as the DeepSeek fit (deepseek/scripts/fit_probe.py):
# a shrinkage-LDA probe (shrinkage 0.2) separating "paragraph ends after the answer has appeared in the thinking" (the
# model is only re-checking) from every other thinking position, scaled so the medians sit 1 apart, added to the
# </think> row as beta * g; beta from an exact offline simulation (only one logit changes, so the edited probability is
# exact at every recorded position).  Held-out report: fit on half the traces, simulate on the other half.
# usage: python3 fit.py CAPDIR traces.jsonl [BETA OUT.gguf]   (with BETA: refit on all traces and write the grafted GGUF:
#        a copy of the model + a separate output.weight = token_embd (tied head) with the </think> row edited, Q8_0)
import sys, os, json, re, numpy as np
sys.path.insert(0, os.path.join(os.environ.get('LLAMA_CPP', 'llama.cpp'), 'gguf-py')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf
from gguf import quants, GGMLQuantizationType as Q
from grade import correct, answer_of, norm
SRC = os.environ.get('QWEN_GGUF', 'models/Qwen3.5-4B-Q8_0.gguf')
cap, tr = sys.argv[1], sys.argv[2]
R = gguf.GGUFReader(SRC)
TOK = R.fields['tokenizer.ggml.tokens'].contents()
CLOSE = TOK.index('</think>'); OPEN = TOK.index('<think>')
te = next(t for t in R.tensors if t.name == 'token_embd.weight'); assert te.tensor_type == Q.Q8_0
EMB = quants.dequantize(te.data, Q.Q8_0).astype(np.float32)       # (vocab, 2560): the tied head
V, DIM = EMB.shape; print(f'vocab {V}, dim {DIM}, </think> = {CLOSE}', flush=True)
bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1)); cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
BYTE = {chr(c): b for b, c in zip(bs, cs)}
def piece(t):
    s = TOK[t]
    try: return bytes(BYTE[c] for c in s).decode('utf-8', 'replace')
    except KeyError: return s                                        # (special tokens)
rows = {json.loads(l)['id']: json.loads(l) for l in open(tr)}
IDX = json.load(open(os.path.join(cap, 'index.json')))
def head_stats(X):
    lse = np.zeros(len(X)); lc = X @ EMB[CLOSE]
    for r0 in range(0, len(X), 2048):
        lg = X[r0:r0 + 2048] @ EMB.T; m = lg.max(1); lse[r0:r0 + 2048] = m + np.log(np.exp(lg - m[:, None]).sum(1))
    return lse, lc
T = []
for e in IDX:
    r = rows[e['id']]; g = r['gen_ids']
    closed = CLOSE in g
    if closed and not correct(r['text'], r['expected']): continue      # (closed traces must end right)
    ci = g.index(CLOSE) if closed else len(g); pieces = [piece(t) for t in g]; off = np.cumsum([0] + [len(p) for p in pieces]); think = ''.join(pieces[:ci])
    # unclosed traces (hit the token limit) are kept when the right answer was already stated in the thinking: everything
    # after that point is the over-thinking the graft is for
    cands = [x for x in {answer_of(r['text']), r['expected']} if x] if closed else [r['expected']]
    pos = []
    for c in cands:
        if len(norm(c) or '') <= 3:   # (short numbers also show up as intermediate values: require it to be stated)
            m = re.search(r'(=|is|answer|get|gives|total|equals)\s*\$?\\?\(?\s*' + re.escape(c) + r'(?![\d.])', think)
            if m: pos.append(m.end())
        else:
            k = think.find(c)
            if k >= 0: pos.append(k + len(c))
    if not pos: continue
    ans_tok = int(np.searchsorted(off, min(pos)))                      # first token after the answer's first statement
    X = np.load(os.path.join(cap, e['file'])).astype(np.float32); npf = e['n_prompt']
    lse, lc = head_stats(X)
    # row p predicts token p+1; generated token j sits at position npf+j
    lab = np.full(len(X), -1); para = np.zeros(len(X), bool)
    for j in range(len(g)):
        p = npf + j
        if p >= len(X): break
        lab[p] = 0 if j < ans_tok else (1 if j < ci else 2); para[p] = '\n\n' in pieces[j]
    lab[npf - 1] = 0                                                   # (the prompt's last row decides the first thinking token)
    T.append(dict(id=e['id'], X=X, lse=lse, lc=lc, lab=lab, para=para, ans=ans_tok, close=ci, closed=closed))
print(f'{len(T)} usable traces of {len(IDX)} ({sum(t["closed"] for t in T)} closed; answer located in all)', flush=True)
def lda(TT, shrink=0.2):
    P = np.concatenate([t['X'][(t['lab'] == 1) & t['para']] for t in TT]).astype(np.float64)
    N = np.concatenate([t['X'][((t['lab'] == 0) | (t['lab'] == 1)) & ~((t['lab'] == 1) & t['para'])][::3] for t in TT]).astype(np.float64)
    mp, mn = P.mean(0), N.mean(0); Xc = N - mn; S = Xc.T @ Xc / len(Xc); S = (1 - shrink) * S + shrink * np.trace(S) / DIM * np.eye(DIM)
    w = np.linalg.solve(S, mp - mn); a, b = float(np.median(N @ w)), float(np.median(P @ w)); return w / (b - a), len(P), len(N)
def pnew(t, g, beta):
    l2 = t['lc'] + beta * (t['X'].astype(np.float64) @ g)
    return np.exp(l2 - np.logaddexp(t['lse'] + np.log1p(-np.exp(np.minimum(t['lc'] - t['lse'], -1e-9))), l2))
def report(TT, g, beta):
    ps, kept, base, stray = [], 0, 0, 0
    for t in TT:
        p = pnew(t, g, beta); sv = t['lab'] == 0; vf = t['lab'] == 1; an = t['lab'] == 2
        ps.append(1 - np.exp(np.log1p(-np.minimum(p[sv], 1 - 1e-12)).sum()))
        pv = p[vf]; surv = np.exp(np.concatenate([[0], np.cumsum(np.log1p(-np.minimum(pv, 1 - 1e-12)))[:-1]])); pst = surv * pv
        k = np.arange(len(pv)); kept += (pst * k).sum() + (1 - pst.sum()) * len(pv); base += len(pv); stray += p[an].sum()
    return max(ps), float(np.mean(ps)), kept / max(base, 1), base, stray / max(1, sum(1 for t in TT if (t['lab'] == 2).any()))
if __name__ == '__main__':
    if len(sys.argv) <= 4:                                                # (report only when not writing)
        half = [t for i, t in enumerate(T) if i % 2 == 0]; other = [t for i, t in enumerate(T) if i % 2 == 1]
        for fitset, test, name in ((half, other, 'fit even / test odd'), (other, half, 'fit odd / test even')):
            g, nP, nN = lda(fitset); print(f'== {name}: {nP} positives, {nN} negatives', flush=True)
            for beta in (0, 4, 6, 8, 10, 12, 14, 16, 20):
                mx, mean, kept, base, stray = report(test, g, beta)
                print(f'  beta {beta:2d}: held-out P(close while solving) max {mx:.3f} mean {mean:.3f} | re-checking kept {kept:.0%} of {base} tokens | stray </think> per answer {stray:.2f}', flush=True)
    if len(sys.argv) > 4:
        beta, dst = float(sys.argv[3]), sys.argv[4]
        g, nP, nN = lda(T); np.save(os.path.splitext(dst)[0] + '-probe.npy', g)
        raw = np.array(te.data, copy=True)                               # (vocab, 2720) Q8_0 blocks
        new_row = quants.quantize((EMB[CLOSE] + beta * g).astype(np.float32).reshape(1, -1), Q.Q8_0)
        raw[CLOSE] = new_row.reshape(-1)
        eff = quants.dequantize(raw[CLOSE:CLOSE + 1], Q.Q8_0)[0] - EMB[CLOSE]
        print(f'Q8_0 row: |written edit - wanted| / |wanted| = {np.linalg.norm(eff - beta * g) / np.linalg.norm(beta * g):.4f}', flush=True)
        W = gguf.GGUFWriter(dst, arch=R.fields['general.architecture'].contents(), endianess=R.endianess)
        for f in R.fields.values():
            if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith('GGUF.'): continue
            vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            W.add_key_value(f.name, f.contents(), vt, sub_type=st)
        W.add_key_value('general.graft', f'one-row edit: </think> row += {beta} * LDA probe (re-checking after the answer); separate output head', gguf.GGUFValueType.STRING)
        for t in R.tensors: W.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        W.add_tensor_info('output.weight', raw.shape, raw.dtype, raw.nbytes, Q.Q8_0)
        W.write_header_to_file(); W.write_kv_data_to_file(); W.write_ti_data_to_file()
        for t in R.tensors: W.write_tensor_data(t.data, tensor_endianess=R.endianess)
        W.write_tensor_data(raw, tensor_endianess=R.endianess); W.close()
        print('wrote', dst, flush=True)
