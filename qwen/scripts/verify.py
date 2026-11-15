# Check the grafted GGUF (Nov 15 2026): at chosen positions of recorded traces, compare (1) the offline-predicted
# P(</think>) from the captured head input + probe, (2) the live original model, (3) the live grafted model; and check that
# all other tokens keep their relative probabilities.  usage: python3 verify.py stage PORT  (stage = orig | graft | report; run from qwen/)
import sys, json, os, urllib.request, numpy as np
CLOSE = 248069; S = os.getcwd()                        # run from the qwen/ folder (traces in data/, captures in cap-fit/)
rows = {json.loads(l)['id']: json.loads(l) for l in open(f'{S}/data/fit-traces.jsonl')}
IDX = {e['id']: e for e in json.load(open(f'{S}/cap-fit/index.json'))}
PICK = ['test/algebra/686.json', 'test/prealgebra/465.json', 'test/geometry/221.json']
stage = sys.argv[1]
if stage in ('orig', 'graft'):
    port = sys.argv[2]; pos = json.load(open(f'{S}/logs/verify-pos.json')); res = []
    for uid, p, _ in pos:
        r = rows[uid]; ids = (r['prompt_ids'] + r['gen_ids'])[:p + 1]
        req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', data=json.dumps({'prompt': ids, 'n_predict': 1, 'n_probs': 30, 'temperature': 0, 'cache_prompt': False}).encode(), headers={'Content-Type': 'application/json'})
        out = json.load(urllib.request.urlopen(req, timeout=600))['completion_probabilities'][0]['top_logprobs']
        res.append({str(o['id']): o['logprob'] for o in out})
    name = {'orig': 'original', 'graft': 'edited'}[stage]; json.dump(res, open(f'{S}/logs/verify-{name}.json', 'w')); print(stage, 'done', len(res))
else:
    sys.path.insert(0, os.path.join(os.environ.get('LLAMA_CPP', 'llama.cpp'), 'gguf-py')); import gguf; from gguf import quants
    if not os.path.exists(f'{S}/logs/verify-pos.json'):     # choose positions + offline prediction
        R = gguf.GGUFReader(os.environ.get('QWEN_GGUF', 'models/Qwen3.5-4B-Q8_0.gguf')); te = [t for t in R.tensors if t.name == 'token_embd.weight'][0]
        E = quants.dequantize(te.data, te.tensor_type).astype(np.float32); g = np.load(os.environ.get('PROBE', 'models/think-graft-probe.npy'))
        pos = []
        for uid in PICK:
            e = IDX[uid]; X = np.load(f"{S}/cap-fit/{e['file']}").astype(np.float32); npf = e['n_prompt']
            G = X @ g; c = np.where(np.array(rows[uid]['gen_ids']) == CLOSE)[0]; end = npf + (c[0] if len(c) else len(X) - npf - 1)
            cand = [int(npf + 50), int(npf + np.argmax(G[npf:end])), int(npf + (end - npf) // 2)]
            for p in cand:
                lg = X[p] @ E.T; lg2 = lg.copy(); lg2[CLOSE] += 14 * G[p]
                lp = lambda v: float(v[CLOSE] - v.max() - np.log(np.exp(v - v.max()).sum()))
                pos.append([uid, p, {'offline_orig': lp(lg), 'offline_graft': lp(lg2)}])
        json.dump(pos, open(f'{S}/logs/verify-pos.json', 'w')); print('positions chosen'); sys.exit()
    pos = json.load(open(f'{S}/logs/verify-pos.json')); O = json.load(open(f'{S}/logs/verify-original.json')); Gr = json.load(open(f'{S}/logs/verify-edited.json'))
    for (uid, p, off), o, gr in zip(pos, O, Gr):
        c = str(CLOSE); lo, lgr = o.get(c), gr.get(c)
        common = [k for k in o if k in gr and k != c][:10]
        # without </think>, the rest must be the same distribution up to one normalising shift
        sh = [gr[k] - o[k] for k in common]
        print(f"{uid.split('/',1)[1]:28s} pos {p:6d}: P(</think>) offline orig {np.exp(off['offline_orig']):.4f} live orig {np.exp(lo) if lo is not None else '<top30':} | offline graft {np.exp(off['offline_graft']):.4f} live graft {np.exp(lgr) if lgr is not None else '<top30'} | other tokens shift spread {np.ptp(sh):.4f}")
