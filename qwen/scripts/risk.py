# Where would edited have stopped thinking on a baseline trace, and what would it have answered there? (Nov 15 2026)
# edited changes only the </think> logit, so a baseline (unedited, thinking on) trace is also a edited trajectory up to
# the point edited closes: P(edited closes at row j) = survival(j) * p_graft(j), exactly.  At every row with real exit
# mass we force </think> after that row and let the grafted model write its answer (greedy), then grade it.
# usage: python3 risk.py capdir traces.jsonl PORT out.json   (PORT = llama-server running the GRAFTED model)
#        python3 risk.py report out.json [more.json ...]
import sys, os, json, urllib.request, numpy as np
CLOSE = 248069; S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S)
from grade import correct, answer_of
if sys.argv[1] == 'report':
    for f in sys.argv[2:]:
        for t in json.load(open(f)):
            wrong = sum(e['mass'] for e in t['exits'] if not e['right']); tot = sum(e['mass'] for e in t['exits'])
            print(f"{t['id']:36s} baseline {'RIGHT' if t['base_right'] else 'wrong'} {t['n_think']:6d} think tok | edited exits before baseline's close: {tot:5.1%}  of which WRONG answer: {wrong:5.1%}  | expected exit at {t['exp_exit']:.0f} tok")
            for e in sorted(t['exits'], key=lambda e: -e['mass'])[:6]:
                if e['mass'] > 0.01: print(f"     at token {e['j']:6d}: mass {e['mass']:.3f} -> answers {e['ans']!r:22s} {'right' if e['right'] else 'WRONG'}   | ...{e['ctx']}")
    sys.exit()
sys.path.insert(0, os.path.join(os.environ.get('LLAMA_CPP', 'llama.cpp'), 'gguf-py')); import gguf; from gguf import quants
cap, tr, port, out = sys.argv[1:5]
R = gguf.GGUFReader(os.environ.get('QWEN_GGUF', 'models/Qwen3.5-4B-Q8_0.gguf')); te = [t for t in R.tensors if t.name == 'token_embd.weight'][0]
E = quants.dequantize(te.data, te.tensor_type).astype(np.float32); g = np.load(os.environ.get('PROBE', 'models/think-graft-probe.npy')); BETA = 14
rows = {json.loads(l)['id']: json.loads(l) for l in open(tr)}
def post(body):
    req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=900))
res = json.load(open(out)) if os.path.exists(out) else []; have = {t['id'] for t in res}
for e in json.load(open(f'{cap}/index.json')):
    if e['id'] in have: continue
    r = rows[e['id']]; gen = r['gen_ids']; npf = e['n_prompt']
    ci = gen.index(CLOSE) if CLOSE in gen else len(gen)
    X = np.load(f"{cap}/{e['file']}").astype(np.float32)[npf - 1: npf - 1 + ci]   # rows deciding gen tokens 0..ci-1
    lc = X @ E[CLOSE]; lse = np.zeros(len(X))
    for a in range(0, len(X), 2048):
        lg = X[a:a + 2048] @ E.T; m = lg.max(1); lse[a:a + 2048] = m + np.log(np.exp(lg - m[:, None]).sum(1))
    l2 = lc + BETA * (X.astype(np.float64) @ g); rest = lse + np.log1p(-np.exp(np.minimum(lc - lse, -1e-9)))
    p = np.exp(l2 - np.logaddexp(rest, l2)); surv = np.exp(np.concatenate([[0], np.cumsum(np.log1p(-np.minimum(p, 1 - 1e-12)))[:-1]]))
    mass = surv * p                                                        # P(edited's first close is at row j)
    # test exits: every row with mass > 0.004, merged into windows of 200 tokens (one probe per window, mass summed)
    exits = []; idx = np.where(mass > 0.004)[0]
    for k in idx:
        if exits and k - exits[-1]['j0'] < 200: exits[-1]['mass'] += float(mass[k]); continue
        exits.append(dict(j0=int(k), j=int(k), mass=float(mass[k])))
    for x in exits:
        j = x['j']                                                         # close after gen token j-1 (row j decides token j)
        a = post({'prompt': r['prompt_ids'] + gen[:j] + [CLOSE], 'n_predict': 1500, 'temperature': 0, 'cache_prompt': False})['content']
        x['ans'] = answer_of('</think>' + a); x['right'] = correct('</think>' + a, r['expected'])
        req = urllib.request.Request(f'http://127.0.0.1:{port}/detokenize', data=json.dumps({'tokens': gen[max(0, j - 30):j]}).encode(), headers={'Content-Type': 'application/json'})
        x['ctx'] = json.load(urllib.request.urlopen(req))['content'].replace('\n', ' / ')[-110:]
    t = dict(id=e['id'], expected=r['expected'], base_right=correct(r['text'], r['expected']), n_think=ci,
             exp_exit=float((mass * np.arange(len(mass))).sum() + (1 - mass.sum()) * ci), exits=exits)
    res.append(t); json.dump(res, open(out, 'w'), indent=1)
    print(f"{e['id']}: {len(exits)} exit windows, total exit mass {mass.sum():.3f}, wrong {sum(x['mass'] for x in exits if not x['right']):.3f}", flush=True)
