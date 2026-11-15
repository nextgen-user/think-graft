# Counterfactual continuation (Nov 14 08:00): would the ORIGINAL model's longer checking have caught an error the
# grafted run closed on?  Takes a grafted run's thinking up to its first </think>, appends it to the same prompt and lets
# a given pack continue (several seeds, same sampling as bench_think).  Records whether it closes, how many more
# thinking tokens it spends, whether the expected answer shows up and what it answers.
# usage: cf.py TAG SRC_VARIANT ID SET EXE ENV.json ARGS.json SEEDS(a,b,c) BUDGET  -> runs/cf-results.jsonl
import sys, json, time, re, pathlib, hashlib, subprocess, urllib.request
A = sys.argv[1:]; sys.argv = [sys.argv[0]]; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bench_think as BTH
G, BT = BTH.G, BTH.BT
tag, src, pid, sset, exe, envf, argf, seeds, budget = A[0], A[1], A[2], A[3], A[4], A[5], A[6], [int(s) for s in A[7].split(',')], int(A[8])
D = G.D
row = [json.loads(l) for l in open(D / 'runs/think-results.jsonl')]
row = [r for r in row if r['variant'] == src and r['id'] == pid][-1]
prob = [p for p in {'tough': BT.PROBLEMS, 'held': BTH.HELD}[sset] if p['id'] == pid][0]
prompt = G.TPL.render([dict(role='user', content=prob['task'] + BT.SUFFIX)], enable_thinking=True, reasoning_effort='max')
prefix = row['text'].split('</think>')[0]
ids = G.TOK.encode(prompt + prefix, parse_special=True)
print(f'{tag}: {pid} from {src}, prefix {len(G.TOK.encode(prefix, parse_special=True))} thinking tokens, ends {prefix[-80:]!r}', flush=True)
env = json.loads(pathlib.Path(envf).read_text()); args = json.loads(pathlib.Path(argf).read_text())
for _ in range(120):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8095/metrics', timeout=5) as f:
            if json.load(f)['live']['state'] == 'idle': break
    except Exception: pass
    time.sleep(5)
sha = hashlib.sha256(pathlib.Path(G.live['exe']).read_bytes()).hexdigest()
subprocess.run(['systemctl', '--user', 'stop', 'strata-deepseek'], check=True)
try:
    eng = G.Engine(tag, exe, env, args)
    try:
        for s in seeds:
            eng.p.stdin.write(f'GEN {budget} temperature=1.0 top_p=1.0 seed={s} ' + ','.join(map(str, ids)) + '\n'); eng.p.stdin.flush()
            toks = []; t0 = time.monotonic()
            while True:
                l = eng.line()
                if l.startswith('T '):
                    toks.append(int(l.split()[1]))
                    if len(toks) % 100 == 0: (D / 'runs/think-progress.txt').write_text(f'{tag} seed {s}: {len(toks)} tokens in {time.monotonic() - t0:.0f} s\n')
                elif l.startswith('DONE '):
                    f = l.split(); info = dict(n=int(f[1]), prompt_ms=float(f[3]), gen_ms=float(f[4]), finish=f[5]); break
                elif l.startswith('ERR'): raise RuntimeError(l)
            txt = G.TOK.decode([t for t in toks if t != 1]); closed = '</think>' in txt
            more = len(G.TOK.encode(txt.split('</think>')[0], parse_special=True))
            final = txt.split('</think>')[-1] if closed else ''
            m = re.findall(r'ANSWER:\s*(.+)', final); ans = m[-1].strip() if m else None
            r = dict(tag=tag, src=src, id=pid, seed=s, closed=closed, more_think=more, expected_seen=prob['expected'] in txt.split('</think>')[0],
                     answer=ans, correct=ans is not None and BTH.value(ans) == BTH.value(prob['expected']), **info, text=txt)
            with open(D / 'runs/cf-results.jsonl', 'a') as fo: fo.write(json.dumps(r) + '\n')
            print(json.dumps({k: r[k] for k in ('seed', 'closed', 'more_think', 'expected_seen', 'answer', 'correct', 'n', 'gen_ms')}), flush=True)
    finally: eng.close()
finally:
    subprocess.run(['systemctl', '--user', 'start', 'strata-deepseek'], check=True)
    assert hashlib.sha256(pathlib.Path(G.live['exe']).read_bytes()).hexdigest() == sha
    print('Production restored.', flush=True)
