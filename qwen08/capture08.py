"""Resumable head/state capture using the SAME libllama build as generation.
Stores fp16 hidden states and 21 fp32 sampler statistics per row, without a CPU head matmul.
"""
import hashlib, json, os, pathlib, subprocess, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
model = os.environ.get('SRC', str(ROOT/'models/Qwen3.5-0.8B-BF16.gguf'))
exe = ROOT/'tools/capture_stats'
rows = {(r['id'], r.get('sample', 0)): r for r in map(json.loads, src.open())}
order = (ROOT/'data/fit_ids.txt').read_text().splitlines()
assert set(rows) == {(uid,0) for uid in order}, 'Fit input is incomplete or has unexpected samples'
args, idx = [], []
for uid in order:
    r = rows[uid,0]; ids = np.asarray(r['prompt_ids']+r['gen_ids'], dtype=np.int32)
    sha = hashlib.sha256(ids.tobytes()).hexdigest(); stem = sha[:20]
    ip, xp, sp = (out/(stem+s) for s in ('.i32','.f16','.stats.f32'))
    done = pathlib.Path(str(xp)+'.done')
    ready = (done.exists() and xp.exists() and xp.stat().st_size == len(ids)*1024*2
             and sp.exists() and sp.stat().st_size == len(ids)*21*4)
    if not ready:
        ids.tofile(ip); args += [str(ip),str(xp),str(sp)]
    idx.append(dict(id=uid,sample=0,file=xp.name,stats=sp.name,n_prompt=len(r['prompt_ids']),
                    n=len(ids),expected=r['expected'],token_sha256=sha))
if args:
    with (out/'capt.log').open('a') as log:
        subprocess.run([str(exe), model, '256', '248069']+args,check=True,stderr=log)
for e in idx:
    assert (out/e['file']).stat().st_size == e['n']*1024*2
    assert (out/e['stats']).stat().st_size == e['n']*21*4
meta = dict(schema=2,dim=1024,chunk=256,presence_penalty=1.5,top_k=20,top_p=0.95,
            temperature=0.6,order=['penalties','top_k','top_p','temperature'],
            executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),traces=idx)
tmp=out/'index.json.tmp'; tmp.write_text(json.dumps(meta,indent=2)+'\n'); tmp.replace(out/'index.json')
print(len(idx),'traces captured',flush=True)
