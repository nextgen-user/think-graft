# Capture head inputs for every trace with tools/capt (libllama, causal chunks of 1024, cache kept): see capture.py's
# note.  usage: python3 capture2.py traces.jsonl OUTDIR  -> OUTDIR/<n>.npy (float16, tokens x 2560) + OUTDIR/index.json
import json, sys, os, subprocess, numpy as np
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
M = os.environ.get('QWEN_GGUF', 'models/Qwen3.5-4B-Q8_0.gguf'); CAPT = os.environ.get('CAPT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'capt'))
rows = [json.loads(l) for l in open(src)]; args = []
for n, r in enumerate(rows):
    np.array(r['prompt_ids'] + r['gen_ids'], np.int32).tofile(f'{out}/{n}.i32'); args += [f'{out}/{n}.i32', f'{out}/{n}.f16']
subprocess.run([CAPT, M, '1024'] + args, check=True, stderr=open(f'{out}/capt.log', 'w'))
idx = []
for n, r in enumerate(rows):
    E = np.fromfile(f'{out}/{n}.f16', np.float16).reshape(-1, 2560); assert len(E) == len(r['prompt_ids']) + len(r['gen_ids'])
    np.save(f'{out}/{n}.npy', E); os.remove(f'{out}/{n}.f16'); os.remove(f'{out}/{n}.i32')
    idx.append(dict(id=r['id'], file=f'{n}.npy', n_prompt=len(r['prompt_ids']), n=len(E), expected=r['expected']))
json.dump(idx, open(f'{out}/index.json', 'w')); print(len(idx), 'traces captured')
