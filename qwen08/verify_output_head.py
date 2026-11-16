"""Verify an output-head-only candidate, including multiple changed token rows.
Reads only base MATH-training captures. PORT MODEL -> results/live-validation.json.
"""
import hashlib,json,pathlib,sys,urllib.request
import numpy as np
from fit08 import gguf,SRC
from gguf import quants
ROOT=pathlib.Path(__file__).resolve().parent;port=int(sys.argv[1]);model=pathlib.Path(sys.argv[2])
base=gguf.GGUFReader(SRC);edit=gguf.GGUFReader(str(model));tensors={t.name:t for t in edit.tensors}
for t in base.tensors:
    assert tensors[t.name].tensor_type==t.tensor_type and np.array_equal(t.data,tensors[t.name].data),f'Changed backbone tensor: {t.name}'
assert len(edit.tensors)==len(base.tensors)+1
te=next(t for t in base.tensors if t.name=='token_embd.weight');head=tensors['output.weight']
changed=np.flatnonzero(np.any(te.data!=head.data,axis=1)).tolist()
npz=model.with_suffix('.deltas.npz')
if npz.exists():
    z=np.load(npz);targets=z['target_ids'];deltas=z['effective_deltas']
else:
    targets=np.array([248069]);deltas=np.load(str(model.with_suffix(''))+'-effective-delta.npy')[None,:]
assert sorted(changed)==sorted(targets.tolist()),changed
print('PASS backbone byte comparison; changed output rows:',changed,flush=True)
emb=quants.dequantize(te.data,te.tensor_type).astype(np.float32)
rows={r['id']:r for r in map(json.loads,open(ROOT/'results/fit-traces.jsonl'))}
index=json.load(open(ROOT/'cap-fit-v2/index.json'))['traces'];by={e['id']:e for e in index}
fit=(ROOT/'loop-repair/fit_ids.txt').read_text().splitlines()
choices=[]
# Productive prefixes and repeated-token prefixes are fixed without benchmark data.
for uid in fit:
    r=rows[uid];g=r['gen_ids'];p=len(r['prompt_ids'])
    if len(choices)<2:choices.append((uid,min(p+200,p+len(g)-1)))
    candidates=[j for j in range(1000,min(len(g)-1,14000)) if g[j+1] in [13784,13428]]
    if candidates and len(choices)<4:choices.append((uid,p+candidates[len(candidates)//2]))
    if len(choices)>=4:break

checks=[]
for uid,pos in choices:
    r=rows[uid];e=by[uid];prefix=(r['prompt_ids']+r['gen_ids'])[:pos+1]
    x=np.memmap(ROOT/'cap-fit-v2'/e['file'],np.float16,mode='r',shape=(e['n'],1024))[pos].astype(np.float32)
    logits=emb@x
    logits[targets]+=deltas@x
    logits[np.unique(prefix)]-=np.float32(1.5)
    idx=np.argpartition(logits,-20)[-20:];idx=idx[np.argsort(-logits[idx])]
    val=logits[idx];p=np.exp(val-val[0],dtype=np.float32);p/=p.sum(dtype=np.float32)
    keep=np.r_[True,np.cumsum(p,dtype=np.float32)[:-1]<np.float32(.95)]
    idx=idx[keep];val=val[keep];p=np.exp((val-val[0])/np.float32(.6),dtype=np.float32);p/=p.sum(dtype=np.float32)
    predicted=dict(zip(map(int,idx),map(float,p)))
    body=dict(prompt=prefix,n_predict=1,temperature=.6,top_p=.95,top_k=20,min_p=0.,presence_penalty=1.5,
              repeat_last_n=len(prefix)+1024,repeat_penalty=1.,seed=0,return_tokens=True,cache_prompt=False,
              samplers=['penalties','top_k','top_p','temperature'],n_probs=20,post_sampling_probs=True)
    req=urllib.request.Request(f'http://localhost:{port}/completion',json.dumps(body).encode(),{'Content-Type':'application/json'})
    result=json.load(urllib.request.urlopen(req,timeout=1200))
    blocks=result.get('probs',result.get('completion_probabilities',[]))
    assert len(blocks)==1, f'Unexpected probability response keys: {list(result)}'
    actual={t['id']:t['prob'] for t in blocks[0]['top_probs']}
    err=max(abs(actual.get(t,0)-predicted.get(t,0)) for t in set(actual)|set(predicted))
    rec=dict(id=uid,row=pos,max_probability_error=err,stop_predicted=predicted.get(248069,0),stop_live=actual.get(248069,0))
    checks.append(rec);print(json.dumps(rec),flush=True)
passed=len(checks)==4 and max(c['max_probability_error'] for c in checks)<.04
report=dict(status='passed' if passed else 'needs_review',model_path=str(model),model_sha256=hashlib.file_digest(model.open('rb'),'sha256').hexdigest(),
            unchanged_base_tensors=len(base.tensors),changed_output_rows=changed,checks=checks,
            method='output-head-only; no FFN or other backbone tensor changed')
(ROOT/'results/live-validation.json').write_text(json.dumps(report,indent=2)+'\n')
if not passed:raise SystemExit('Numerical alignment needs review; no benchmark restart.')
print('OUTPUT_HEAD_VERIFIED',flush=True)
