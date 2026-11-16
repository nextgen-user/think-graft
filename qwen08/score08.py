"""Reference-compatible math-verify 0.9.0 and official strict prompt-level IFEval.
Run with .venv-eval/bin/python score08.py results/<set>-edited.jsonl.
Reports avg@n, with bootstrap intervals resampling entire problems.
"""
import collections, functools, hashlib, json, pathlib, re, sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from math_verify import parse, verify

# Extraction follows reference/mathqa.py (RikiyaT/lm_understand-logs,
# exp8_math/code/mathqa.py). The unchanged reference is included alongside it.
def last_box(text):
    text=text.split('</think>')[-1].strip() if '</think>' in text else text
    idx=text.rfind('\\boxed')
    if idx<0: idx=text.rfind('\\fbox')
    if idx<0: return None
    key='\\boxed' if text.startswith('\\boxed',idx) else '\\fbox'
    rest=text[idx+len(key):]; r=rest.lstrip()
    if r.startswith('{'):
        depth=0
        for k,ch in enumerate(r):
            depth+=(ch=='{')-(ch=='}')
            if depth==0: return r[1:k].strip() or None
        return None
    if key=='\\boxed' and rest.startswith(' '):
        return rest.strip().split('$')[0].split('\n')[0].strip() or None
    return None

@functools.lru_cache(maxsize=1<<16)
def parsed(s): return parse('\\boxed{'+s+'}',parsing_timeout=5)

def math_correct(text,gold):
    pred=last_box(text)
    if pred is None or gold is None: return False,pred
    g,p=parsed(gold),parsed(pred)
    if not g or not p: return False,pred
    try: return bool(verify(g,p,timeout_seconds=5)),pred
    except Exception: return False,pred

def ifeval(row):
    from instruction_following_eval import instructions_registry as IR
    response=row['text'].split('</think>')[-1].strip() if '</think>' in row['text'] else ''
    dataset=json.load(open(ROOT/'data/ifeval.json'))
    prompt=next(r['prompt'] for r in dataset if f'ifeval/{r["key"]}'==row['id'])
    checks=[]
    for iid,kw in zip(row['instruction_id_list'],row['kwargs']):
        obj=IR.INSTRUCTION_DICT[iid](iid)
        obj.build_description(**{k:v for k,v in (kw or {}).items() if v is not None})
        args=obj.get_instruction_args()
        if args and 'prompt' in args: obj.build_description(prompt=prompt)
        checks.append(bool(response) and bool(obj.check_following(response)))
    return all(checks),checks

def main():
    path=pathlib.Path(sys.argv[1]); cache=path.with_suffix('.scored.jsonl')
    known={r['key']:r for r in map(json.loads,cache.open())} if cache.exists() else {}
    rows={(r['id'],r.get('sample',0)):r for r in map(json.loads,path.open())}
    scored=[]
    with cache.open('a') as f:
        for (uid,sample),row in rows.items():
            key=hashlib.sha256(json.dumps([uid,sample,row['text'],row.get('expected'),row.get('instruction_id_list'),row.get('kwargs')],sort_keys=True).encode()).hexdigest()
            if key in known: s=known[key]
            else:
                ok,pred=ifeval(row) if row['set']=='ifeval' else math_correct(row['text'],row['expected'])
                ids=row.get('gen_ids',[]); close=ids.index(248069) if 248069 in ids else len(ids)
                s=dict(key=key,id=uid,sample=sample,correct=ok,pred=pred,n_gen=row['n_gen'],thinking=close,cap=row['stop_type']=='limit')
                f.write(json.dumps(s)+'\n');f.flush()
            scored.append(s)
    by=collections.defaultdict(list)
    for r in scored: by[r['id']].append(r)
    means=np.array([np.mean([r['correct'] for r in rs]) for rs in by.values()])
    rng=np.random.default_rng(20261116)
    boot=np.array([rng.choice(means,len(means),replace=True).mean() for _ in range(10000)])
    result=dict(problems=len(by),generations=len(scored),samples_per_problem=dict(collections.Counter(map(len,by.values()))),
                accuracy=float(means.mean()),accuracy_problem_bootstrap_95CI=np.quantile(boot,[.025,.975]).tolist(),
                cap_rate=float(np.mean([r['cap'] for r in scored])),mean_tokens=float(np.mean([r['n_gen'] for r in scored])),
                mean_thinking_tokens=float(np.mean([r['thinking'] for r in scored])),
                comparison='External reference scores are not a controlled paired baseline.')
    path.with_suffix('.summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
