"""Read-only review of saved benchmark outputs and published reference scores.
No inference, parameter fitting, or changes to the running evaluation.
"""
import collections,concurrent.futures,datetime,gzip,hashlib,json,pathlib,platform,subprocess,urllib.request
import numpy as np
from scipy.stats import binomtest,wilcoxon
from score08 import math_correct,last_box

ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'publication';OUT.mkdir(exist_ok=True)
def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def read(path):return [json.loads(line) for line in pathlib.Path(path).open()]

def repetition_signals(ids,window=4096,n=8,span=128,threshold=.8):
    last={};recent=collections.deque(maxlen=span);density=[];first=None;total=0
    for i in range(len(ids)):
        repeated=False
        if i>=n-1:
            key=tuple(ids[i-n+1:i+1]);old=last.get(key)
            repeated=old is not None and i-old<=window;last[key]=i
        recent.append(int(repeated));total+=int(repeated)
        if i>=512 and len(recent)==span and sum(recent)/span>=threshold and first is None:first=i
        if i%128==0:density.append([i,sum(recent)/len(recent)])
    return dict(first_alert_token=first,repeated_fraction=total/max(len(ids),1),density=density)

def main():
    revision=json.load(open(ROOT/'reference/rikiyat/upstream-tree-filtered.json'))['tree_sha']
    cache=ROOT/'reference/rikiyat/upstream/unedited/parts'
    def fetch(shard):
        stem=f'exp8_math/eval/unedited/parts/math500_s{shard}of10/'
        target=cache/f'math500_s{shard}of10';target.mkdir(parents=True,exist_ok=True)
        result={}
        for name in ('math500_per_problem.json','scores.json','math500_first_samples.jsonl.gz'):
            path=target/name
            if not path.exists():path.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{revision}/{stem}{name}',timeout=60).read())
            result[name]=path
        return result
    with concurrent.futures.ThreadPoolExecutor(5) as pool:parts=list(pool.map(fetch,range(10)))
    baseline=[];texts=[]
    for part in parts:
        baseline.extend(json.load(open(part['math500_per_problem.json'])))
        texts.extend(json.loads(l) for l in gzip.open(part['math500_first_samples.jsonl.gz'],'rt'))
    baseline_by={r['id']:r for r in baseline};assert len(baseline)==len(baseline_by)==500
    save('unedited-full-math500-reference.json',dict(per_problem=baseline,correct_by_index=[sum(r['correct'][i] for r in baseline) for i in range(4)],source_revision=revision,source='Original upstream unedited full MATH500 evaluation; distinct from the25-problem HF smoke test.'))
    rows=read(ROOT/'results/math500-edited.jsonl');assert len(rows)==500
    ref=json.load(open(ROOT/'data/astra_scores_edited_math500.json'))
    reference={r['id']:bool(r['correct'][0]) for r in ref['per_problem']}
    scored=[];wins=[];losses=[];signals=[]
    for row in rows:
        good,pred=math_correct(row['text'],row['expected']);uid=row['id'];assert uid in baseline_by
        if good and not reference[uid]:wins.append(uid)
        if reference[uid] and not good:losses.append(uid)
        sig=repetition_signals(row['gen_ids']);close=row['gen_ids'].index(248069) if 248069 in row['gen_ids'] else None
        cap=row['stop_type']=='limit';extra=max(0,row['gen_ids'].count(248069)-1)
        scored.append(dict(id=uid,correct=bool(good),prediction=pred,tokens=row['n_gen'],cap=cap,closed_thinking=close is not None,thinking_tokens=close if close is not None else len(row['gen_ids']),extra_close_tags=extra))
        signals.append(dict(id=uid,correct=bool(good),cap=cap,closed_thinking=close is not None,**{k:v for k,v in sig.items() if k!='density'}))
    assert len({r['id'] for r in rows})==500
    delta=np.array([int(s['correct'])-int(reference[s['id']]) for s in scored])
    rng=np.random.default_rng(20261117)
    boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(10000)])
    discordant=len(wins)+len(losses)
    caps=[s for s in scored if s['cap']]
    metric=dict(problems=500,ours_correct=sum(s['correct'] for s in scored),reference_correct=sum(reference.values()),difference_percentage_points=100*float(delta.mean()),
                new_correct=wins,new_wrong=losses,exploratory_difference_bootstrap95_percentage_points=(100*np.quantile(boot,[.025,.975])).tolist(),
                exploratory_mcnemar_exact_p=float(binomtest(len(wins),discordant,.5).pvalue),
                statistical_limit='Descriptive cross-engine comparison, not a controlled causal significance test; strongest global reference sample index selected retrospectively.',
                output_tokens=sum(s['tokens'] for s in scored),mean_output_tokens=float(np.mean([s['tokens'] for s in scored])),
                mean_thinking_tokens=float(np.mean([s['thinking_tokens'] for s in scored])),
                capped=len(caps),capped_correct=sum(s['correct'] for s in caps),capped_without_close=sum(not s['closed_thinking'] for s in caps),capped_after_close=sum(s['closed_thinking'] for s in caps),
                extra_close_tag_cases=sum(s['extra_close_tags']>0 for s in scored),extra_close_tags=sum(s['extra_close_tags'] for s in scored),
                unedited_original_sample0_correct=sum(r['correct'][0] for r in baseline),unedited_reference_problems=500,
                provenance=dict(ours='Local frozen output-head g4 model, seed0,n1',edited_reference='HF independent rerun, seed0,n4, index0',unedited_reference='Original upstream complete benchmark, seed0,n4,index0'))
    token_data=json.load(open(ROOT/'data/reference-output-token-counts.json'))['sets']['math500']['edited']
    ours={r['id']:r for r in scored};ids=sorted(set(token_data)&set(ours));ot=np.array([ours[u]['tokens'] for u in ids]);rt=np.array([token_data[u]['output_tokens_retokenized'] for u in ids])
    metric['matched_token_subset']=dict(problems=len(ids),ours_mean=float(ot.mean()),reference_mean_retokenized=float(rt.mean()),saving=1-float(ot.sum()/rt.sum()),exploratory_wilcoxon_p=float(wilcoxon(ot-rt,method='exact').pvalue),limitation='Only20 published reference first-sample texts; lengths retokenized and approximate, not the500-problem token saving.')
    save('math500-analysis.json',metric);save('math500-scored.json',scored);save('ngram8-diagnostic.json',dict(settings=dict(n=8,lookback_tokens=4096,density_span=128,threshold=.8,minimum_position=512),
                interpretation='Prefix-only alert coverage on existing edited outputs. Not precision/recall, not an intervention comparison, and not evidence that stopping at the alert yields a correct final answer.',
                groups={name:dict(traces=len(rs),alerted=sum(r['first_alert_token'] is not None for r in rs)) for name,rs in [('capped',[r for r in signals if r['cap']]),('finished',[r for r in signals if not r['cap']])]},per_problem=signals))
    (OUT/'unedited-reference-first-samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in texts))
    print(json.dumps(metric,indent=2),flush=True)
    print('NGRAM_DIAGNOSTIC',json.dumps({name:dict(n=len(rs),alerted=sum(r['first_alert_token'] is not None for r in rs)) for name,rs in [('capped',[r for r in signals if r['cap']]),('finished',[r for r in signals if not r['cap']])]}),flush=True)

if __name__=='__main__':main()
