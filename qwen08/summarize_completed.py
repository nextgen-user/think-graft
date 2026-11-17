"""Audit saved scored generations; no model calls or refitting.
Usage: .venv/bin/python summarize_completed.py /path/to/original/experiment
The argument supplies original results/data, while output lives beside this script.
"""
import collections,gzip,hashlib,json,pathlib,shutil,sys
from scipy.stats import binomtest
ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else ROOT
out={'protocol':'One sample per problem, seed 0, frozen three-row output-head model. External references use different serving engines.', 'benchmarks':{}}
for dataset,n in [('math500',500),('gsm8k',1319),('ifeval',541)]:
 path=SOURCE/'results'/f'{dataset}-edited.jsonl'
 if path.exists(): rows=[json.loads(l) for l in path.open()]
 else:
  with gzip.open(str(path)+'.gz','rt') as f:rows=[json.loads(l) for l in f]
 cache=SOURCE/'results'/f'{dataset}-edited.scored.jsonl'
 if cache.exists():scored={r['key']:r for r in map(json.loads,cache.open())}
 else:
  with gzip.open(str(cache)+'.gz','rt') as f:scored={r['key']:r for r in map(json.loads,f)}
 assert len(rows)==n and len({r['id'] for r in rows})==n
 ss=[]
 for r in rows:
  assert r['sample']==0 and r['seed']==0 and r['model_id']=='output-head-repair-f3b305519ce9'
  g=r['generation_settings']
  for k,v in dict(top_k=20,repeat_last_n=33792,repeat_penalty=1,frequency_penalty=0,min_p=0,presence_penalty=1.5).items():assert g[k]==v
  assert abs(g['temperature']-.6)<1e-6 and abs(g['top_p']-.95)<1e-6
  assert g['samplers']==['penalties','top_k','top_p','temperature']
  key=hashlib.sha256(json.dumps([r['id'],r['sample'],r['text'],r.get('expected'),r.get('instruction_id_list'),r.get('kwargs')],sort_keys=True).encode()).hexdigest()
  s=scored[key];assert s['n_gen']==r['n_gen'] and s['cap']==(r['stop_type']=='limit');ss.append(s)
 summary=json.load(open(SOURCE/'results'/f'{dataset}-edited.summary.json'))
 correct=sum(s['correct'] for s in ss);caps=sum(s['cap'] for s in ss)
 assert abs(summary['accuracy']-correct/n)<1e-12 and abs(summary['cap_rate']-caps/n)<1e-12
 refs=json.load(open(ROOT/'data'/f'astra_scores_edited_{dataset}.json'))
 index=refs.get('selected_sample_index',0);refrows=refs['per_problem'];by={r['id']:r for r in refrows}
 assert set(by)=={r['id'] for r in rows}
 ours_only=sum(s['correct'] and not by[s['id']]['correct'][index] for s in ss)
 ref_only=sum(not s['correct'] and by[s['id']]['correct'][index] for s in ss)
 refcorrect=sum(r['correct'][index] for r in refrows)
 extra=[max(0,r['text'].count('</think>')-1) for r in rows]
 report=dict(problems=n,correct=correct,accuracy_percent=100*correct/n,capped=caps,cap_percent=100*caps/n,total_output_tokens=sum(r['n_gen'] for r in rows),mean_output_tokens=summary['mean_tokens'],accuracy_bootstrap95CI=summary['accuracy_problem_bootstrap_95CI'],capped_after_close=sum(s['cap'] and '</think>' in r['text'] for r,s in zip(rows,ss)),extra_closing_tags=sum(extra),responses_with_extra_closing_tags=sum(x>0 for x in extra),selected_reference=dict(sample_index=index,correct=refcorrect,accuracy_percent=100*refcorrect/n,ours_only=ours_only,reference_only=ref_only,descriptive_mcnemar_p=binomtest(ours_only,ours_only+ref_only).pvalue if ours_only+ref_only else 1,limitation='Retrospectively selected strongest global reference index; external serving engine. Not a causal superiority test.'))
 out['benchmarks'][dataset]=report
 for suffix in ('.jsonl','.scored.jsonl'):
  source=SOURCE/'results'/f'{dataset}-edited{suffix}'
  if source.exists():
   with source.open('rb') as src,gzip.GzipFile(str(ROOT/'results'/(source.name+'.gz')),'wb',mtime=0) as dest:shutil.copyfileobj(src,dest)
 shutil.copyfile(SOURCE/'results'/f'{dataset}-edited.summary.json',ROOT/'results'/f'{dataset}-edited.summary.json') if SOURCE!=ROOT else None
out['published_reference']={'base':dict(math500=7.9,gsm8k=13.8,ifeval=21.0,math500_cap=93.0),'astra':dict(math500=48.1,gsm8k=67.1,ifeval=62.3,math500_cap=33.1),'source':'https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming','interpretation':'Author-reported multi-sample aggregates, distinct from selected-index comparison above. Cap column is MATH-500 only.'}
(ROOT/'publication/completed-benchmarks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
