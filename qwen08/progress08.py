"""Live, cached grading plus comparisons on the same completed problem IDs.
Reference uses one globally selected sample index per dataset. Token lengths
are reconstructed from first-sample texts and shown only for reference index0.
"""
import argparse, collections, fcntl, hashlib, json, pathlib, time, urllib.request
from score08 import math_correct, ifeval
ROOT=pathlib.Path(__file__).resolve().parent

def read_rows(path):
    if not path.exists(): return []
    rows=[]
    for line in path.open():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    return rows

def pct(n,d): return f'{100*n/d:.1f}%' if d else '-'
def average(values):return sum(values)/len(values) if values else None

def references_for(name,token_sets):
    refs={};indices={}
    for arm in ('base','edited'):
        path=ROOT/f'data/astra_scores_{arm}_{name}.json'
        if path.exists():
            data=json.loads(path.read_text())
            refs[arm]={r['id']:r['correct'] for r in data['per_problem']}
            indices[arm]=data.get('selected_sample_index',0)
    tokens={arm:{uid:r for uid,r in values.items() if r.get('sample_index',0)==indices.get(arm,0)}
            for arm,values in token_sets.get(name,{}).items()}
    return refs,indices,tokens

def comparison(name,rows,reference):
    refs,indices,tokens=reference;ours={r['id']:r for r in rows}
    print(f'\n{name.upper()}: COMPARISON ON OUR COMPLETED PROBLEMS ONLY')
    if not rows:
        print('No completed results yet; comparison will appear automatically.')
        return
    print(f'{"REFERENCE":<11} {"ACC N":>6} {"OURS ACC":>10} {"REF ACC":>10} {"TOKEN N":>8} {"OURS AVG":>11} {"REF AVG ~":>12} {"TOKENS SAVED":>13}')
    for arm,label in [('base','Unedited'),('edited','RikiyaT')]:
        valid=sorted(u for u in set(ours)&set(refs.get(arm,{})) if ours[u]['correct'] is not None)
        oa=average([float(ours[u]['correct']) for u in valid])
        ra=average([float(refs[arm][u][indices[arm]]) for u in valid])
        tids=sorted(set(ours)&set(tokens.get(arm,{})))
        ot=average([ours[u]['tokens'] for u in tids])
        rt=average([tokens[arm][u]['output_tokens_retokenized'] for u in tids])
        fmt=lambda v:f'{v*100:.1f}%' if v is not None else '-'
        num=lambda v:f'{v:,.0f}' if v is not None else '-'
        print(f'{label:<11} {len(valid):>6} {fmt(oa):>10} {fmt(ra):>10} {len(tids):>8} {num(ot):>11} {num(rt):>12} {fmt(1-ot/rt) if ot is not None and rt else "-":>13}')
    print('Reference sample indices: '+', '.join(f'{label}={indices[arm]}' for arm,label in [('base','Unedited'),('edited','RikiyaT')] if arm in indices))

def recent_results(name,rows,reference,show_all):
    if not rows:return
    refs,indices,tokens=reference
    print(f'\nPER-PROBLEM {name.upper()} '+('(all completed)' if show_all else '(last12 completed)'))
    print(f'{"PROBLEM":<40} {"OURS":>7} {"TOKENS":>8} {"BASE":>7} {"TOKENS ~":>10} {"RIKIYA":>8} {"TOKENS ~":>10}')
    for row in rows if show_all else rows[-12:]:
        uid=row['id'];label=uid.removeprefix('test/').removesuffix('.json')
        def cell(arm):
            good=refs.get(arm,{}).get(uid)
            return ('RIGHT' if good[indices[arm]] else 'WRONG') if good is not None else '-'
        def tok(arm):
            entry=tokens.get(arm,{}).get(uid)
            return f'{entry["output_tokens_retokenized"]:,}' if entry else '-'
        status='RIGHT' if row['correct'] is True else ('WRONG' if row['correct'] is False else 'ERROR')
        if row['cap']:status+='*'
        print(f'{label:<40} {status:>7} {row["tokens"]:>8,} {cell("base"):>7} {tok("base"):>10} {cell("edited"):>8} {tok("edited"):>10}')
    print('* = output limit; reference status uses one fixed sample index per benchmark.')

def repair_progress():
    print('TRAINING-ONLY GRAFT REPAIR | benchmark stopped; previous results archived')
    state=ROOT/'results/run-state.json'
    if state.exists():
        detail=json.loads(state.read_text()).get('detail')
        if detail: print(detail)
    print('80 MATH-training problems for fitting; 16 separate training problems for calibration.')
    print('No benchmark examples enter this fit/calibration. Calibration budget: 8,192 output tokens.\n')
    print(f'{"CALIBRATION ARM":<24} {"DONE":>8} {"CORRECT":>9} {"CAPPED":>8} {"AVG OUTPUT":>12}')
    for label,name in [('Previous graft','control-old'),('Loop-aware strength 12','calibration-a12'),('Loop-aware strength 24','calibration-a24'),('Head recovery penalty 4','calibration-head-g4'),('Head recovery penalty 8','calibration-head-g8'),('Head recovery penalty 12','calibration-head-g12'),('Combined FFN strength .5','calibration-combined-s0.5'),('Combined FFN strength 1','calibration-combined-s1'),('Combined FFN strength 2','calibration-combined-s2'),('Recalibrated FFN + head','calibration-aligned-combined'),('Recalibrated FFN .5','calibration-aligned-s0.5'),('Recalibrated FFN 2','calibration-aligned-s2'),('RikiyaT (matched run)','calibration-rikiyat')]:
        if name=='calibration-aligned-s2' and (ROOT/'loop-repair/aligned-s2/status.json').exists():
            print(f'{label:<24} REJECTED: fitting-prefix diagnostic failed; no live evaluation')
            continue
        rs=read_rows(ROOT/f'loop-repair/{name}.jsonl')
        good=sum(math_correct(r['text'],r['expected'])[0] for r in rs)
        print(f'{label:<24} {str(len(rs))+"/16":>8} {good:>9} {sum(r["stop_type"]=="limit" for r in rs):>8} {sum(r["n_gen"] for r in rs)/len(rs) if rs else 0:>12,.0f}')
    finish=read_rows(ROOT/'loop-repair/finish-cue-trials.jsonl')
    if finish:
        correct=sum(math_correct(r['text'],r['expected'])[0] for r in finish)
        print(f'\nFit-only answer-cue diagnostic: {len(finish)}/32 done, {correct} correct (forced cue; not a benchmark).')
    fresh=[]
    for label,name in [('Previous graft','fresh-old'),('Selected output-head repair','fresh-selected')]:
        rs=read_rows(ROOT/f'loop-repair/{name}.jsonl')
        if rs:
            good=sum(math_correct(r['text'],r['expected'])[0] for r in rs)
            fresh.append(f'{label}: {len(rs)}/16 done, {good} correct, {sum(r["stop_type"]=="limit" for r in rs)} capped')
    if fresh: print('\nFRESH TRAINING CHECK (excluded from fit and development):\n'+'\n'.join(fresh))
    try:
        slots=json.load(urllib.request.urlopen('http://localhost:8200/slots',timeout=2))
        counts=[]
        for slot in slots:
            if not slot.get('is_processing'):continue
            t=slot['next_token'];t=t[0] if isinstance(t,list) else t;counts.append(str(t['n_decoded']))
        print('\nLIVE: '+(', '.join(counts)+' tokens decoded' if counts else 'server idle'))
    except Exception:print('\nLIVE: server stopped between stages')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true',help='Show every completed problem for the selected dataset')
    parser.add_argument('--dataset',choices=['math500','gsm8k','ifeval'],help='Per-problem table; default: current benchmark')
    args=parser.parse_args()
    state=ROOT/'results/run-state.json'
    if state.exists() and json.loads(state.read_text()).get('phase')=='repair':
        repair_progress();return
    config_path=ROOT/'results/evaluation-config.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else {}
    lock=(ROOT/'results/progress-cache.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    cachepath=ROOT/'results/progress-cache.json'
    cache=json.loads(cachepath.read_text()) if cachepath.exists() else {}
    datasets={};errors=[]
    for name,n in [('math500',500),('gsm8k',1319),('ifeval',541)]:
        rows={r['id']:r for r in read_rows(ROOT/f'results/{name}-edited.jsonl')}
        scored=[]
        for uid,row in rows.items():
            if row.get('seed')!=0 or row.get('sample')!=0:
                raise ValueError(f'Unexpected seed/sample in {name}: {uid}; this view expects one sample, seed 0')
            if config.get('model_id') and row.get('model_id')!=config['model_id']:
                raise ValueError(f'Model identity mismatch in {name}: {uid}')
            key=hashlib.sha256(json.dumps([name,uid,row['text'],row.get('expected'),row.get('instruction_id_list'),row.get('kwargs')],sort_keys=True).encode()).hexdigest()
            try:
                if key not in cache:
                    ok,pred=ifeval(row) if name=='ifeval' else math_correct(row['text'],row['expected'])
                    cache[key]=dict(correct=bool(ok),pred=pred)
                score=cache[key]
            except Exception as exc:
                errors.append(f'{uid}: {type(exc).__name__}: {exc}'); score=dict(correct=None,pred=None)
            scored.append(dict(id=uid,**score,tokens=row.get('n_gen') or len(row['gen_ids']),cap=row['stop_type']=='limit'))
        datasets[name]=(n,scored)
    tmp=cachepath.with_suffix('.tmp');tmp.write_text(json.dumps(cache));tmp.replace(cachepath)
    fcntl.flock(lock,fcntl.LOCK_UN)

    label=config.get('model_label','our one-row graft, beta 12')
    print(f'Qwen3.5-0.8B | {label} | ONE sample per problem | seed 0')
    print('Output tokens = thinking + final answer. Presence penalty 1.5; temperature 0.6.\n')
    print(f'{"BENCHMARK":<11} {"DONE":>10} {"LEFT":>6} {"CORRECT":>8} {"WRONG":>7} {"ACC":>8} {"CAPPED":>7} {"OUTPUT TOTAL":>14} {"AVG/ANSWER":>11}')
    for name,(n,rs) in datasets.items():
        right=sum(r['correct'] is True for r in rs);wrong=sum(r['correct'] is False for r in rs)
        total=sum(r['tokens'] for r in rs);cap=sum(r['cap'] for r in rs)
        print(f'{name:<11} {str(len(rs))+"/"+str(n):>10} {n-len(rs):>6} {right:>8} {wrong:>7} {pct(right,right+wrong):>8} {cap:>7} {total:>14,} {total/len(rs) if rs else 0:>11,.0f}')
    if config.get('start_unix'):
        elapsed=max(0,time.time()-config['start_unix'])
        count=sum(len(rs) for _,rs in datasets.values())
        all_count=sum(n for n,_ in datasets.values())
        print(f'\nOVERALL: {count}/{all_count} completed; elapsed {elapsed/3600:.2f} hours.')
        estimate=config.get('latest_remaining_eta_hours')
        if estimate and config.get('latest_eta_unix'):
            age=max(0,(time.time()-config['latest_eta_unix'])/3600)
            lo,hi=(max(0,h-age) for h in estimate)
            if hi>0:
                print(f'Latest rough ETA: {lo:.0f}–{hi:.0f} hours remaining (estimate updated{age:.1f}h ago; token lengths may differ by dataset).')
            else:print('Previous ETA has elapsed; use current throughput and stage progress for a new estimate.')
        elif config.get('planning_eta_hours'):
            lo,hi=config['planning_eta_hours']
            print(f'Initial planning estimate: {lo}–{hi} hours total; actual duration depends on token-limit hits.')

    token_path=ROOT/'data/reference-output-token-counts.json'
    token_sets=json.loads(token_path.read_text())['sets'] if token_path.exists() else {}
    references={name:references_for(name,token_sets) for name in datasets}
    current=next((name for name,(n,rs) in datasets.items() if len(rs)<n),'ifeval')
    print(f'CURRENT BENCHMARK: {current.upper()}')
    for name,(_,rows) in datasets.items():
        comparison(name,rows,references[name])
    print('\nACC N/TOKEN N = shared problem IDs with scores/token counts for the selected reference sample.')
    print('RikiyaT uses its globally highest-scoring sample index per benchmark, fixed across all problems.')
    print('This selects the strongest reference index; it is not an unbiased multi-sample average.')
    print('~ Tokens are retokenized published sample0 texts only; omitted when selected reference index is1.')
    print('Reference runs use a different engine; identical seed numbers do not imply identical random draws.')
    selected=args.dataset or current
    recent_results(selected,datasets[selected][1],references[selected],args.all)
    print('\nREFERENCE TOTALS (complete reference sets; not matched to a partial run)')
    for name,reference in references.items():
        refs,indices,_=reference
        for arm,label in [('base','Unedited'),('edited','RikiyaT')]:
            outcomes=refs.get(arm,{})
            good=sum(bool(v[indices[arm]]) for v in outcomes.values()) if outcomes else 0
            print(f'{name.upper()} {label}: {good}/{len(outcomes)} = {pct(good,len(outcomes))}; sample index{indices.get(arm,"-")}.')
    print('Early completed results favor faster cases; unfinished long samples are not counted as successes.')
    try:
        slots=json.load(urllib.request.urlopen('http://localhost:8200/slots',timeout=2))
        active=[s for s in slots if s.get('is_processing')]
        counts=[]
        for slot in active:
            t=slot['next_token'];t=t[0] if isinstance(t,list) else t;counts.append(str(t['n_decoded']))
        print('\nLIVE: '+(f'{len(active)} streams; tokens so far: '+', '.join(counts) if counts else 'server idle'))
        if config.get('start_unix') and time.time()-config['start_unix']>=60:
            elapsed=time.time()-config['start_unix']
            generated=sum(r['tokens'] for _,rs in datasets.values() for r in rs)+sum(map(int,counts))
            rate=generated/elapsed
            print(f'Observed aggregate throughput: ~{rate:.0f} output tokens/s (completed + active generations).')
            if rate>0:
                remaining=sum(n-len(rs) for n,rs in datasets.values())
                cap_tokens=max(0,remaining*config.get('output_budget',32768)-sum(map(int,counts)))
                print(f'If every remaining answer uses its full budget: ~{cap_tokens/rate/3600:.1f} hours left at this observed rate.')
    except Exception: print('\nLIVE: server not reachable')
    if errors:
        print('GRADING ERRORS (not counted wrong):')
        print('\n'.join(errors[:10]))

if __name__=='__main__':main()
