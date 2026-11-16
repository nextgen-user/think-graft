"""Cache published GSM8K/IFEval reference scores; never generates model outputs."""
import concurrent.futures,gzip,hashlib,json,pathlib,urllib.request
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parent;D=ROOT/'data'
revision=json.load(open(ROOT/'reference/rikiyat/upstream-tree-filtered.json'))['tree_sha']
base=f'https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{revision}/'
cache=ROOT/'reference/rikiyat/upstream';cache.mkdir(exist_ok=True)
paths=[]
for variant in ('p50c','unedited'):
    for dataset,n in [('gsm8k',10),('ifeval',5)]:
        for shard in range(n):
            stem=f'exp8_math/eval/{variant}/parts/{dataset}_s{shard}of{n}/'
            paths.extend(stem+name for name in (f'{dataset}_per_problem.json','scores.json',f'{dataset}_first_samples.jsonl.gz'))

def download(path):
    target=cache/path.removeprefix('exp8_math/eval/');target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        request=urllib.request.Request(base+path,headers={'User-Agent':'qwen08-reference-audit'})
        data=urllib.request.urlopen(request,timeout=90).read()
        temporary=target.with_suffix(target.suffix+'.partial');temporary.write_bytes(data);temporary.replace(target)
    return target

with concurrent.futures.ThreadPoolExecutor(8) as pool:
    downloaded=list(pool.map(download,paths))
print('Cached',len(downloaded),'published files',flush=True)
T=Tokenizer.from_file(str(ROOT/'models/Qwen3.5-0.8B-hf/tokenizer.json'))
tokens_path=D/'reference-output-token-counts.json';token_data=json.load(open(tokens_path));sample_data={}
gsm=json.load(open(D/'gsm8k_test.json'));ifeval=json.load(open(D/'ifeval.json'))
allowed={'gsm8k':{f'gsm8k/{i}' for i in range(len(gsm))},'ifeval':{f"ifeval/{r['key']}" for r in ifeval}}
def normalize(uid,dataset):
    if dataset=='gsm8k':assert uid.startswith('gsm8k_test_');return 'gsm8k/'+uid.removeprefix('gsm8k_test_')
    assert uid.startswith('ifeval_');return 'ifeval/'+uid.removeprefix('ifeval_')

for dataset,n in [('gsm8k',10),('ifeval',5)]:
    token_data['sets'].setdefault(dataset,{})
    for variant,arm in [('p50c','edited'),('unedited','base')]:
        per=[];samples=[];metadata=[]
        for shard in range(n):
            stem=cache/variant/'parts'/f'{dataset}_s{shard}of{n}'
            rs=json.load(open(stem/f'{dataset}_per_problem.json'));meta=json.load(open(stem/'scores.json'))
            assert meta['_protocol']['name']=='think35'
            sampling=meta['_protocol']['sampling']
            for key,value in dict(seed=0,temperature=.6,top_p=.95,top_k=20,presence_penalty=1.5,max_tokens=32768).items():assert sampling[key]==value
            metadata.append(meta)
            for row in rs:
                uid=normalize(row['id'],dataset);assert uid in allowed[dataset]
                per.append(dict(row,id=uid,source_id=row['id']))
            for line in gzip.open(stem/f'{dataset}_first_samples.jsonl.gz','rt'):
                row=json.loads(line);uid=normalize(row['id'],dataset)
                if dataset=='gsm8k':
                    gold=gsm[int(uid.split('/')[1])]['answer'].split('####')[-1].strip().replace(',','')
                    assert str(row['answer']).strip()==gold
                samples.append(dict(row,id=uid,source_id=row['id']))
        by={r['id']:r for r in per};assert len(by)==len(per) and set(by)==allowed[dataset]
        counts={len(r['correct']) for r in per};assert len(counts)==1
        k=counts.pop();correct=[sum(r['correct'][i] for r in per) for i in range(k)]
        index=max(range(k),key=lambda i:correct[i]) if arm=='edited' else 0
        report=dict(per_problem=per,selected_sample_index=index,correct_by_index=correct,
                    selection='RikiyaT: single highest-scoring global sample index per benchmark, ties lowest; baseline:index0.',
                    source=dict(repository='RikiyaT/lm_understand-logs',revision=revision,variant=variant,shards=n),
                    summary=dict(problems=len(per),samples=k,sampling=metadata[0]['_protocol']['sampling']),
                    limitation='External vLLM text-export evaluation; not a same-engine matched control.')
        (D/f'astra_scores_{arm}_{dataset}.json').write_text(json.dumps(report,indent=2)+'\n')
        (D/f'astra_samples_{arm}_{dataset}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
        entries={}
        for r in samples:
            size=len(T.encode(r['text'],add_special_tokens=False).ids)
            entries[r['id']]=dict(output_tokens_retokenized=size,sample_index=0)
        token_data['sets'][dataset][arm]=entries
        print(dataset,arm,'problems',len(per),'correct_by_index',correct,'selected_index',index,'first-text token coverage',len(entries),flush=True)
        sample_data[f'{dataset}/{arm}']=dict(index=index,correct_by_index=correct,problems=len(per))
temporary=tokens_path.with_suffix('.tmp');temporary.write_text(json.dumps(token_data,indent=2)+'\n');temporary.replace(tokens_path)
provenance=dict(revision=revision,files=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in downloaded],benchmarks=sample_data,
                ID_checks='Complete key coverage; GSM8K sample answers agree with local dataset; IFEval keys match all541 local keys.',
                token_rule='First-sample texts only. Token comparisons must be omitted if selected reference index is not0.')
(cache/'fetch-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
