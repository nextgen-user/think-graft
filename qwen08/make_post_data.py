"""Build recorded-trace playback assets from frozen outputs, with no inference."""
import codecs,gzip,json,os,pathlib
import numpy as np
from tokenizers import Tokenizer
from score08 import math_correct
ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=pathlib.Path(os.environ.get('STUDY_SOURCE',str(ROOT)))
WEB=ROOT.parent/'blog/2/data';WEB.mkdir(parents=True,exist_ok=True)
T=Tokenizer.from_file(os.environ['TOKENIZER_JSON'])
def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    return [json.loads(line) for line in opener(path,'rt')]
def available(path):return path if path.exists() else pathlib.Path(str(path)+'.gz')
ours={r['id']:r for r in rows(available(SOURCE/'results/math500-edited.jsonl'))}
base={r['id']:r for r in rows(SOURCE/'data/astra_samples_base_math500.jsonl')}
edited={r['id']:r for r in rows(SOURCE/'data/astra_samples_edited_math500.jsonl')}
problems={r['unique_id']:r for r in rows(SOURCE/'data/math500.jsonl')}
bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
for b in range(256):
    if b not in bs:bs.append(b);cs.append(256+n);n+=1
byte={chr(c):b for b,c in zip(bs,cs)}
def stream(ids):
    decoder=codecs.getincrementaldecoder('utf-8')('replace');parts=[];ends=[];size=0
    for token in ids:
        piece=T.id_to_token(token)
        try:raw=bytes(byte[c] for c in piece)
        except KeyError:raw=piece.encode()
        text=decoder.decode(raw);parts.append(text);size+=len(text.encode('utf-16-le'))//2;ends.append(size)
    tail=decoder.decode(b'',final=True)
    if tail:parts.append(tail);ends[-1]+=len(tail.encode('utf-16-le'))//2
    return ''.join(parts),ends
def pack(row,native=False):
    ids=row['gen_ids'] if native else T.encode(row['text'],add_special_tokens=False).ids
    text,ends=stream(ids);close=ids.index(248069) if 248069 in ids else None
    good,_=math_correct(row['text'],row.get('expected',row.get('answer')))
    return dict(text=text,ends=ends,tokens=len(ids),correct=bool(good),close=close,
                native_tokens=native,extra_close_tags=max(0,ids.count(248069)-1),
                capped=row.get('stop_type')=='limit' if native else None)
choices=[('test/number_theory/572.json','Shorter successful run'),
         ('test/algebra/1349.json','Only our sample gets the answer'),
         ('test/intermediate_algebra/1994.json','A failure: the reference finishes'),
         ('test/prealgebra/1622.json','A failure: all samples miss')]
index=[]
for i,(uid,label) in enumerate(choices):
    item=dict(id=uid,label=label,problem=problems[uid]['problem'],answer=problems[uid]['answer'],
              trajectories=dict(base=pack(base[uid]),reference=pack(edited[uid]),ours=pack(ours[uid],True)),
              note='Recorded outputs, replayed by token position. Reference text retokenized with the pinned tokenizer; reference counts are approximate. Different serving engines, not identical random draws.')
    name=f'trace-{i}.json';(WEB/name).write_text(json.dumps(item,separators=(',',':')))
    index.append(dict(id=uid,label=label,file=name))
(WEB/'traces.json').write_text(json.dumps(index,indent=2)+'\n')
study=json.load(open(SOURCE/'publication/math500-analysis.json'))
study['environment']=json.load(open(SOURCE/'publication/environment.json'))
study['ngram_diagnostic']=json.load(open(SOURCE/'publication/ngram8-diagnostic.json'))['groups']
study['model']=json.load(open(ROOT/'publication/model-manifest.json'))
(WEB/'study.json').write_text(json.dumps(study,indent=2)+'\n')
fit=next(r for r in rows(available(SOURCE/'results/fit-traces.jsonl')) if r['id']=='train/2006')
fitproblem=json.load(open(SOURCE/'data/math_train.json'))[2006]
(WEB/'fitting-example.json').write_text(json.dumps(dict(id='train/2006',problem=fitproblem['problem'],answer='300',base_tokens=fit['n_gen'],excerpt=fit['text'][:1800],role='Fitting example, not a held-out evaluation problem.'),indent=2)+'\n')
print('Built',len(index),'recorded comparisons and study data')
