"""Loop-aware graft fitting, exclusively from the 96 MATH-training traces.
No benchmark data are read. Uses existing base captures, no extra model passes.
"""
import collections,json,pathlib,sys
import numpy as np
from fit08 import gguf,SRC,load_traces,lda
from sampling08 import stop_probability
ROOT=pathlib.Path(__file__).resolve().parent
DEST=ROOT/'loop-repair';DEST.mkdir(exist_ok=True)

def token_pieces(reader):
    tok=reader.fields['tokenizer.ggml.tokens'].contents()
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
    for b in range(256):
        if b not in bs:bs.append(b);cs.append(256+n);n+=1
    byte={chr(c):b for b,c in zip(bs,cs)}
    def piece(i):
        try:return bytes(byte[c] for c in tok[i]).decode('utf-8','replace')
        except KeyError:return tok[i]
    return piece

def loop_boundaries(pieces):
    text=''.join(pieces);off=np.cumsum([0]+[len(s) for s in pieces]);recent=collections.deque(maxlen=12)
    p=0;bounds=[]
    for line in text.splitlines(keepends=True):
        start=p;p+=len(line);normal=' '.join(line.split())
        if not normal:continue
        recent.append(normal)
        if len(recent)<6:continue
        counts=collections.Counter(recent)
        if counts[normal]>=3 and len(counts)/len(recent)<=.5:
            j=int(np.searchsorted(off,p,side='left')-1)
            if j>=64:bounds.append(j)
    return np.array(sorted(set(bounds)),dtype=np.int64)

def fit_loop(records,dim):
    P=[];N=[]
    for t in records:
        if len(t['loop_rows'])==0:continue
        pm=t['loop_rows']
        # Equal per-trace caps prevent a 32k frozen loop from outweighing shorter examples.
        pm=pm[np.linspace(0,len(pm)-1,min(128,len(pm)),dtype=int)]
        mask=(t['lab']>=0)&(t['lab']<3)
        # All rows of confirmed repetition are excluded from the productive negatives.
        mask[t['loop_mask']]=False
        nm=np.flatnonzero(mask)
        if len(nm):nm=nm[np.linspace(0,len(nm)-1,min(1024,len(nm)),dtype=int)]
        P.append(np.asarray(t['X'][pm],np.float64));N.append(np.asarray(t['X'][nm],np.float64))
    P=np.concatenate(P);N=np.concatenate(N)
    mp,mn=P.mean(0),N.mean(0);c=N-mn;cov=c.T@c/len(c)
    cov=.8*cov+.2*np.trace(cov)/dim*np.eye(dim)
    g=np.linalg.solve(cov,mp-mn)
    gap=np.median(P@g)-np.median(N@g)
    assert gap>0
    g/=gap
    print('Loop LDA:',len(P),'positive',len(N),'negative rows',flush=True)
    return g

def main():
    reader=gguf.GGUFReader(SRC)
    # Use original labeler, but restore every training trace, including no-answer loops.
    import fit08
    fit08.LABELS='conservative'
    ts,dim,close=load_traces(ROOT/'cap-fit-v2',ROOT/'results/fit-traces.jsonl',reader)
    rows={r['id']:r for r in map(json.loads,open(ROOT/'results/fit-traces.jsonl'))}
    order=(ROOT/'data/fit_ids.txt').read_text().splitlines();cal_ids=set(order[::6]);piece=token_pieces(reader)
    audit=[]
    # Re-load original answer-located labels for the re-checking component.
    fit08.LABELS='graft2'
    old,_,_=load_traces(ROOT/'cap-fit-v2',ROOT/'results/fit-traces.jsonl',reader)
    orig={t['id']:t for t in old}
    for t in ts:
        r=rows[t['id']];assert r['id'].startswith('train/') and r['set']=='math_train'
        ids=r['gen_ids'];ci=ids.index(close) if close in ids else len(ids);p=len(r['prompt_ids'])
        pieces=[piece(i) for i in ids]
        bounds=loop_boundaries(pieces[:ci]);t['loop_rows']=p+bounds
        # Include the entire current repeated line as looping state; exclude it from negatives.
        mask=np.zeros(len(t['X']),bool)
        for j in bounds:mask[max(p,p+j-96):p+j+1]=True
        t['loop_mask']=mask;t['answer']=orig[t['id']]['answer'] if t['id'] in orig else None
        if t['id'] in orig:t['lab']=orig[t['id']]['lab'];t['para']=orig[t['id']]['para']
        audit.append(dict(id=t['id'],split='calibration' if t['id'] in cal_ids else 'fit',loop_boundaries=len(bounds),
                          first_loop_token=int(bounds[0]) if len(bounds) else None,answer_token=t['answer']))
    (DEST/'labels.json').write_text(json.dumps(audit,indent=2)+'\n')
    train=[t for t in ts if t['id'] not in cal_ids];cal=[t for t in ts if t['id'] in cal_ids]
    gdone,_,_=lda([t for t in train if t['id'] in orig],dim)
    gloop=fit_loop(train,dim)
    np.save(DEST/'done-probe.npy',gdone);np.save(DEST/'loop-probe.npy',gloop)
    result=[]
    for alpha in (0,4,8,12,16,20,24,28,32):
        delta=12*gdone+alpha*gloop
        early=[];old_early=[];loopclose=[];stray=[]
        for t in cal:
            score=np.asarray(t['X'],np.float32)@delta.astype(np.float32)
            new=stop_probability(t['stats'],score)
            base=stop_probability(t['stats'],np.asarray(t['X'],np.float32)@(12*gdone).astype(np.float32))
            p=len(rows[t['id']]['prompt_ids'])
            horizon=min([x for x in [t['answer'],int(t['loop_rows'][0]-p) if len(t['loop_rows']) else None] if x is not None],default=len(rows[t['id']]['gen_ids']))
            end=p+horizon
            def risk(a):return float(-np.expm1(np.log1p(-np.clip(a,0,1-1e-12)).sum()))
            early.append(risk(new[p-1:end]));old_early.append(risk(base[p-1:end]))
            if len(t['loop_rows']):
                start=t['loop_rows'][0];loopclose.append(risk(new[start:min(len(new),start+512)]))
            if (t['lab']==2).any():stray.append(float(new[t['lab']==2].sum()))
        rec=dict(loop_strength=alpha,mean_early=float(np.mean(early)),max_early=max(early),
                 mean_extra_early=float(np.mean(np.maximum(np.array(early)-old_early,0))),
                 max_extra_early=float(np.max(np.maximum(np.array(early)-old_early,0))),
                 loop_close_by_512_mean=float(np.mean(loopclose)) if loopclose else None,
                 loop_close_by_512_min=min(loopclose) if loopclose else None,stray_expected_mean=float(np.mean(stray)) if stray else 0)
        result.append(rec);print(json.dumps(rec),flush=True)
    (DEST/'sweep.json').write_text(json.dumps(result,indent=2)+'\n')
    (DEST/'calibration_ids.txt').write_text('\n'.join(u for u in order if u in cal_ids)+'\n')
    (DEST/'fit_ids.txt').write_text('\n'.join(u for u in order if u not in cal_ids)+'\n')
    print('TRAIN_ONLY_FIT_DONE',len(train),'fit',len(cal),'calibration',sum(bool(len(t['loop_rows'])) for t in train),'looping fit traces',flush=True)

if __name__=='__main__':main()
