"""State-dependent loop-token suppression folded ONLY into output-head rows.
All inputs are the 80 MATH-training fit problems and their unmodified captures.
"""
import collections,json,pathlib,re,numpy as np
from fit08 import gguf,SRC
from loop_repair_fit import token_pieces,loop_boundaries
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'loop-repair'

def main():
    r=gguf.GGUFReader(SRC);piece=token_pieces(r)
    meta=json.load(open(ROOT/'cap-fit-v2/index.json'));idx={e['id']:e for e in meta['traces']}
    rows={z['id']:z for z in map(json.loads,open(ROOT/'results/fit-traces.jsonl'))}
    train=(OUT/'fit_ids.txt').read_text().splitlines();cal=set((OUT/'calibration_ids.txt').read_text().splitlines());assert not set(train)&cal
    P=[];N=[];lp=collections.Counter();np_=collections.Counter();trace_frequency=collections.Counter();ln=nn=0
    for uid in train:
        row=rows[uid];assert uid.startswith('train/') and row['set']=='math_train'
        e=idx[uid];p=e['n_prompt'];g=row['gen_ids'];ci=g.index(248069) if 248069 in g else len(g)
        pieces=[piece(t) for t in g[:ci]]
        bounds=loop_boundaries(pieces)
        mask=np.zeros(e['n'],bool)
        for j in bounds:mask[max(p,p+int(j)-96):p+int(j)+1]=True
        eligible=np.arange(p,p+ci-1)
        positives=eligible[mask[eligible]];negatives=eligible[~mask[eligible]]
        X=np.memmap(ROOT/'cap-fit-v2'/e['file'],np.float16,mode='r',shape=(e['n'],1024))
        if len(positives):P.append(np.asarray(X[positives[np.linspace(0,len(positives)-1,min(256,len(positives)),dtype=int)]],np.float64))
        if len(negatives):N.append(np.asarray(X[negatives[np.linspace(0,len(negatives)-1,min(1024,len(negatives)),dtype=int)]],np.float64))
        nextids=np.asarray(g[1:ci],int)
        initial=[];tail=''
        for piece_text in pieces[:-1]:
            tail=(tail+piece_text).rsplit('\n',1)[-1][-256:]
            initial.append(not re.search('[A-Za-z0-9]',tail))
        initial=np.array(initial,bool)
        alpha=np.array([bool(re.fullmatch('[A-Za-z]{2,}',piece(int(tok)).strip())) for tok in nextids])
        lmask=mask[eligible]&initial&alpha;nmask=~mask[eligible]&initial&alpha
        for tok in nextids[lmask]:lp[int(tok)]+=1;ln+=1
        for tok in nextids[nmask]:np_[int(tok)]+=1;nn+=1
        trace_frequency.update(set(map(int,nextids[lmask])))
    P=np.concatenate(P);N=np.concatenate(N);mp,mn=P.mean(0),N.mean(0);c=N-mn;cov=c.T@c/len(c)
    cov=.8*cov+.2*np.trace(cov)/1024*np.eye(1024);w=np.linalg.solve(cov,mp-mn)
    gap=np.median(P@w)-np.median(N@w);assert gap>0;w/=gap
    np.save(OUT/'loop-all-token-probe.npy',w)
    choices=[]
    for tok,count in lp.items():
        word=piece(tok).strip()
        if not re.fullmatch('[A-Za-z]{2,}',word) or count<200 or trace_frequency[tok]<8:continue
        ratio=((count+1)/(ln+1))/((np_[tok]+1)/(nn+1))
        if ratio>=2:choices.append(dict(id=tok,piece=piece(tok),loop_count=count,productive_count=np_[tok],loop_trace_count=trace_frequency[tok],ratio=ratio))
    choices.sort(key=lambda z:(-z['ratio'],-z['loop_count']))
    chosen=choices[:8]
    report=dict(fit_ids=train,fit_problems=len(train),calibration_excluded=sorted(cal),
                classifier_positive_rows=len(P),classifier_negative_rows=len(N),
                positive_median=float(np.median(P@w)),negative_median=float(np.median(N@w)),
                selected_tokens=chosen,all_eligible_tokens=choices,
                policy='Top 8 line-initial alphabetic tokens with >=200 loop occurrences across >=8 fitting traces and >=2x relative frequency; chosen solely from fitting traces.')
    (OUT/'head-recovery-fit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return P,N,w

if __name__=='__main__':main()
