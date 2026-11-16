"""Graft2-style shrinkage LDA on MATH-train only; no gradient fitting.
Usage: python3 fit08.py CAPDIR traces.jsonl [BETA OUT.gguf]
Capture schema 2 stores actual-engine logits; calibration includes the sampler.
"""
import hashlib, json, os, pathlib, re, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(pathlib.Path(os.environ.get('LLAMA_CPP',str(ROOT/'llama.cpp')))/'gguf-py'))
import gguf
from gguf import quants
from legacy_labels import correct, norm, answer_of
from sampling08 import stop_probability
LABELS = os.environ.get('LABELS','graft2')
SRC = os.environ.get('SRC',str(ROOT/'models/Qwen3.5-0.8B-BF16.gguf'))


def answer_end(think, expected):
    """Conservative labels: a matching balanced box or an explicit answer phrase.
    Merely writing '= 0' or a bare number does not label the rest as re-checking.
    This detects a stated candidate, not a proof the answer was settled.
    """
    target=norm(expected); ends=[]
    for m in re.finditer(r'\\boxed\s*\{',think):
        depth=1; start=m.end()
        for k in range(start,len(think)):
            depth += (think[k]=='{')-(think[k]=='}')
            if depth==0:
                if norm(think[start:k])==target: ends.append(k+1)
                break
    raw=str(expected).strip().strip('$')
    pat=(r'\b(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*\$?\s*'
         +re.escape(raw)+r'(?![\w.])')
    ends += [m.end() for m in re.finditer(pat,think,re.I)]
    return min(ends) if ends else None


def graft2_answer_end(think, expected, text, closed):
    """Original Qwen graft2 heuristic, retained for a direct recipe replication.
    First answer-like occurrence for short references; literal occurrence for long
    references. These labels are intentionally audited: intermediate-value hits
    can be false positives, and answer occurrence is not answer correctness.
    """
    cands={expected,answer_of(text)} if closed else {expected}
    ends=[]
    for candidate in sorted(c for c in cands if c):
        if len(norm(candidate) or '')<=3:
            m=re.search(r'(=|is|answer|get|gives|total|equals)\s*\$?\\?\(?\s*'+re.escape(candidate)+r'(?![\d.])',think)
            if m: ends.append(m.end())
        else:
            k=think.find(candidate)
            if k>=0: ends.append(k+len(candidate))
    return min(ends) if ends else None


def load_traces(cap,tr,reader):
    meta=json.loads((cap/'index.json').read_text()); assert meta['schema']==2
    dim=meta['dim']; tokens=reader.fields['tokenizer.ggml.tokens'].contents(); close=tokens.index('</think>')
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256)); cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    byte={chr(c):b for b,c in zip(bs,cs)}
    def piece(t):
        try: return bytes(byte[c] for c in tokens[t]).decode('utf-8','replace')
        except KeyError: return tokens[t]
    rows={(r['id'],r.get('sample',0)):r for r in map(json.loads,open(tr))}
    traces=[]; audit=[]
    for i,e in enumerate(meta['traces']):
        r=rows[e['id'],e['sample']]; ids=r['gen_ids']; p=e['n_prompt']; count=e['n']
        allids=np.asarray(r['prompt_ids']+ids,np.int32)
        assert hashlib.sha256(allids.tobytes()).hexdigest()==e['token_sha256']
        ci=ids.index(close) if close in ids else len(ids)
        pieces=[piece(t) for t in ids]; think=''.join(pieces[:ci]); off=np.cumsum([0]+[len(s) for s in pieces])
        end=(graft2_answer_end(think,r['expected'],r['text'],ci<len(ids)) if LABELS=='graft2'
             else answer_end(think,r['expected']))
        # As in graft2, a completed wrong solution supplies no positive labels.
        if ci<len(ids) and not correct(r['text'],r['expected']): end=None
        answer=None if end is None else int(np.searchsorted(off,end,side='left')-1)
        lab=np.full(count,-1,np.int8); lab[p-1:p+ci]=0
        if answer is not None: lab[p+answer:p+ci]=1
        if ci<len(ids): lab[p+ci:]=2
        para=np.zeros(count,bool)
        para[p:]=['\n\n' in s for s in pieces]
        X=np.memmap(cap/e['file'],np.float16,mode='r',shape=(count,dim))
        stats=np.memmap(cap/e['stats'],np.float32,mode='r',shape=(count,21))
        traces.append(dict(id=e['id'],fold=i%2,X=X,stats=stats,lab=lab,para=para,answer=answer))
        audit.append(dict(id=e['id'],fold=i%2,expected=r['expected'],answer_token=answer,
                          positive_rows=int(((lab==1)&para).sum()),closed=ci<len(ids),
                          label_excerpt=None if end is None else think[max(0,end-180):end+80]))
    (cap/f'label-audit-{LABELS}.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(f'label policy {LABELS}: {len(traces)} traces, {sum(t["answer"] is not None for t in traces)} with a stated reference answer; '
          f'{sum(int(((t["lab"]==1)&t["para"]).sum()) for t in traces)} positive rows',flush=True)
    # Original graft2 fits only traces with a detected reference answer.
    if LABELS=='graft2': traces=[t for t in traces if t['answer'] is not None]
    return traces,dim,close


def lda(traces,dim,shrink=.2):
    # Streaming sufficient statistics avoid >10 GB float64 concatenations.
    ns=ps=0; nsum=np.zeros(dim); psum=np.zeros(dim); cross=np.zeros((dim,dim))
    for t in traces:
        pm=(t['lab']==1)&t['para']; nm=((t['lab']==0)|(t['lab']==1))&~pm
        pos=np.asarray(t['X'][pm],np.float64); neg=np.asarray(t['X'][nm][::3],np.float64)
        ps+=len(pos); ns+=len(neg); psum+=pos.sum(0); nsum+=neg.sum(0); cross+=neg.T@neg
    if ps<2 or ns<dim: raise ValueError(f'Insufficient labels: {ps} positive, {ns} negative')
    mp,mn=psum/ps,nsum/ns; cov=cross/ns-np.outer(mn,mn)
    cov=(1-shrink)*cov+shrink*np.trace(cov)/dim*np.eye(dim)
    w=np.linalg.solve(cov,mp-mn)
    # Same scaling as graft2; no hidden intercept or extra parameter is introduced.
    P=np.concatenate([np.asarray(t['X'][(t['lab']==1)&t['para']],np.float64)@w for t in traces])
    N=np.concatenate([np.asarray(t['X'][((t['lab']==0)|(t['lab']==1))&~((t['lab']==1)&t['para'])][::3],np.float64)@w for t in traces])
    gap=np.median(P)-np.median(N)
    if gap<=0: raise ValueError('Nonpositive median separation')
    return w/gap,ps,ns


def report(traces,g,beta):
    risks=[]; kept=base=0.; stray=0.; nanswers=0
    for t in traces:
        score=np.asarray(t['X'],np.float32)@g.astype(np.float32)
        prob=stop_probability(t['stats'],beta*score)
        if t['answer'] is not None:
            sv=t['lab']==0; vf=t['lab']==1
            risks.append(float(-np.expm1(np.log1p(-np.clip(prob[sv],0,1-1e-12)).sum())))
            pv=prob[vf]; surv=np.exp(np.r_[0.,np.cumsum(np.log1p(-np.clip(pv,0,1-1e-12)))[:-1]])
            # Expected retained re-checking tokens, conditional on reaching its start.
            kept+=surv.sum(); base+=len(pv)
        if (t['lab']==2).any(): stray+=prob[t['lab']==2].sum(); nanswers+=1
    if not risks: raise ValueError('No answer-located traces in validation fold')
    return dict(max=max(risks),mean=float(np.mean(risks)),kept=kept/max(base,1),
                rechecking_tokens=int(base),stray=stray/max(nanswers,1),risk_traces=len(risks))


def write_model(reader,g,beta,dst,close):
    assert not any(t.name=='output.weight' for t in reader.tensors), 'Expected tied-head base'
    te=next(t for t in reader.tensors if t.name=='token_embd.weight'); qt=te.tensor_type
    raw=np.array(te.data,copy=True)
    orig=quants.dequantize(raw[close:close+1],qt)[0].astype(np.float32)
    wanted=orig+beta*g
    raw[close]=quants.quantize(wanted.astype(np.float32)[None,:],qt).reshape(-1).view(raw.dtype)
    eff=quants.dequantize(raw[close:close+1],qt)[0]-orig
    rel=float(np.linalg.norm(eff-beta*g)/np.linalg.norm(beta*g))
    print(f'{qt.name}: relative edit rounding error {rel:.6f}',flush=True)
    tmp=str(dst)+'.partial'
    wr=gguf.GGUFWriter(tmp,arch=reader.fields['general.architecture'].contents(),endianess=reader.endianess)
    for f in reader.fields.values():
        if f.name in (gguf.Keys.General.ARCHITECTURE, 'general.graft') or f.name.startswith('GGUF.'): continue
        vt=f.types[0]; st=f.types[-1] if vt==gguf.GGUFValueType.ARRAY else None
        wr.add_key_value(f.name,f.contents(),vt,sub_type=st)
    if 'general.graft' in reader.fields:
        wr.add_key_value('general.graft.backbone',reader.fields['general.graft'].contents(),gguf.GGUFValueType.STRING)
    wr.add_key_value('general.graft',f'one-row edit: </think> += {beta} * shrinkage-LDA; MATH-train only',gguf.GGUFValueType.STRING)
    for t in reader.tensors: wr.add_tensor_info(t.name,t.data.shape,t.data.dtype,t.data.nbytes,t.tensor_type)
    wr.add_tensor_info('output.weight',raw.shape,raw.dtype,raw.nbytes,qt)
    wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_ti_data_to_file()
    for t in reader.tensors: wr.write_tensor_data(t.data,tensor_endianess=reader.endianess)
    wr.write_tensor_data(raw,tensor_endianess=reader.endianess); wr.close(); os.replace(tmp,dst)
    np.save(str(dst.with_suffix(''))+'-probe.npy',g)
    np.save(str(dst.with_suffix(''))+'-effective-delta.npy',eff)
    print('wrote',dst,flush=True)


def main():
    cap=pathlib.Path(sys.argv[1]); tr=sys.argv[2]; reader=gguf.GGUFReader(SRC)
    traces,dim,close=load_traces(cap,tr,reader)
    if len(sys.argv)==3:
        sweep=[]
        for fold in (0,1):
            g,nP,nN=lda([t for t in traces if t['fold']==fold],dim)
            print(f'== fit {"even" if fold==0 else "odd"} / test {"odd" if fold==0 else "even"}: {nP} positives, {nN} negatives',flush=True)
            test=[t for t in traces if t['fold']!=fold]
            for beta in (0,4,6,8,10,12,14,16,20):
                r=report(test,g,beta); sweep.append(dict(fit_fold=fold,beta=beta,**r))
                print(f'  beta {beta:2}: held-out P(close while solving) max {r["max"]:.6f} mean {r["mean"]:.6f} | re-checking kept {r["kept"]:.2%} of {r["rechecking_tokens"]} tokens | stray </think> per answer {r["stray"]:.4f}',flush=True)
        (cap/f'sweep-{LABELS}.json').write_text(json.dumps(sweep,indent=2)+'\n')
    elif len(sys.argv)==5:
        g,nP,nN=lda(traces,dim); write_model(reader,g,float(sys.argv[3]),pathlib.Path(sys.argv[4]),close)
    else: raise SystemExit('Usage: fit08.py CAPDIR traces.jsonl [BETA OUT.gguf]')

if __name__=='__main__': main()
