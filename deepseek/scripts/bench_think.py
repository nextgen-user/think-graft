#!/usr/bin/env python3
# Thinking-mode bench (Nov 14), bench_tough.py generalised: a problem set, a token budget, any binary / env / args, the
# same sampling as the tough bench (effort max, temperature 1.0, top_p 1.0, seed 7 + i).  Production paused and restored.
# Sets: tough (bench_tough's three) and held (six new problems, answers brute-forced in this file's comments' script,
# never used to calibrate anything: code tracing, combinatorics, probability, number theory, scheduling, algorithms).
# usage: bench_think.py TAG EXE ENV.json ARGS.json SET [BUDGET] [ID,ID..]   -> runs/think-results.jsonl, runs/think-progress.txt
import sys,os,json,time,re,hashlib,pathlib,subprocess,urllib.request
sys.argv,ARGS=[sys.argv[0]],sys.argv[1:]
import gen_eval as G
import bench_tough as BT
D=G.D
HELD=[
 dict(id='code_trace',expected='85',task='What does this Python function return for n = 40?\n\n```python\ndef f(n):\n    a, b, out = 0, 1, 0\n    for i in range(n):\n        a, b = b, (a + b) % 97\n        if (a * i) % 5 == 2:\n            out += a\n        elif b % 3 == 0:\n            out -= i\n    return out\n```'),
 dict(id='bitstrings',expected='853',task='How many binary strings of length 12 contain no three consecutive 1s and have an even number of 1s?'),
 dict(id='dice_nine',expected='21/2',task='A fair six-sided die is rolled repeatedly until two consecutive rolls sum to 9. What is the expected number of rolls? Give the answer as a reduced fraction.'),
 dict(id='numtheory',expected='228',task='How many integers n with 1 <= n <= 3000 satisfy both: n^2 + 1 is divisible by 13, and the sum of the decimal digits of n is odd?'),
 dict(id='makespan',expected='10',task='Seven tasks have durations A=3, B=2, C=4, D=2, E=3, F=1, G=2 (hours). Precedence: C needs A; D needs A and B; E needs B; F needs C and D; G needs E and F. Two workers are available; each task is done by one worker without interruption, and a worker does one task at a time. What is the minimum total time (makespan) to finish all tasks?'),
 dict(id='edit_distance',expected='8',task='What is the Levenshtein edit distance (insertions, deletions and substitutions each cost 1) between the strings "dynamicprogramming" and "dramaticprograms"?'),
]
COMPACT='\n\nSolve this efficiently: reason only as much as the problem needs, keep the working short, and do not re-check steps you are already sure of. End your response with a final line of the form:\nANSWER: <answer>'
def value(a):   # (Nov 14) the tough bench's parser plus LaTeX delimiters: \(\frac{2618}{59049}\) was graded wrong
    a=a.strip()
    for x in ('\\(','\\)','\\[','\\]','\\boxed{'): a=a.replace(x,'')
    v=BT.value(a)
    if isinstance(v,str):   # (a unit or words after a number: "10 hours" -> 10)
        m=re.match(r'\s*(-?\d+(?:/\d+)?)\b',a.replace('\\frac{','').replace('}{','/').replace('}',''))
        if m: v=BT.value(m.group(1))
    return v
def main():
    import signal; signal.signal(signal.SIGTERM,lambda *a: sys.exit(143))   # (a stopped queue still restores production: finally runs)
    tag,exe=ARGS[0],ARGS[1]; env=json.loads(pathlib.Path(ARGS[2]).read_text()) if ARGS[2].endswith('.json') else json.loads(ARGS[2])
    args=json.loads(pathlib.Path(ARGS[3]).read_text()) if ARGS[3].endswith('.json') else json.loads(ARGS[3])
    probs={'tough':BT.PROBLEMS,'held':HELD}[ARGS[4]]; budget=int(ARGS[5]) if len(ARGS)>5 else 32000
    if len(ARGS)>6: probs=[p for p in probs if p['id'] in ARGS[6].split(',')]
    out=D/'runs'/'think-results.jsonl'; prog=D/'runs'/'think-progress.txt'
    PROD_UP=subprocess.run(['systemctl','--user','is-active','-q','strata-deepseek']).returncode==0   # (Nov 15: production may be disabled - wait for / restore only what was running)
    for _ in range(120):
        if not PROD_UP: break
        try:
            with urllib.request.urlopen('http://127.0.0.1:8095/metrics',timeout=5) as f:
                if json.load(f)['live']['state']=='idle': break
        except Exception: pass
        time.sleep(5)
    sha=hashlib.sha256(pathlib.Path(G.live['exe']).read_bytes()).hexdigest()
    subprocess.run(['systemctl','--user','stop','strata-deepseek'],check=True)
    try:
        eng=G.Engine(tag,exe,env,args)
        try:
            for i,p in enumerate(probs):
                seed=7+[q['id'] for q in {'tough':BT.PROBLEMS,'held':HELD}[ARGS[4]]].index(p['id'])
                # (Nov 14 afternoon) BT_THINK=off: the template's no-thinking turn (the model reasons in its answer, if at all);
                # BT_PROMPT=compact: an instruction to solve efficiently instead of the bench's "check your work".  Defaults unchanged.
                suffix=COMPACT if os.environ.get('BT_PROMPT')=='compact' else BT.SUFFIX
                if os.environ.get('BT_THINK')=='off': text=G.TPL.render([dict(role='user',content=p['task']+suffix)],enable_thinking=False)
                else: text=G.TPL.render([dict(role='user',content=p['task']+suffix)],enable_thinking=True,reasoning_effort=os.environ.get('BT_EFFORT','max'))   # (BT_EFFORT: low 50, high 75, max 100, or 1..100)
                # (Nov 14 evening) BT_PREFILL: text placed right after <think>, in the model's own voice (the "terse line")
                pre=os.environ.get('BT_PREFILL','').replace('\\n','\n')
                if pre and not os.environ.get('BT_THINK')=='off': text=text+pre
                ids=G.TOK.encode(text,parse_special=True)
                eng.p.stdin.write(f'GEN {budget} temperature=1.0 top_p=1.0 seed={seed} '+','.join(map(str,ids))+'\n'); eng.p.stdin.flush()
                toks=[]; t0=time.monotonic()
                while True:
                    l=eng.line()
                    if l.startswith('T '):
                        toks.append(int(l.split()[1]))
                        if len(toks)%100==0:
                            el=time.monotonic()-t0
                            prog.write_text(f"{tag} {p['id']} ({i+1}/{len(probs)}): {len(toks)} tokens in {el:.0f} s ({len(toks)/el:.2f} tok/s)\n")
                    elif l.startswith('DONE '):
                        f=l.split(); info=dict(n=int(f[1]),prompt_ms=float(f[3]),gen_ms=float(f[4]),finish=f[5]); break
                    elif l.startswith('ERR'): raise RuntimeError(l)
                txt=G.TOK.decode([t for t in toks if t!=1])
                final=txt.split('</think>')[-1] if '</think>' in txt else ''
                m=re.findall(r'ANSWER:\s*(.+)',final) or re.findall(r'ANSWER:\s*(.+)',txt)
                ans=m[-1].strip() if m else None
                ok=ans is not None and value(ans)==value(p['expected'])
                think=len(G.TOK.encode(txt.split('</think>')[0],parse_special=True)) if '</think>' in txt else None
                row=dict(variant=tag,set=ARGS[4],id=p['id'],expected=p['expected'],answer=ans,correct=ok,thought_done='</think>' in txt,think_tokens=think,**info,env=env,text=txt,prefill=pre or None,effort=os.environ.get('BT_EFFORT','max'))
                with open(out,'a') as fo: fo.write(json.dumps(row)+'\n')
                print(json.dumps({k:row[k] for k in ('variant','id','expected','answer','correct','thought_done','think_tokens','n','gen_ms','finish')}),flush=True)
        finally: eng.close()
    finally:
        if PROD_UP: subprocess.run(['systemctl','--user','start','strata-deepseek'],check=True)
        assert hashlib.sha256(pathlib.Path(G.live['exe']).read_bytes()).hexdigest()==sha
        print('Production restored.' if PROD_UP else 'Production was off before the run: left off.',flush=True)
if __name__=='__main__': main()
