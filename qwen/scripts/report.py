# Summary table for the Qwen maths arms. usage: python3 report.py out1.jsonl [out2.jsonl ...] (same problem ids compared)
import json, sys, numpy as np
from grade import correct
CLOSE = 248069
A = {}
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l); A.setdefault(r['arm'], {})[r['id']] = r
common = set.intersection(*[set(v) for v in A.values()])
print(f'{len(common)} problems in common\n')
print(f"{'arm':14s} {'right':>7s} {'think tok':>10s} {'answer tok':>10s} {'total tok':>10s} {'median':>7s} {'unclosed':>8s} {'extra </think>':>14s} {'s/problem':>9s}")
for arm, rs in A.items():
    R = [rs[i] for i in sorted(common)]
    ok = [correct(r['text'], r['expected']) for r in R]
    th = [r['gen_ids'].index(CLOSE) if CLOSE in r['gen_ids'] else len(r['gen_ids']) for r in R] if R[0]['thinking'] else [0]*len(R)
    tot = [len(r['gen_ids']) for r in R]; ans = [t - h for t, h in zip(tot, th)]
    unc = sum(1 for r in R if r['thinking'] and CLOSE not in r['gen_ids'])
    extra = sum(max(0, r['gen_ids'].count(CLOSE) - 1) for r in R)
    print(f"{arm:14s} {sum(ok):3d}/{len(R):<3d} {np.mean(th):10.0f} {np.mean(ans):10.0f} {np.mean(tot):10.0f} {np.median(tot):7.0f} {unc:8d} {extra:14d} {np.mean([r['seconds'] for r in R]):9.1f}")
if len(A) > 1:
    arms = list(A)
    for a in arms[1:]:
        fl = [i for i in sorted(common) if correct(A[arms[0]][i]['text'], A[arms[0]][i]['expected']) != correct(A[a][i]['text'], A[a][i]['expected'])]
        print(f'\nflips {arms[0]} vs {a}: ' + ', '.join(f"{i} ({'+' if correct(A[a][i]['text'], A[a][i]['expected']) else '-'})" for i in fl))
