# Solve vs check split of every maths run (Nov 14): thinking tokens before the first appearance of the correct answer
# ("solve"), after it ("check"), and the visible answer.  The effort dial should move both; the graft's beta only "check".
# usage: phase_split.py  -> prints a table, writes delete/phase-split.json
import json, sys, pathlib
D = pathlib.Path(__file__).resolve().parent.parent; R = pathlib.Path('~/mycodesandexp/Strata-DeepSeek')
sys.path.insert(0, str(R / 'tools/dsv41')); sys.path.insert(0, str(R / 'tools'))
from smoke import tokenizer
TOK = tokenizer(pathlib.Path('~/models/DeepSeek-V4.1-Flash-IQ2_XXS/strata-pack'))
_ns = {}; exec(open(D / 'bench-agentic/cmp.py').read().split("def maths():")[0], _ns); right = _ns['right']   # (its own D)
rows = [json.loads(l) for l in open(D / 'runs/think-results.jsonl')]
for l in open(D / 'runs/tough-results.jsonl'):
    r = json.loads(l); r['variant'] = 'base-tough'; r['set'] = 'tough'; rows.append(r)
ARM = {'base': 'plain max', 'graft2': 'graft2 max', 'graft4': 'v4 max', 'stack': 'v4d max kf45', 'g5-max': 'v5 max kf45', 'eff-high': 'plain high kf45', 'eff-low': 'plain low kf45',
       'v5k43': 'v5 max', 'lowk43': 'plain low', 'maxk43': 'plain max'}
EXP_STR = {'digit_sum_double': ['791'], 'round_table_eyes': ['2618/59049', '2618}{59049'], 'coin_pattern_race': ['5/12', '5}{12'], 'code_trace': ['85'],
           'bitstrings': ['853'], 'dice_nine': ['21/2', '21}{2', '10.5'], 'makespan': ['= 10', 'is 10', 'makespan 10', ' 10 hours', '10 hours']}
import re
STATED = {'code_trace': re.compile(r'(?i)(return[s]?|answer|result|final|out\s*=|output)[^\n]{0,25}\b85\b|\b85\b[^\n]{0,15}(final|answer)'),
          'makespan': re.compile(r'(?i)(makespan|answer|minimum|final|total)[^\n]{0,30}\b10\b|\b10\b[^\n]{0,20}(hours?|final|answer)')}
out = []
for r in rows:
    v = r['variant'].rsplit('-', 1)[0]
    if v not in ARM or r['id'] not in EXP_STR: continue
    t = r['text']; closed = '</think>' in t; th = t.split('</think>')[0] if closed else t
    if r['id'] in STATED:   # (code tracing / scheduling: the number also shows up as an intermediate value; use where it is stated as the result)
        mm = STATED[r['id']].search(th); k = mm.start() if mm else -1
    else:
        ks = [th.find(e) for e in EXP_STR[r['id']] if th.find(e) >= 0]; k = min(ks) if ks else -1
    nth = len(TOK.encode(th, parse_special=True)); solve = len(TOK.encode(th[:k], parse_special=True)) if k >= 0 else None
    out.append(dict(arm=ARM[v], id=r['id'], right=right(r), closed=closed, think=nth, solve=solve, check=(nth - solve) if solve is not None else None,
                    answer=r['n'] - nth if closed else 0, total=r['n'], minutes=round(r['gen_ms'] / 60000, 1)))
json.dump(out, open(D / 'delete/phase-split.json', 'w'), indent=1)
print(f'{"arm":11s} {"problem":18s} ok  solve  check  answer  total')
for o in sorted(out, key=lambda o: (o['id'], o['arm'])):
    print(f'{o["arm"]:11s} {o["id"][:18]:18s} {"✓" if o["right"] else "✗"} {str(o["solve"] or "-"):>6s} {str(o["check"] if o["check"] is not None else "-"):>6s} {o["answer"]:6d} {o["total"]:6d}')
