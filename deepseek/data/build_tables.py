# Rebuilds every table in blog-data/tables from the raw logs (Nov 15 2026). Run with the Strata venv python (needs the
# DeepSeek tokenizer):  ~/mycodesandexp/Strata-DeepSeek/.venv/bin/python blog-data/build_tables.py
# Sources (all read-only): adaptive-experts/runs/think-results.jsonl + runs/tough-results.jsonl (maths),
# adaptive-experts/bench-agentic/results.jsonl (agentic), dsh session transcripts in
# ~/.local/share/deepseek-harness/state/sessions/*bench-<tag>-<task>*/ (per-step thinking), ~/deepseek-work/bench/quality/*.json
# (headless-browser checks of the site task).
import json, sys, csv, re, glob, os, subprocess, pathlib, statistics
B = pathlib.Path(__file__).resolve().parent; D = B.parent / 'adaptive-experts'; T = B / 'tables'
R = pathlib.Path('~/mycodesandexp/Strata-DeepSeek'); sys.path.insert(0, str(R / 'tools/dsv41')); sys.path.insert(0, str(R / 'tools'))
from smoke import tokenizer
TOK = tokenizer(pathlib.Path('~/models/DeepSeek-V4.1-Flash-IQ2_XXS/strata-pack'))
def ntok(s): return len(TOK.encode(s, parse_special=True)) if s else 0
def wcsv(name, rows):
    if not rows: return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(T / name, 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    print('wrote', name, len(rows), 'rows')
# ------------------------------------------------------------------ maths
ARMS_M = {   # variant prefix -> (arm, engine, effort, graft)
    'base-held': ('plain max', 'kf43', 'max (100)', 'none'), 'gf-tough': ('plain max', 'kf14 (older engine, Nov 8)', 'max (100)', 'none'),
    'graft2': ('graft2', 'kf43', 'max (100)', 'graft2 (LDA probe, beta 14)'), 'graft4': ('graft v4', 'kf43', 'max (100)', 'v4'),
    'stack': ('graft v4d', 'kf45 (lossy deferral)', 'max (100)', 'v4d'), 'g5-max': ('graft v5', 'kf45 (lossy deferral)', 'max (100)', 'v5'),
    'v5k43': ('graft v5', 'kf43', 'max (100)', 'v5 (graft5w8)'), 'lowk43': ('plain low', 'kf43', 'low (50)', 'none')}
src = open(D / 'delete/phase_split.py').read()
EXP_STR = eval(src.split('EXP_STR = ')[1].split('\nimport re')[0]); exec('import re\nSTATED = ' + src.split('STATED = ')[1].split('\nout = []')[0])
_ns = {}; exec(open(D / 'bench-agentic/cmp.py').read().split("def maths():")[0], _ns); right = _ns['right']
rows = [json.loads(l) for l in open(D / 'runs/think-results.jsonl')]
for l in open(D / 'runs/tough-results.jsonl'):
    r = json.loads(l); r['variant'] = 'gf-tough'; r['set'] = 'tough'; rows.append(r)
M = []
for r in rows:
    v = r['variant']; key = next((k for k in ARMS_M if v.startswith(k)), None)
    if key is None: continue
    arm, eng, eff, gr = ARMS_M[key]; t = r['text']; closed = '</think>' in t; th = t.split('</think>')[0] if closed else t
    nth = ntok(th); solve = None
    if r['id'] in EXP_STR:
        if r['id'] in STATED: mm = STATED[r['id']].search(th); k = mm.start() if mm else -1
        else: ks = [th.find(e) for e in EXP_STR[r['id']] if th.find(e) >= 0]; k = min(ks) if ks else -1
        solve = ntok(th[:k]) if k >= 0 else None
    M.append(dict(variant=v, arm=arm, engine=eng, effort=eff, graft=gr, set=r.get('set'), problem=r['id'], expected=r.get('expected'),
                  answer=r.get('answer'), correct=bool(right(r)), closed_thinking=closed, total_tokens=r['n'], thinking_tokens=nth,
                  solve_tokens=solve, check_tokens=(nth - solve) if solve is not None else None, answer_tokens=(r['n'] - nth) if closed else 0,
                  stray_close_tags=max(0, t.count('</think>') - 1), minutes=round(r['gen_ms'] / 60000, 1), finish=r.get('finish'),
                  budget=32000 if (r.get('set') == 'tough') else 16000))
wcsv('maths_runs.csv', M)
# arm summary over the 6 problems every kf43 arm solved + round_table separately
SHARED = ['digit_sum_double', 'coin_pattern_race', 'code_trace', 'bitstrings', 'dice_nine', 'makespan']
S = []
for arm, eng in (('plain max', None), ('plain low', 'kf43'), ('graft2', 'kf43'), ('graft v5', 'kf43')):
    rr = {}
    for m in M:
        if m['arm'] != arm or (eng and m['engine'] != eng) or (arm == 'graft v5' and m['engine'] != 'kf43'): continue
        if arm == 'plain max' and m['engine'].startswith('kf45'): continue
        rr.setdefault(m['problem'], m)        # first run of each problem
    have = [p for p in SHARED if p in rr]
    rt = rr.get('round_table_eyes')
    S.append(dict(arm=arm, engines=';'.join(sorted(set(rr[p]['engine'] for p in have))), problems=len(have), correct=sum(rr[p]['correct'] for p in have),
                  total_tokens=sum(rr[p]['total_tokens'] for p in have), thinking_tokens=sum(rr[p]['thinking_tokens'] for p in have),
                  solve_tokens=sum((rr[p]['solve_tokens'] or 0) for p in have), check_tokens=sum((rr[p]['check_tokens'] or 0) for p in have),
                  answer_tokens=sum(rr[p]['answer_tokens'] for p in have), minutes=round(sum(rr[p]['minutes'] for p in have), 1),
                  round_table=('right' if rt and rt['correct'] else ('no answer (32k cap)' if rt else 'not run')), round_table_tokens=rt['total_tokens'] if rt else None))
wcsv('maths_summary_6_shared_problems.csv', S)
# ------------------------------------------------------------------ agentic
SS = pathlib.Path('~/.local/share/deepseek-harness/state/sessions')
ARMS_A = {'before': ('original, dsh default (medium)', 'kf43 (production, Nov 13)'), 'after-off': ('thinking off', 'kf43 (production, Nov 13)'),
          'ag-base': ('original, dsh medium', 'kf43'), 'ag-graft': ('graft2, dsh medium', 'kf43'), 'ag5-base': ('original, dsh medium', 'kf45 (lossy deferral)'),
          'ag5-off': ('thinking off', 'kf45 (lossy deferral)'), 'ag5-v5': ('graft v5, dsh medium', 'kf45 (lossy deferral)'), 'ag5-v4': ('graft v4, dsh medium', 'kf45 (lossy deferral)'),
          'k43-g7': ('graft v7, dsh medium', 'kf43'), 'k43-g2': ('graft2, dsh medium', 'kf43'), 'k43-low': ('original, dsh LOW', 'kf43'), 'after-kf45': ('(infrastructure failure, ignore)', 'kf45')}
def session_steps(tag, task):
    fs = glob.glob(f'{SS}/*bench-{tag}-{task}--/*/session.v4.jsonl.zstd')
    if not fs: return None
    f = max(fs, key=os.path.getsize); raw = subprocess.run(['zstd', '-dc', f], capture_output=True).stdout.decode(errors='replace')
    out, cur, prev = [], None, None
    for l in raw.splitlines():
        if not l.strip(): continue
        x = json.loads(l)
        if x['type'] == 'tool/result':
            c = x['data']['message']['content']; txt = ''.join(p.get('text', '') for p in c if isinstance(p, dict))
            prev = (bool(x['data']['message'].get('isError')), re.sub(r'\s+', ' ', txt)[-160:])
        elif x['type'] == 'assistant/message':
            c = x['data']['message']['content']; th = ''.join(p.get('text', '') for p in c if p['type'] == 'reasoning')
            cur = dict(step=x['data']['step'], thinking_tokens=ntok(th), capped='I have thought about this long enough' in th, tool_calls='',
                       previous_tool_result_error=prev[0] if prev else None, previous_tool_result_tail=prev[1] if prev else '',
                       thinking_start=re.sub(r'\s+', ' ', th[:200])); out.append(cur)
        elif x['type'] == 'tool/call' and cur is not None:
            cur['tool_calls'] += (';' if cur['tool_calls'] else '') + f"{x['data']['name']}({ntok(x['data']['arguments'])} tok)"
    return out
A, AS, seen = [], [], {}
for r in [json.loads(l) for l in open(D / 'bench-agentic/results.jsonl')]:
    arm, eng = ARMS_A.get(r['tag'], (r['tag'], '?')); seen[(r['tag'], r['task'])] = seen.get((r['tag'], r['task']), 0) + 1
    st = session_steps(r['tag'], r['task']) if seen[(r['tag'], r['task'])] == 1 else None   # (sessions are per folder: only the last run's survives)
    row = dict(tag=r['tag'], run=seen[(r['tag'], r['task'])], task=r['task'], arm=arm, engine=eng, dsh_patch=r.get('patch'), check=r['check'], ok=r['ok'],
               wall_s=r['wall_s'], output_tokens=r.get('outputTokens'), uncached_input_tokens=r.get('uncachedInputTokens'), cache_read_tokens=r.get('cacheReadTokens'))
    if st:
        row.update(steps=len(st), thinking_tokens=sum(s['thinking_tokens'] for s in st), capped_steps=sum(s['capped'] for s in st),
                   zero_thinking_steps=sum(1 for s in st if s['thinking_tokens'] == 0), error_results_seen=sum(1 for s in st if s['previous_tool_result_error']))
        for s in st: AS.append(dict(tag=r['tag'], task=r['task'], arm=arm, engine=eng, **s))
    A.append(row)
wcsv('agentic_runs.csv', A); wcsv('agentic_steps.csv', AS)
# ------------------------------------------------------------------ site browser checks
Q = []
for f in sorted(glob.glob('~/deepseek-work/bench/quality/*-site.json')):
    q = json.load(open(f)); tag = pathlib.Path(f).name.replace('-site.json', '')
    Q.append(dict(tag=tag, arm=ARMS_A.get(tag, (tag,))[0], console_errors=len(q.get('errors', [])) if isinstance(q.get('errors'), list) else q.get('errors'),
                  dark_mode_works=q.get('dark', {}).get('changes'), dark_mode_persists=q.get('dark', {}).get('persists'),
                  invalid_email_inline_error=q.get('form', {}).get('inline_error'), valid_submit_feedback=q.get('form', {}).get('valid_feedback'),
                  mobile_no_horizontal_scroll=q.get('mobile', {}).get('no_hscroll'), mobile_scroll_width_px=q.get('mobile', {}).get('scroll_width'),
                  menu_prices=q.get('static', {}).get('prices'), labelled_inputs=f"{q.get('static', {}).get('inputs_labelled')}/{q.get('static', {}).get('inputs')}",
                  words=q.get('static', {}).get('words')))
wcsv('site_browser_checks.csv', Q)
