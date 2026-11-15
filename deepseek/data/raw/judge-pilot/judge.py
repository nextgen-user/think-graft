# Blind judging for the publication study (Nov 14): each task instance's outputs are copied into anonymous folders
# A, B, C, D (order shuffled; the mapping is kept OUTSIDE the judged folder), and `codex exec` (gpt-6-luna, read-only
# sandbox, no approvals needed) scores every candidate on a fixed rubric (1-10 per criterion, overall, rank) as JSON.
# Each instance is judged twice with different orders (position bias).  The judge never sees arm names, times, token
# counts or the agent's reasoning.  Results: ~/deepseek-work/judge/results.jsonl.
# usage: judge.py [task ...]      (only instances whose outputs all exist; done ones are skipped)
import os, sys, json, random, shutil, subprocess, difflib, glob, hashlib
B = os.path.dirname(os.path.abspath(__file__)); W = os.path.expanduser('~/deepseek-work/bench'); J = os.path.expanduser('~/deepseek-work/judge')
Q = os.path.join(W, 'quality'); os.makedirs(os.path.join(J, 'keys'), exist_ok=True)
sys.argv, TASKS_ARG = [sys.argv[0]], sys.argv[1:]
src = open(os.path.join(B, 'run.py')).read(); g = {'__file__': os.path.join(B, 'run.py'), '__name__': 'x'}
exec(compile(src[:src.rfind('main()')], 'run.py', 'exec'), g); TASKS = g['TASKS']
GROUPS = {'site': [('r0', {'orig': 'ag5-base', 'v5': 'ag5-v5', 'off': 'ag5-off'})]}
for t in ('site', 'promo', 'analysis', 'postmortem'):
    GROUPS.setdefault(t, [])
    for r in (1, 2): GROUPS[t].append((f'r{r}', {a: f'st-{a}-r{r}' for a in ('orig', 'low', 'v5', 'off')}))
# (Nov 14 16:10: everything on kf43; the original's site = this morning's ag-base, kf43, same conditions)
GROUPS = {t: [('k43', {'orig': 'ag-base' if t == 'site' else 'k43-orig', 'low': 'k43-low', 'v5': 'k43-v5', 'off': 'k43-off'})]
          for t in ('site', 'promo', 'analysis', 'postmortem')}
CRIT = {
    'site': ['visual_design', 'content_depth', 'correctness', 'code_quality'],
    'promo': ['correctness', 'design', 'tests', 'code_quality'],
    'analysis': ['findings', 'data_quality_handling', 'numbers_accuracy', 'recommendations', 'clarity_and_chart'],
    'postmortem': ['root_cause', 'impact_and_timeline', 'red_herrings', 'action_items', 'clarity'],
}
RUBRIC = {
    'site': "visual_design: layout, typography, colour, polish on desktop and phone (screenshots). content_depth: useful content "
            "beyond the bare brief (navigation, contact details, helpful copy). correctness: everything stated is consistent and "
            "right (e.g. hours shown in two places agree), the brief's requirements are met. code_quality: structure, semantic "
            "HTML, accessibility, robustness of the JavaScript. functional.json holds browser test results (all candidates passed "
            "dark mode, persistence, invalid-email error and phone layout tests).",
    'promo': "correctness: from test_results.txt (hidden spec tests x/9 and the full suite) and your reading of the code against "
             "the brief. design: clarity and extensibility of the promo model and of the pricing/report changes. tests: quality "
             "and coverage of the tests the candidate added (tests/test_promos.py). code_quality: readability, minimal and clean "
             "changes (changes.diff), no breakage of existing behaviour.",
    'analysis': "Use KEY.md (what was planted in the data) to check the report. findings: which planted effects were found and "
                "stated correctly; penalise claimed effects that are not in the data. data_quality_handling: which planted data "
                "problems were noticed and handled sensibly. numbers_accuracy: are the stated numbers right. recommendations: "
                "actionable and grounded in the data. clarity_and_chart: readable for an owner; chart(s) useful and correct.",
    'postmortem': "Use KEY.md (what happened). root_cause: the real cause identified precisely (and not a red herring). "
                  "impact_and_timeline: failed orders count, duration, times right, incl. the true end of the incident. "
                  "red_herrings: named and correctly ruled out. action_items: concrete, prevent recurrence, address detection. "
                  "clarity: well organised, concise, accurate.",
}
def schema(task):
    c = {k: {'type': 'integer', 'minimum': 1, 'maximum': 10} for k in CRIT[task]}
    item = {'type': 'object', 'additionalProperties': False, 'required': ['id', 'scores', 'overall', 'rank', 'reasons'],
            'properties': {'id': {'type': 'string'}, 'scores': {'type': 'object', 'additionalProperties': False, 'required': list(c), 'properties': c},
                           'overall': {'type': 'integer', 'minimum': 1, 'maximum': 10}, 'rank': {'type': 'integer', 'minimum': 1, 'maximum': 4},
                           'reasons': {'type': 'string'}}}
    return {'type': 'object', 'additionalProperties': False, 'required': ['candidates'], 'properties': {'candidates': {'type': 'array', 'items': item}}}
def collect(task, tag, out):   # one candidate's judged material (no logs, no times, no reasoning)
    d = os.path.join(W, f'{tag}-{task}'); os.makedirs(out); imgs = []
    if task == 'site':
        shutil.copy(os.path.join(d, 'index.html'), out)
        for k in ('desktop', 'mobile'):
            p = os.path.join(Q, f'{tag}-{k}.png')
            if os.path.exists(p): shutil.copy(p, os.path.join(out, f'{k}.png')); imgs.append(os.path.join(out, f'{k}.png'))
        fj = os.path.join(Q, f'{tag}-site.json')
        if os.path.exists(fj):
            r = json.load(open(fj)); r.pop('tag', None); json.dump(r, open(os.path.join(out, 'functional.json'), 'w'), indent=1)
    elif task == 'promo':
        fx = os.path.join(B, 'fixtures', 'promo'); lines = []
        for root, _, files in os.walk(d):
            if '__pycache__' in root: continue
            for f in sorted(files):
                if not f.endswith('.py'): continue
                p = os.path.join(root, f); rel = os.path.relpath(p, d); q = os.path.join(fx, rel)
                a = open(q).read().splitlines(True) if os.path.exists(q) else []
                lines += difflib.unified_diff(a, open(p).read().splitlines(True), 'a/' + rel, 'b/' + rel)
                os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True); shutil.copy(p, os.path.join(out, rel))
        open(os.path.join(out, 'changes.diff'), 'w').writelines(lines)
        ok, why = g['check']('promo', d); open(os.path.join(out, 'test_results.txt'), 'w').write(why + '\n')
    elif task == 'analysis':
        for f in os.listdir(d):
            if f == 'report.md' or f.lower().endswith('.png'):
                shutil.copy(os.path.join(d, f), out)
                if f.lower().endswith('.png'): imgs.append(os.path.join(out, f))
    elif task == 'postmortem':
        if os.path.exists(os.path.join(d, 'postmortem.md')): shutil.copy(os.path.join(d, 'postmortem.md'), out)
    return imgs
def done_keys():
    p = os.path.join(J, 'results.jsonl')
    return {(json.loads(l)['task'], json.loads(l)['rep'], json.loads(l)['shuffle']) for l in open(p)} if os.path.exists(p) else set()
def run(task, rep, arms, shuffle):
    name = f'{task}-{rep}-s{shuffle}'; jd = os.path.join(J, name); shutil.rmtree(jd, ignore_errors=True); os.makedirs(jd)
    order = sorted(arms); random.Random(int(hashlib.md5(name.encode()).hexdigest()[:8], 16)).shuffle(order)
    letters = 'ABCD'[:len(order)]; mapping = dict(zip(letters, order)); imgs = []
    for L, arm in mapping.items(): imgs += collect(task, arms[arm], os.path.join(jd, L))
    if task in ('analysis', 'postmortem'): shutil.copy(os.path.join(B, 'keys', f'{task}.md'), os.path.join(jd, 'KEY.md'))
    json.dump(mapping, open(os.path.join(J, 'keys', name + '.json'), 'w'))
    json.dump(schema(task), open(os.path.join(J, f'schema-{task}.json'), 'w'))
    brief = TASKS[task][1]
    prompt = (f"You are a strict, fair expert reviewer. {len(letters)} anonymous candidates ({', '.join(letters)}) each did the same "
              f"task; their work is in the folders {', '.join(letters)}/ of the current directory" + (" and in the attached images "
              "(each image's path names its candidate folder)" if imgs else "") + ". The task they were given:\n\n" + brief +
              f"\n\nRubric (score each criterion 1-10, then an overall 1-10 for the quality and depth of the work, and a rank 1 = "
              f"best with no ties): {RUBRIC[task]}\nJudge only the work itself. Do not reward length for its own sake. Read every "
              f"candidate's files before scoring. Return JSON with one entry per candidate, id = its folder letter; reasons = two "
              f"or three short sentences.")
    cmd = ['codex', 'exec', '-m', 'gpt-6-luna', '-s', 'read-only', '--skip-git-repo-check', '-C', jd, '--output-schema',
           os.path.join(J, f'schema-{task}.json'), '-o', os.path.join(J, name + '.out.json')]
    for im in imgs: cmd += ['-i', im]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=3600)   # (prompt on stdin: -i takes several files)
    open(os.path.join(J, name + '.log'), 'w').write(r.stdout[-20000:] + '\n---\n' + r.stderr[-20000:])
    res = json.load(open(os.path.join(J, name + '.out.json')))
    with open(os.path.join(J, 'results.jsonl'), 'a') as f:
        for c in res['candidates']:
            f.write(json.dumps(dict(task=task, rep=rep, shuffle=shuffle, letter=c['id'], arm=mapping.get(c['id']), tag=arms.get(mapping.get(c['id'], ''), ''),
                                    scores=c['scores'], overall=c['overall'], rank=c['rank'], reasons=c['reasons'])) + '\n')
    print(name, {mapping[c['id']]: (c['overall'], c['rank']) for c in res['candidates'] if c['id'] in mapping}, flush=True)
if __name__ == '__main__':
    done = done_keys()
    for task in (TASKS_ARG or list(CRIT)):
        for rep, arms in GROUPS[task]:
            arms = {a: t for a, t in arms.items() if os.path.isdir(os.path.join(W, f'{t}-{task}'))}
            if len(arms) < 2: continue
            if task == 'site':
                need = [t for t in arms.values() if not os.path.exists(os.path.join(Q, f'{t}-site.json'))]
                if need: subprocess.run(['node', os.path.join(B, 'quality_site.cjs'), *need], capture_output=True, timeout=1800)
            for sh in (0, 1):
                if (task, rep, sh) in done: continue
                try: run(task, rep, arms, sh)
                except Exception as e: print(f'{task}-{rep}-s{sh} FAILED: {e}', flush=True)
