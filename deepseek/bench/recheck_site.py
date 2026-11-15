# Grades site runs whose result row was lost (Nov 14: run.py's check_site crashed on a local import), after the fact:
# the checklist and screenshot from the folder, the wall time from the folder's creation (made just before dsh starts) to
# the log's last write, the token usage from the dsh session store.  usage: recheck_site.py TAG [TAG..]
import sys, os, json, time
sys.argv, TAGS = [sys.argv[0]], sys.argv[1:]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run.py')).read().replace('\nmain()\n', '\n')
exec(compile(src, 'run.py', 'exec'))
for tag in TAGS:
    d = os.path.join(W, f'{tag}-site'); lg = os.path.join(W, f'{tag}-site.log')
    t0 = os.stat(d).st_birthtime if hasattr(os.stat(d), 'st_birthtime') else min(os.path.getmtime(lg), os.stat(d).st_ctime)
    import subprocess as sp
    b = sp.run(['stat', '-c', '%W', d], capture_output=True, text=True).stdout.strip()
    if b.isdigit() and int(b) > 0: t0 = int(b)
    wall = os.path.getmtime(lg) - t0; ok, why = check('site', d)
    r = dict(tag=tag, task='site', dsh='dsh-bench', patch='patch-8096.yml', ok=ok, check=why, wall_s=round(wall, 1), rc='recheck', **usage(d, t0))
    print(json.dumps(r))
    with open(os.path.join(B, 'results.jsonl'), 'a') as f: f.write(json.dumps(r) + '\n')
