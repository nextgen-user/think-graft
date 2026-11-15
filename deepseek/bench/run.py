#!/usr/bin/env python3
# Agentic task-completion benchmark through the DeepSeek harness (headless profile) against the live server (Nov 13).
# Two tasks, each checked automatically: "eyes" (the tough bench's round-table problem #2, tools allowed, answer in
# answer.txt) and "orders" (find and fix three bugs in a small order-processing package; visible + hidden tests, the
# test file must stay unchanged).  Per run: wall time, pass/fail, the session's token usage.  Results: results.jsonl.
# usage: run.py TAG [--patch FILE] [task ...]
import sys, os, json, time, shutil, subprocess, hashlib, glob, tempfile
B = os.path.dirname(os.path.abspath(__file__)); W = os.path.expanduser('~/deepseek-work/bench')
DSH = os.environ.get('DSH') or os.path.expanduser('~/.local/bin/dsh'); SESS = os.path.expanduser('~/.local/share/deepseek-harness/state/storages/session_projcache/sessions')
EYES_Q = ("Twelve people sit evenly spaced around a round table, so each person has a left neighbour, a right neighbour, and "
          "exactly one person directly across the table. Simultaneously, each person independently and uniformly at random "
          "chooses to look at their left neighbour, their right neighbour, or the person directly across. Two people make "
          "eye contact if they look at each other. What is the probability that nobody makes eye contact?")
TASKS = {
    'eyes': (None, EYES_Q + " Work in the current directory; you may write and run code. Write the answer as a reduced "
             "fraction such as 3/7 on a single line in answer.txt, then reply with the answer."),
    'orders': ('orders', "The tests of the Python project in the current directory fail (run: python3 -m unittest). Find "
               "the bugs in the orders package and fix them so that all tests pass. Do not modify anything under tests/. "
               "Then reply with a one-line summary of each fix."),
    'site': (None, "Create a single-page website in index.html in the current directory (inline CSS and JavaScript, no "
             "external files or libraries) for a neighbourhood bakery called Crumb & Co. It needs: a hero section with "
             "the name and a tagline; a menu section with at least 6 items, each with a short description and a price in "
             "dollars; an opening-hours section for every day of the week; a contact form with name, email and message "
             "fields whose JavaScript validation shows an inline error message when the email is invalid instead of "
             "submitting; a dark-mode toggle button that switches a class on the body and remembers the choice in "
             "localStorage; and a responsive layout that stacks into one column below 600px wide. Make it look polished."),
    # (Nov 14, the publication study: tasks where depth and quality of work show; graded blind by bench-agentic/judge.sh)
    'promo': ('promo', "The Python package `orders` in the current directory prices order texts (run the tests with: python3 -m "
              "unittest). Add promo-code support. (1) An order text may contain one line of the form `CODE: <code>` (the word CODE "
              "and the code itself are case-insensitive, extra spaces allowed); it is not an item line. Add `parse_code(text)` to "
              "orders/parser.py returning the code in upper case, or None when the order has no code line; `parse_order` keeps "
              "returning only the item lines. (2) Define the codes in a new module orders/promos.py: SPRING10 = 10% off, valid "
              "through 2026-05-31 inclusive; BIG50 = $50.00 off, only when the order's subtotal before any discount is at least "
              "$300.00; WELCOME5 = $5.00 off. (3) `total(lines, code=None, on=None)` in orders/pricing.py, where `on` is a "
              "datetime.date (default: today): apply the volume tier discount first, then the promo code, then tax, rounding to "
              "cents (half up) only at the end as now. A fixed-amount code never takes the pre-tax amount below $0.00. Codes are "
              "case-insensitive everywhere; unknown, expired or ineligible codes give no promo discount and do not raise. "
              "(4) `customer_totals(orders, on=None)` in orders/report.py applies each order's code. (5) Keep the existing tests "
              "passing and add tests for the new behaviour in tests/test_promos.py. Then reply with a short summary of your design."),
    'analysis': ('analysis', "sales.csv in the current directory holds a bakery chain's daily sales for January to June 2026 (columns: "
                 "date, store, product, qty, unit_price). Analyse it and write report.md for the owner: the most important findings, "
                 "with numbers; any data-quality problems you find and how you handled them; and your recommendations. Include at "
                 "least one chart, saved as a PNG file in this directory and referenced from report.md. Python 3 with pandas and "
                 "matplotlib is available."),
    'postmortem': ('postmortem', "The files in the current directory are from last night's production incident at an online bakery "
                   "shop: app.log (application log), deploys.txt (deploy history) and disk.txt (disk usage of /var). Investigate "
                   "and write postmortem.md: a short summary, a timeline with times, the root cause, the impact (how many orders "
                   "failed and for how long), anything that looked related but was not, and concrete action items."),
}
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def check(task, d):
    if task == 'site': return check_site(d)
    if task == 'promo':   # the visible tests + the hidden spec tests (hidden/test_promo_hidden.py), in a copy
        t = tempfile.mkdtemp(); shutil.copytree(d, t, dirs_exist_ok=True); shutil.copy(os.path.join(B, 'hidden/test_promo_hidden.py'), os.path.join(t, 'tests'))
        r = subprocess.run([sys.executable, '-m', 'unittest', '-v'], cwd=t, capture_output=True, text=True, timeout=120); shutil.rmtree(t, ignore_errors=True)
        hid = [l for l in r.stderr.splitlines() if 'HiddenPromo' in l]; ok_h = sum(l.rstrip().endswith('ok') for l in hid)
        return r.returncode == 0, f"hidden {ok_h}/9, all tests {'OK' if r.returncode == 0 else (r.stderr.strip().splitlines() or ['?'])[-1]}"
    if task == 'analysis':
        rp = os.path.join(d, 'report.md'); pngs = [f for f in os.listdir(d) if f.lower().endswith('.png')]
        if not os.path.exists(rp): return False, 'no report.md'
        txt = open(rp, errors='replace').read(); ref = [f for f in pngs if f in txt]
        return bool(ref), f"report {len(txt)} chars, {len(pngs)} png, {len(ref)} referenced"
    if task == 'postmortem':
        rp = os.path.join(d, 'postmortem.md')
        if not os.path.exists(rp): return False, 'no postmortem.md'
        return True, f"postmortem {len(open(rp, errors='replace').read())} chars"
    if task == 'eyes':
        try: a = open(os.path.join(d, 'answer.txt')).read().strip().replace(' ', '')
        except OSError: return False, 'no answer.txt'
        return a == '2618/59049', a[:60]
    if sha(os.path.join(d, 'tests/test_orders.py')) != sha(os.path.join(B, 'fixtures/orders/tests/test_orders.py')):
        return False, 'tests/test_orders.py was modified'
    t = tempfile.mkdtemp(); shutil.copytree(d, t, dirs_exist_ok=True); shutil.copy(os.path.join(B, 'hidden/test_hidden.py'), os.path.join(t, 'tests'))
    r = subprocess.run([sys.executable, '-m', 'unittest'], cwd=t, capture_output=True, text=True, timeout=120)
    shutil.rmtree(t, ignore_errors=True)
    last = (r.stderr.strip().splitlines() or ['?'])[-1]
    return r.returncode == 0, last
def check_site(d):
    import re, html.parser
    p = os.path.join(d, 'index.html')
    if not os.path.exists(p): return False, 'no index.html'
    h = open(p, errors='replace').read(); low = h.lower()
    try: html.parser.HTMLParser().feed(h); parsed = True
    except Exception: parsed = False
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', h, re.S | re.I); js = '\n'.join(scripts)
    jsok = True
    if js.strip():
        t = tempfile.NamedTemporaryFile('w', suffix='.js', delete=False); t.write(js); t.close()
        jsok = subprocess.run(['node', '--check', t.name], capture_output=True).returncode == 0; os.unlink(t.name)
    days = sum(d_ in low for d_ in ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'))
    checks = {
        'parses': parsed and len(h) > 3000, 'name': 'crumb &amp; co' in low or 'crumb & co' in low,
        'six_prices': len(re.findall(r'\$\s?\d+(?:\.\d{2})?', h)) >= 6, 'all_days': days == 7,
        'form_fields': all(x in low for x in ('<form', 'textarea')) and ('type="email"' in low or "name=\"email\"" in low or 'id="email"' in low),
        'email_validation': 'email' in js.lower() and any(x in js for x in ('.test(', 'checkValidity', 'validity', 'match(')) and 'preventDefault' in js,
        'dark_toggle': 'localstorage' in js.lower() and 'dark' in js.lower() and 'classList' in js,
        'responsive': '@media' in low and '600px' in low, 'js_syntax': jsok and bool(js.strip()),
        'no_external': not re.search(r'<(script|link)[^>]+(src|href)=["\']https?://', h, re.I),
    }
    try:   # a screenshot for judging the look (not scored).  Firefox is a snap: it cannot read or write under hidden
        # folders (~/.local/...), and a running instance owns the default profile, so the site is copied to
        # ~/deepseek-work/shot-tmp and shot there with a throwaway profile.
        base = os.path.expanduser('~/deepseek-work/shot-tmp'); os.makedirs(base, exist_ok=True)
        tmp = tempfile.mkdtemp(dir=base); site = os.path.join(tmp, 'site'); shutil.copytree(d, site)
        prof = os.path.join(tmp, 'profile'); os.makedirs(prof); png = os.path.join(tmp, 'shot.png')
        subprocess.run(['firefox', '--headless', '--no-remote', '--profile', prof, '--window-size=1280,1800', '--screenshot', png,
                        'file://' + os.path.join(site, 'index.html')], capture_output=True, timeout=90)
        if os.path.exists(png): shutil.copy(png, os.path.join(d, '..', os.path.basename(d) + '.png'))
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception: pass
    bad = [k for k, v in checks.items() if not v]
    return not bad, f"{len(checks) - len(bad)}/{len(checks)}" + (f" missing {bad}" if bad else '')
def usage(d, t0):
    best = None
    for f in glob.glob(os.path.join(SESS, '*.json')):
        if os.path.getmtime(f) < t0: continue
        try: rec = json.load(open(f))['record']
        except Exception: continue
        if rec['identity'].get('cwd') == d: best = rec['rows']
    if not best: return {}
    u = best.get('tokenUsage', {}).get('val', {}).get('totals', {}); s = best.get('sessionStats', {}).get('val', {})
    return dict(u, **{k: s.get(k) for k in ('steps', 'llmMs', 'toolMs', 'ttftMs', 'decodeMs', 'decodeTokens') if k in s})
def main():
    a = sys.argv[1:]; tag = a.pop(0); patch = None
    if a and a[0] == '--patch': patch = os.path.abspath(a[1]); a = a[2:]
    for task in (a or list(TASKS)):
        fx, prompt = TASKS[task]; d = os.path.join(W, f'{tag}-{task}')
        shutil.rmtree(d, ignore_errors=True)
        if fx: shutil.copytree(os.path.join(B, 'fixtures', fx), d)
        else: os.makedirs(d)
        cmd = [DSH, '--profile', 'headless'] + (['--patch', patch] if patch else []) + [prompt]
        t0 = time.time()
        with open(os.path.join(d, '..', f'{tag}-{task}.log'), 'w') as log:
            try: rc = subprocess.run(cmd, cwd=d, stdout=log, stderr=subprocess.STDOUT, timeout=3600).returncode
            except subprocess.TimeoutExpired: rc = 'timeout'
        wall = time.time() - t0; ok, why = check(task, d)
        r = dict(tag=tag, task=task, dsh=os.path.basename(DSH), patch=os.path.basename(patch) if patch else None, ok=ok, check=why, wall_s=round(wall, 1), rc=rc, **usage(d, t0))
        print(json.dumps(r), flush=True)
        with open(os.path.join(B, 'results.jsonl'), 'a') as f: f.write(json.dumps(r) + '\n')
main()
