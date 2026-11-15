# The publication study's three new agentic tasks (Nov 14): fixtures the agent sees, hidden tests and answer keys it
# does not (keys/ is given to the blind judge only).  Deterministic (fixed seeds).  usage: python3 make_tasks.py
import os, shutil, random, datetime as dt, json, math
B = os.path.dirname(os.path.abspath(__file__)); F = os.path.join(B, 'fixtures'); K = os.path.join(B, 'keys')
os.makedirs(K, exist_ok=True)
# ---- promo: the fixed orders package (a correct run's output) + its tests; hidden tests pin the specified behaviour
src = os.path.expanduser('~/deepseek-work/bench/ag5-base-orders'); dst = os.path.join(F, 'promo'); shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
open(os.path.join(B, 'hidden', 'test_promo_hidden.py'), 'w').write('''import unittest
from datetime import date
from decimal import Decimal
from orders.parser import parse_order, parse_code
from orders.pricing import total
from orders.report import customer_totals
D = date(2026, 5, 31)


class HiddenPromo(unittest.TestCase):
    def test_parse_code(self):
        self.assertEqual(parse_code("CODE: spring10\\n2x A @ 1"), "SPRING10")
        self.assertEqual(parse_code("2x A @ 1\\n  code:  big50  "), "BIG50")
        self.assertIsNone(parse_code("2x A @ 1"))

    def test_code_line_not_an_item(self):
        self.assertEqual([l.item for l in parse_order("CODE: WELCOME5\\n2x A @ 1\\n1x B @ 2")], ["A", "B"])

    def test_spring_valid_and_expired(self):
        lines = parse_order("10x A @ 10")
        self.assertEqual(total(lines, "SPRING10", D), Decimal("106.20"))          # 100 -> 90 -> 106.20
        self.assertEqual(total(lines, "SPRING10", date(2026, 6, 1)), Decimal("118.00"))

    def test_code_case_insensitive(self):
        self.assertEqual(total(parse_order("10x A @ 10"), "spring10", D), Decimal("106.20"))

    def test_big50_minimum(self):
        self.assertEqual(total(parse_order("30x A @ 10"), "BIG50", D), Decimal("295.00"))     # 300 -> 250
        self.assertEqual(total(parse_order("1x A @ 299.99"), "BIG50", D), Decimal("353.99"))  # below minimum

    def test_order_tier_then_promo_then_tax(self):
        self.assertEqual(total(parse_order("60x A @ 10"), "BIG50", D), Decimal("613.60"))     # 600 -> 570 -> 520
        self.assertEqual(total(parse_order("60x A @ 10"), "SPRING10", D), Decimal("605.34"))  # 600 -> 570 -> 513

    def test_welcome_and_floor(self):
        self.assertEqual(total(parse_order("2x A @ 10"), "WELCOME5", D), Decimal("17.70"))
        self.assertEqual(total(parse_order("1x A @ 3"), "WELCOME5", D), Decimal("0.00"))

    def test_unknown_and_none(self):
        self.assertEqual(total(parse_order("2x A @ 10"), "BOGUS", D), Decimal("23.60"))
        self.assertEqual(total(parse_order("2x A @ 10")), Decimal("23.60"))

    def test_report_applies_codes(self):
        got = customer_totals([("a", "CODE: WELCOME5\\n2x A @ 10"), ("b", "1x B @ 5"), ("a", "1x B @ 5")], on=D)
        self.assertEqual(got, {"a": Decimal("23.60"), "b": Decimal("5.90")})
        self.assertTrue(all(isinstance(v, Decimal) for v in got.values()))
''')
# ---- analysis: a bakery chain's daily sales Jan-Jun 2026 with planted effects and planted data problems
rnd = random.Random(2026); dst = os.path.join(F, 'analysis'); shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
P = {'Sourdough loaf': (7.50, 30), 'Butter croissant': (4.00, 55), 'Cinnamon roll': (4.50, 35), 'Seeded rye': (6.50, 14),
     'Rosemary focaccia': (8.00, 12), 'Chocolate babka': (8.50, 10), 'Seasonal galette': (6.00, 12), 'Almond cookie': (2.50, 40)}
stores = ['Elm', 'Maple', 'Oak']; rows = []; d0 = dt.date(2026, 1, 1); days = (dt.date(2026, 6, 30) - d0).days + 1
def pois(l):
    L, k, p = math.exp(-l), 0, 1.0
    while True:
        k += 1; p *= rnd.random()
        if p <= L: return k - 1
for i in range(days):
    d = d0 + dt.timedelta(i); wk = 1.6 if d.weekday() >= 5 else 1.0
    for s in stores:
        sf = {'Elm': 1.0, 'Maple': 0.85, 'Oak': 0.7}[s] * (1.35 if s == 'Elm' and d >= dt.date(2026, 3, 16) else 1.0)
        for name, (price, base) in P.items():
            tr = 1.0 - 0.6 * i / (days - 1) if name == 'Seasonal galette' else 1.0
            rows.append([d.isoformat(), s, name, pois(base * wk * sf * tr), f'{price:.2f}'])
clean = [list(r) for r in rows]   # (a deep copy: the planted errors below must not touch the key)
for _ in range(24): rows.insert(rnd.randrange(len(rows)), list(rows[rnd.randrange(len(rows))]))      # exact duplicates
neg = rnd.sample(range(len(rows)), 6)
for j in neg: rows[j][3] = -rows[j][3] if rows[j][3] else -2                                          # negative quantities
for j in rnd.sample([k for k in range(len(rows)) if k not in neg], 3): rows[j][4] = ''                # missing prices
with open(os.path.join(dst, 'sales.csv'), 'w') as f:
    f.write('date,store,product,qty,unit_price\n'); f.writelines(','.join(map(str, r)) + '\n' for r in rows)
def rev(rs, cond): return sum(r[3] * float(r[4]) for r in rs if cond(r))
wkend = [d0 + dt.timedelta(i) for i in range(days) if (d0 + dt.timedelta(i)).weekday() >= 5]
we = rev(clean, lambda r: dt.date.fromisoformat(r[0]).weekday() >= 5) / len(wkend); wd = rev(clean, lambda r: dt.date.fromisoformat(r[0]).weekday() < 5) / (days - len(wkend))
gal = lambda m: sum(r[3] for r in clean if r[2] == 'Seasonal galette' and r[0][5:7] == m)
elm_b = rev(clean, lambda r: r[1] == 'Elm' and r[0] < '2026-03-16') / (dt.date(2026, 3, 16) - d0).days
elm_a = rev(clean, lambda r: r[1] == 'Elm' and r[0] >= '2026-03-16') / (days - (dt.date(2026, 3, 16) - d0).days)
open(os.path.join(K, 'analysis.md'), 'w').write(f'''# Answer key: analysis (planted in the generator, make_tasks.py)
Planted effects (clean data, before the planted errors):
1. Weekend effect: Saturday/Sunday demand is 1.6x weekdays (avg daily revenue weekend ${we:,.0f} vs weekday ${wd:,.0f}, ratio {we / wd:.2f}).
2. "Seasonal galette" declines steadily (demand -60% from the first day to the last; monthly units Jan {gal("01")} -> Jun {gal("06")}, {100 * (gal("06") / gal("01") - 1):.0f}%); every other product is flat.
3. Store Elm jumps from 2026-03-16 (avg daily revenue {elm_b:,.0f} before vs {elm_a:,.0f} after, {100 * (elm_a / elm_b - 1):+.0f}%; demand +35%); Maple and Oak are flat.
   Store sizes differ: Elm > Maple (0.85x) > Oak (0.70x).
Planted data problems:
- 24 exact duplicate rows; - 6 rows with negative quantities; - 3 rows with an empty unit_price.
A strong report finds all three effects with numbers, notices all three data problems and says how it handled them,
and makes actionable recommendations (e.g. weekend staffing/stock, review or drop the galette, find out what changed at
Elm on Mar 16 and copy it). Claims of effects that are not in the data (e.g. a product trend other than the galette)
are errors.
''')
# ---- postmortem: last night's incident; the cause is a config flag in a deploy, with a red herring in between
dst = os.path.join(F, 'postmortem'); shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst); rnd = random.Random(7)
open(os.path.join(dst, 'deploys.txt'), 'w').write('''2026-11-13 22:10  web v2.3.0   (feature: gift cards)               by priya
2026-11-14 01:50  web v2.3.1   (fix: checkout copy; config: audit.verbose=true for the fraud review)  by dan
2026-11-14 03:05  web v2.3.2   (revert audit.verbose=false)          by priya
''')
disk = []; t = dt.datetime(2026, 11, 14, 1, 30)
use = 61.0
while t <= dt.datetime(2026, 11, 14, 3, 30):
    if dt.datetime(2026, 11, 14, 1, 50) <= t < dt.datetime(2026, 11, 14, 2, 11): use = min(100.0, use + 1.9)
    if t == dt.datetime(2026, 11, 14, 2, 11): use = 100.0
    if t == dt.datetime(2026, 11, 14, 2, 41): use = 64.0
    if dt.datetime(2026, 11, 14, 2, 41) < t < dt.datetime(2026, 11, 14, 3, 5): use = min(99.0, use + 0.9)
    if t == dt.datetime(2026, 11, 14, 3, 5): use = 87.0
    disk.append(f'{t:%H:%M} /var {use:5.1f}% used')
    t += dt.timedelta(minutes=1)
open(os.path.join(dst, 'disk.txt'), 'w').write('# /var usage, sampled every minute (db data and logs live on /var)\n' + '\n'.join(disk) + '\n')
L = []; failed = 0
def ev(h, m, s, lvl, msg): L.append((dt.datetime(2026, 11, 14, h, m, s), lvl, msg))
for m in range(30, 60): ev(1, m, rnd.randrange(60), 'INFO', f'order {rnd.randrange(40000, 41000)} placed total=${rnd.uniform(8, 90):.2f}')
ev(1, 50, 12, 'INFO', 'deploy web v2.3.1 started'); ev(1, 51, 3, 'INFO', 'deploy web v2.3.1 healthy (12/12 instances)')
ev(1, 51, 4, 'INFO', 'audit: verbose mode on (full request bodies)')
for m in range(52, 60): ev(1, m, 30, 'DEBUG', f'audit: log file audit.log size {1.2 + (m - 51) * 0.6:.1f} GB')
for m in range(0, 11): ev(2, m, 30, 'DEBUG', f'audit: log file audit.log size {6.0 + m * 0.62:.1f} GB')
for s in range(0, 120, 7): ev(2, 5, s % 60 if s < 60 else s - 60, 'WARN', 'cdn: upstream 502 from edge fra-2 for /static/app.css (retrying)') if s < 60 else ev(2, 6, s - 60, 'WARN', 'cdn: upstream 502 from edge fra-2 for /static/app.js (retrying)')
ev(2, 7, 2, 'INFO', 'cdn: edge fra-2 healthy again')
for m in range(0, 11): ev(2, m, rnd.randrange(60), 'INFO', f'order {rnd.randrange(41000, 42000)} placed total=${rnd.uniform(8, 90):.2f}')
ev(2, 3, 11, 'WARN', 'cache: warm-up miss ratio 0.31 for menu:today (expected after deploy)')
for m in range(11, 41):
    for _ in range(rnd.choice([1, 1, 2, 2, 3])):
        failed += 1; ev(2, m, rnd.randrange(60), 'ERROR', f'order {rnd.randrange(42000, 43000)} failed: db write error: could not extend file "base/16384/2619": No space left on device')
    if m % 6 == 0: ev(2, m, 59, 'ERROR', 'healthcheck: /checkout 500 (db write failed)')
ev(2, 38, 40, 'INFO', 'oncall (sam) paged: checkout errors'); ev(2, 41, 5, 'INFO', 'oncall: removed /var/log/web/audit.log.* (38 GB freed)')
for m in range(42, 60): ev(2, m, rnd.randrange(60), 'INFO', f'order {rnd.randrange(43000, 44000)} placed total=${rnd.uniform(8, 90):.2f}')
for m in range(45, 60, 5): ev(2, m, 30, 'DEBUG', f'audit: log file audit.log size {(m - 41) * 0.55:.1f} GB')
ev(3, 5, 20, 'INFO', 'deploy web v2.3.2 healthy (12/12 instances)'); ev(3, 5, 21, 'INFO', 'audit: verbose mode off')
L.sort(); open(os.path.join(dst, 'app.log'), 'w').write(''.join(f'{t:%Y-%m-%d %H:%M:%S} {l:5s} {m}\n' for t, l, m in L))
open(os.path.join(K, 'postmortem.md'), 'w').write(f'''# Answer key: postmortem (planted in make_tasks.py)
Root cause: the 01:50 deploy web v2.3.1 turned on audit.verbose=true (full request bodies); audit.log grew ~0.6 GB/min
and filled /var (61% -> 100% by 02:11). The database lives on /var, so every order write failed with "No space left on
device" from 02:11 to 02:41 (30 minutes): {failed} failed orders, checkout healthchecks 500.
Recovery: on-call (paged 02:38) deleted old audit logs at 02:41 (38 GB) - a temporary fix: the flag was still on and
/var filled again (64% -> ~85% by 03:05) until v2.3.2 reverted audit.verbose at 03:05. The incident really ended at 03:05,
and it would have recurred without the revert.
Red herrings: the CDN 502s from edge fra-2 at 02:05-02:07 (static assets, healed by themselves, before the db errors);
the cache warm-up miss warning at 02:03 (expected after a deploy). Neither caused the outage.
Good action items: disk-usage alerting on /var well before 100% (paging took 27 min), separate log and db volumes, log
rotation/size caps for audit logs, review config changes in deploys (a "copy fix" release shipped a risky flag), alert on
order failure rate, runbook: revert the flag rather than delete logs.
''')
print('fixtures: promo, analysis, postmortem; hidden/test_promo_hidden.py; keys/analysis.md, keys/postmortem.md; failed orders', failed)
