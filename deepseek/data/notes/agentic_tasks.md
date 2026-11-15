# Agentic tasks: the exact user prompts (taken from the dsh session transcripts)

Harness: DeepSeek Harness (dsh, `adaptive-experts/bench-agentic/dsh-bench`), local model through a server copy on port 8096;
checks in `adaptive-experts/bench-agentic/run.py` (check()/check_site()), answer keys in `raw/bench-agentic/keys/`.
dsh effort settings (patch files in `adaptive-experts/serve-conv/`): medium = template effort 63 + thinking budget 2,048 tokens per step
(the default); low = effort 50 + 1,024 per step (`patch-8096-low.yml`); off = thinking disabled (`patch-8096-off.yml`);
max = effort 100, no budget entry (`patch-8096-max.yml`, never completed). When the per-step budget runs out, the server
injects "I have thought about this long enough; time to give my answer." and closes the thinking ("capped step").

## site

Automatic check: 10 checks in run.py check_site + headless-Chromium checks (quality_site.cjs).

```
Create a single-page website in index.html in the current directory (inline CSS and JavaScript, no external files or libraries) for a neighbourhood bakery called Crumb & Co. It needs: a hero section with the name and a tagline; a menu section with at least 6 items, each with a short description and a price in dollars; an opening-hours section for every day of the week; a contact form with name, email and message fields whose JavaScript validation shows an inline error message when the email is invalid instead of submitting; a dark-mode toggle button that switches a class on the body and remembers the choice in localStorage; and a responsive layout that stacks into one column below 600px wide. Make it look polished.
```

## eyes

Automatic check: answer.txt must equal 2618/59049.

```
Twelve people sit evenly spaced around a round table, so each person has a left neighbour, a right neighbour, and exactly one person directly across the table. Simultaneously, each person independently and uniformly at random chooses to look at their left neighbour, their right neighbour, or the person directly across. Two people make eye contact if they look at each other. What is the probability that nobody makes eye contact? Work in the current directory; you may write and run code. Write the answer as a reduced fraction such as 3/7 on a single line in answer.txt, then reply with the answer.
```

## orders

Automatic check: python3 -m unittest must pass.

```
The tests of the Python project in the current directory fail (run: python3 -m unittest). Find the bugs in the orders package and fix them so that all tests pass. Do not modify anything under tests/. Then reply with a one-line summary of each fix.
```

## promo

Automatic check: visible tests + 9 hidden spec tests (hidden/test_promo_hidden.py).

```
The Python package `orders` in the current directory prices order texts (run the tests with: python3 -m unittest). Add promo-code support. (1) An order text may contain one line of the form `CODE: <code>` (the word CODE and the code itself are case-insensitive, extra spaces allowed); it is not an item line. Add `parse_code(text)` to orders/parser.py returning the code in upper case, or None when the order has no code line; `parse_order` keeps returning only the item lines. (2) Define the codes in a new module orders/promos.py: SPRING10 = 10% off, valid through 2026-05-31 inclusive; BIG50 = $50.00 off, only when the order's subtotal before any discount is at least $300.00; WELCOME5 = $5.00 off. (3) `total(lines, code=None, on=None)` in orders/pricing.py, where `on` is a datetime.date (default: today): apply the volume tier discount first, then the promo code, then tax, rounding to cents (half up) only at the end as now. A fixed-amount code never takes the pre-tax amount below $0.00. Codes are case-insensitive everywhere; unknown, expired or ineligible codes give no promo discount and do not raise. (4) `customer_totals(orders, on=None)` in orders/report.py applies each order's code. (5) Keep the existing tests passing and add tests for the new behaviour in tests/test_promos.py. Then reply with a short summary of your design.
```

## analysis

Automatic check: report.md + a referenced PNG; graded by hand against keys/analysis.md.

```
sales.csv in the current directory holds a bakery chain's daily sales for January to June 2026 (columns: date, store, product, qty, unit_price). Analyse it and write report.md for the owner: the most important findings, with numbers; any data-quality problems you find and how you handled them; and your recommendations. Include at least one chart, saved as a PNG file in this directory and referenced from report.md. Python 3 with pandas and matplotlib is available.
```

## postmortem (defined, never completed by any arm)

Check: postmortem.md, graded against keys/postmortem.md.

```
The files in the current directory are from last night's production incident at an online bakery shop: app.log (application log), deploys.txt (deploy history) and disk.txt (disk usage of /var). Investigate and write postmortem.md: a short summary, a timeline with times, the root cause, the impact (how many orders failed and for how long), anything that looked related but was not, and concrete action items.
```
