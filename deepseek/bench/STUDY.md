# Is the probe graft worth it, or is it just a lower effort setting?  (pre-registered Nov 14, 15:00, before results)

## Question
Does graft v5 (one edited row of the output head: </think> closes when the model's own state says the answer is there)
give a better quality-for-time trade than what anyone gets without it, by choosing a thinking effort?

## Arms (all on production's kf45 binary + flags, same machine, each arm a fresh server with an empty cache)
- orig: unedited model, dsh effort medium (what dsh does by default; template effort 63, 2,048-token thinking budget a step)
- low:  unedited model, dsh effort low (effort 50, 1,024-token budget a step)
- v5:   graft v5 overlay (strata-pack-graft5w8d), dsh effort medium
- off:  unedited model, dsh thinking off
Maths (bench_think, 7 problems, same seeds): plain max (have), plain high, graft v5 at max; graft2/v4 runs as history.

## Tasks (agentic, through the real dsh harness)
site (bakery page: design + function), promo (feature in a code base: hidden spec tests + design), analysis (sales CSV with
planted effects and planted data errors -> report + chart), postmortem (incident logs with a planted root cause and red
herrings).  2 repeats per arm (site 3 for orig/v5/off), arm order reversed in repeat 2.  Plus eyes/orders (correctness only).

## Measures
- time: wall seconds per task (dsh start to finish), and output tokens
- objective: promo hidden tests (x/9), eyes/orders pass, maths answers
- quality: blind judge = codex exec, gpt-6-luna, read-only; candidates anonymised A-D in shuffled order, each instance
  judged twice (orders swapped); rubric per task (1-10 per criterion + overall + rank); answer keys for the planted facts;
  the judge never sees arm names, times, tokens or reasoning.  Score = mean overall over shuffles and repeats.

## Decision rule (fixed now)
v5 is WORTH IT if both hold:
1. vs low (the effort setting a user would pick to save time): v5's mean blind score is >= low's + 0.5 with v5's total
   time <= 1.15x low's; or v5's score >= low's - 0.25 with v5's time <= 0.75x low's.
2. vs orig: v5's mean blind score >= orig's - 0.75, with v5's total time <= 0.6x orig's.
If 1 fails, the graft is "just a lower effort setting" for agentic work and we say so.  Maths is reported separately:
v5 at max must match plain max's answers in fewer tokens and beat plain high on answers or tokens.
Per-task results are shown, not only the mean; a task where v5 loses is reported as a loss.

## Known limitations (stated up front)
One machine, one quantised model (IQ2_XXS); small n (2-3 runs per arm and task); an LLM judge (bias toward length and
style is possible; mitigated by rubric, keys, two orders); sampling at temperature 1.0 makes single runs noisy; the
graft's probe was fitted on two maths traces plus agentic start-of-thinking rows from these same three original tasks
(eyes/orders/site) - the four study tasks other than site are new to it.

## Amendment (Nov 14 15:20, before any plain-low or new-task result existed)
Trimmed for time at the user's request: one pass per arm on promo/analysis/postmortem (no repeat 2); site judged once with
four arms (orig/v5/off from 13:00-14:30 + low from this pass); plain high on maths dropped; plain LOW on the 7 maths
problems added as the effort-dial point.  The decision rule above is unchanged, applied to these single runs; with n = 1
per task a one-point judge difference is within noise and will be reported as such.
Added analyses (no extra runs): solve/check/answer token split of every maths run (delete/phase_split.py ->
delete/phase-split.json) and the exact offline beta sweep (delete/beta_sweep.py -> delete/beta-sweep.json; in-sample
traces: the two the probe was fitted on).
The pilot blind judgement of site (3 arms, 2 orders: orig 9, v5 8, off 7) is kept as a pilot (judge/results-pilot-3arms.jsonl).

## Amendment 2 (Nov 14 16:10, before any kf43 study result existed)
Everything moves to kf43 (production until today; no half deferral, which is measurably lossy: qmini 2/4 vs 3/4), the
user's call.  v5 is refitted for kf43's decode path (delete/hx1, agentic rows x8; overlay strata-pack-graft5w8; sim:
digit close 4,064 / coin 4,624, solving-close <= 0.006).  Maths, all kf43: plain max (held: base-held; tough: rerun),
plain low, v5 max.  Agentic, all kf43 (tags k43-*): original = this morning's ag-base for site/eyes/orders (same
conditions: fresh server copy, empty slot folder) + promo/analysis/postmortem; low, v5 and thinking off on all six tasks.
The kf45 runs (ag5-*, g5-max-*, stack-*) are kept as a second-engine reference, not part of the decision.
