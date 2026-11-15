# Observations (with the numbers behind each; all kf43 unless stated; one run per cell)

## Maths

1. **Over-checking dominates max effort.** Plain max on the 6 shared problems: 39,946 thinking tokens, 25,363 of them
   after the right answer had already appeared. dice_nine: right at 2,429 tokens, then 7,541 tokens of re-checking.
   round_table (kf14): reached 2618/59049 at ~17,800 tokens and kept verifying (no repetition) until the 32k cap: no
   final answer line, graded wrong.
2. **graft2 removes most of it, but part of it moves into the visible answer.** graft2's thinking on the 6 problems:
   17,996 (check phase 2,630), visible answer 7,116 (coin 3,611, code_trace 1,922: it re-checks inside the answer).
   Net 25,112 tokens vs 41,559 (−40%), 35.3 vs 64.1 minutes.
3. **Lowering effort gives the same token count, differently.** Plain low: 25,548 tokens, but more solve tokens
   (16,469 vs graft2's 15,366 / max's 14,583) and a wrong first pass on code_trace (93, fixed only by a full
   re-derivation). Low cuts checking (7,781) yet solves sloppier; graft2 keeps max-effort solving.
4. **graft2 was the only arm to solve round_table** (27,696 tokens). One sample; treat as a hint.
5. **Early stopping has a real accuracy cost sometimes.** v4 closed after a shallow check on digit_sum and answered
   785 (right: 791); continuing its thinking with the unedited model found the slip at +1.4k/+4.8k tokens (2 seeds).
   In the low-effort code_trace trace, a shallow "Correct." check passed the wrong 93; only a full re-derivation found
   85. Checking rescued ~2 of ~30 runs.
6. **Stray `</think>`**: graft2's visible answers contain extra close tags (round_table 56, others 1-2) because its
   readout also fires on re-checking inside the answer; fixed in v4+ (and properly by the v7 fit).

## Agent tasks

7. **The cap-thrash failure mode** (original, site task): it drafts the whole HTML/CSS line by line inside its thinking
   (counting lines for dsh's "max 200 lines per tool call" rule), the 2,048-token budget cuts it mid-draft, it emits a
   stalling `ls`/`pwd`, and re-drafts from scratch next step; 4-5 times a run = 63-70% of its 13-15k thinking tokens.
   Earlier thinking IS in its context (verified: each prompt contains the previous one, thinking included); the
   restart is behavioural.
8. **Low effort makes it worse on agent work**: half the budget (1,024) → more capped steps (5), 36 steps instead of 10,
   2,702 s vs 1,995 s at medium, 885k re-read context tokens vs ~64k for graft2. Fewer tokens per step became more steps.
9. **graft2 thinks where it matters** (per-step transcripts): site: 0 on `ls`, 2,062 on the plan, 0 while writing and
   checking; orders: 220 tokens listing the bugs right after reading the code, 0 elsewhere; eyes: 0 (it wrote a
   brute-force script directly); promo (unseen): 1,305 (requirements) + 2,063 (implementation plan), then 0 on routine
   steps, **128 tokens right after "6 tests FAILED"** (diagnosis) and 76 for the fix; analysis (unseen): 0 on routine
   steps, 571 after a code error, 1,411 after the data-quality findings, 1,877 after failing to view its own chart.
10. **Surprise probe (exact)**: graft2's P(no thinking) is ~0.95 on obvious steps, ≤0.004 when there is new
    information or an error message, 0.91 after failing tests (the next action, "read the code", needs no thought).
11. **Thinking off is fast but shallower** (kf45 site: 572 s; blind pilot quality 7 vs original 9 vs v5 8).
12. **v7 (correctly fitted) keeps a little thinking on every step** (site 3,583 tokens, 10 steps, 1 capped step at
    the spec→drafting boundary, then wrote the file immediately; no re-plan) - safer but less aggressive than graft2.
    Its design spec in thinking (dark-mode logic, validation, a "today" row highlight) showed up in the page.
13. **Analysis task (planted traps)**: both graft2 and low found the weekend effect (×1.59) and all three data
    problems, missed the galette decline, and made false claims from comparing month totals instead of daily rates
    (February "dip" = 28-day month). graft2 also partly saw Elm's growth and wrongly called Maple/Oak "declining".
    The subtle effects are where more (targeted) thinking might pay; neither arm found them.

## Quality and variance

14. Two runs of the same original on the same task scored 9 and 6.5 (non-blind): single runs cannot rank close arms.
15. graft2's site had one factual slip: a hard-coded "Open today 7:00-18:00" (wrong Fri-Sun).
16. Low's site overflows at 375 px (397 px wide); every other site passed every browser check.

## Hardware cost (relevant for anyone reproducing on a laptop)

17. Streaming a 243 GB model from one consumer SSD: ~1.5 GB/s reads while generating, ~110 TB read in 36 hours; the
    drive's internal NAND writes are ~5× host writes (refresh/garbage collection), ≈ 0.026% SSD wear per hour of
    running; the drive went ~4% → 7% wear over this project. A scratchpad model that fits in VRAM avoids this.
