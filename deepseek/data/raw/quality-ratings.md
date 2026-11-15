# Agentic output quality (Nov 14 afternoon), beyond the pass/fail checks

Rater: an LLM, NOT blind (the screenshots were seen with their labels before rating). Functional tests by
`quality_site.cjs` (headless Chromium: dark toggle + reload persistence, invalid-email inline error, valid
submit feedback, 375 px no horizontal scroll, console errors, labels); screenshots and sheets in
`~/deepseek-work/bench/quality/`.

## Functional (all runs)
Every site passed every functional test: dark mode switches and persists, invalid email -> inline error and no
submit, valid -> confirmation, no horizontal scroll at 375 px, no console errors, 3/3 inputs labelled.
Orders: every run made the same three fixes (parser `(\d+)`, tier `>=`, Decimal tax), minimal, verified by
running the tests, clear summary (kf43 original also changed report.py's default to Decimal: harmless).
Eyes: right in every run except kf45 + graft v4 (214/19683: no thinking, buggy brute-force script).

## Site, rubric 1-10 (design / depth of content / defects -> overall)
| Run | Time | Design | Depth | Defects | Overall |
|---|---|---|---|---|---|
| kf45 original, thinking on (dsh medium) | 2,109 s | 9 | 9: nav links, featured "Today's bake", 8 items, find-us block (address, email, phone), placeholders, notes | none | **9** |
| kf45 graft v5 | 788 s | 8 | 7: hero card + 2 CTAs, 8 items, hours table, footer; no contact details or nav links | none | **7.5** |
| kf45 thinking off | 572 s | 6 | 5: centred plain hero, no CTAs, 6 items, system fonts, no extras | none | **5.5** |
| kf43 original (morning) | 1,995 s | 6.5 | 6: 6 items + hours side card, plain form | none | **6.5** |
| kf43 graft2 (morning) | 1,042 s | 8.5 | 8.5: chip + CTAs, 8 items, find-us block, placeholders | hard-coded "Open today 7:00-18:00" (wrong Fri-Sun) | **8** |

Caveat that decides how to read this: the two ORIGINAL runs (same task, same model, different day/binary) differ
by 2.5 points (9 vs 6.5) - more than any arm-to-arm gap. One run per arm cannot separate v5 from the original on
quality; it does separate thinking off (lowest in both comparisons). Repeats + blind rating are needed.
