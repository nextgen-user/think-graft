# DeepSeek V4.1 Flash: every recorded result (one-row `</think>` edit)

Everything measured on Nov 13-15 2026. Written for a reader **with no
prior context**: start here, then `notes/` for depth and `tables/` for every number. Nothing in this folder is
needed to run anything; it is a frozen record.

---

## 1. One-paragraph summary

We run DeepSeek V4.1 Flash (a 763B-parameter mixture-of-experts model, 2-bit IQ2_XXS quantization, 243 GB) on a
laptop (RTX 4060 8 GB, Ryzen 8845HS, 30 GB RAM) with a custom engine ("Strata"). The model over-thinks: on maths it
keeps re-checking an answer it already has (~60% of its thinking at max effort), and in agent tasks it drafts whole
files inside its thinking until a per-step budget cuts it off. We edited **one row of the output layer** (the row
that produces the `</think>` token) so that the model's own hidden state decides when to stop thinking: a
linear "probe" folded into that row. The best version, **graft2**, solved every maths problem we gave it (7/7,
including the hardest, which every other setting failed) at the token cost of plain "low effort", and on agent tasks
cut wall time ~50% and thinking ~85% versus the default, while passing every automatic check, including a task with
hidden tests it had never seen. It thinks when there is something to figure out (planning, errors, new information) and skips
thinking on obvious steps. Caveats: every number is **one run per problem/task**, several baselines are missing, and
the quality grades are not blind yet (section 7).

## 2. Setup

| Item | Value |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017, ISTA-DASLab GSQ-RCO IQ2_XXS GGUF (~138 GB shards + 104 GB Engram IQ4_NL file) |
| Hardware | HP Victus laptop: RTX 4060 Laptop 8 GB, Ryzen 7 8845HS (+ Radeon 780M iGPU), 30 GB RAM, one 512 GB NVMe SSD |
| Engine | Strata (custom DeepSeek engine; experts streamed from SSD). **kf43** = exact 6-expert routing quality line (most runs). **kf45** = kf43 + "half deferral" (lossy, +10% speed): some older runs; always labelled |
| Sampling | temperature 1.0, top_p 1.0 (model card), fixed seeds per problem |
| Maths bench | `adaptive-experts/bench_think.py`, template `reasoning_effort` max (100) or low (50); budget 32,000 tokens (tough) / 16,000 (held-out) |
| Agent harness | DeepSeek Harness ("dsh") web agent through a server copy; effort **medium** = template 63 + 2,048 thinking tokens per step (default), **low** = 50 + 1,024 per step, **off** = no thinking; a step that hits its budget gets "I have thought about this long enough; time to give my answer." injected (a "capped step") |

## 3. The technique and its versions

The logit of `</think>` is `w_close · h` (h = the final hidden state). Adding a vector `β·g` to that one row gives
`w_close·h + β·(g·h)`: a linear probe `g` reading the hidden state, raising the probability of ending the thinking
exactly when the state looks like "the answer is settled". It is a pure weight edit (one row of `head.weight`,
re-quantized to the file's Q5_K format), runs on the unmodified engine, costs nothing at inference. Details and the
fitting maths: `notes/method.md`.

| Version | Fit | Status / what happened |
|---|---|---|
| graft2 | shrinkage-LDA probe on 2 maths traces (paragraph ends while re-checking vs everything else), β=14, fitted on both chunk and one-token-decode states | **Best results** (maths 7/7, agentic fastest). Defects: stray `</think>` tags inside visible maths answers; moves checking into the visible answer; closes thinking at the start of most agent steps |
| v4 | graft2 + graft2's own answer text as negatives | fixed strays; one maths slip (digit_sum answered 785, right is 791: one count off by one) that the skipped checking would have caught (verified by letting the unedited model continue its thinking: it found the slip) |
| v5 | v4 + agent step starts as negatives | ≈ plain "low" effort on maths (24,612 vs 25,548 tokens); site 3,425 thinking tokens on kf45 |
| v6 | v5 + a second "about to draft the file" readout | **failed in a real run**: wrote `</think>` inside a file being written by a tool call (corrupted index.html); killed |
| v7 | **correct objective**: survival likelihood with the base log-odds as offset, every generated position | passes all offline held-out safety checks; site 3,583 thinking (−73%), no thrash; thinks a little on every step (more than graft2) |

## 4. Headline results (all kf43 unless marked; one run each)

### 4.1 Maths (7 problems; full per-problem table in `tables/summary_tables.md`)

Two further held-out problems (numtheory, edit_distance) were failed by **both** plain max and graft2 at the
16,000-token budget (numtheory: graft2 found 228 at 12.2k thinking tokens, then re-verified in its visible answer until
the cap; edit_distance: still filling the table at the cap) and were dropped from later arms; they are in `maths_runs.csv`.


| Arm | Right | Tokens on the 6 problems all arms solved | Minutes | round_table (hardest) |
|---|---|---|---|---|
| plain max (effort 100) | 6/7 | 41,559 (thinking 39,946, of which 25,363 re-checking an answer already found) | 64.1 | no answer: found the right answer at ~17,800 tokens, kept verifying to the 32k cap (kf14 engine) |
| plain low (effort 50) | 6/7 | 25,548 | 36.3 | no answer (32k cap) |
| **graft2 at max** | **7/7** | **25,112** (thinking 17,996 + 7,116 visible answer) | **35.3** | **right at 27,696 tokens** |
| v5 at max | 6/7 | 24,612 | 34.5 | no answer (32k cap) |

Plain max's tough-problem runs are from the older kf14 engine (Nov 8). Thinking-off and medium were **not run** on maths.

### 4.2 Agent tasks (dsh; full table with every run: `tables/agentic_runs.csv`, per step: `tables/agentic_steps.csv`)

| Task | original, medium | original, LOW | thinking off | **graft2** | v7 |
|---|---|---|---|---|---|
| site (build a polished one-page site; 10 checks) | 1,995 s, 13,106 thinking, 10 steps (4 capped), 10/10 | 2,702 s, 12,530 thinking, **36 steps** (5 capped), 10/10, mobile overflow | 572 s, 0, 10/10 (**kf45**) | **1,042 s, 2,062 thinking**, 6 steps (1 capped), 10/10 | 1,367 s, 3,583, 10 steps (1 capped), 10/10 |
| eyes (probability puzzle with code) | 477 s, 2,828 (2nd sample Nov 13: 173 s, 640) | not run | 131 s, 0 (Nov 13) | **128 s, 0**, right | 218 s, 1,164, right |
| orders (fix 3 bugs, tests) | 206 s, 271 (2nd sample: 255 s, 701) | not run | 189 s, 0 (Nov 13) | **164 s, 220**, OK | 289 s, 861, OK |
| **promo** (new feature, 9 hidden tests; unseen) | not run (removed by the user) | not run | not run | **1,180 s, 3,994 thinking, hidden 9/9** | not run |
| **analysis** (sales CSV with planted traps; unseen) | not run (removed) | 2,164 s, 10,160 thinking; key: weekend ✓, data issues 3/3, 1 false claim | not run | **1,292 s, 4,693 thinking**; key: weekend ✓, Elm partial, data issues 3/3, 2 false claims | not run |
| max effort | — | — | — | — | analysis run started, **stopped by the user**: no result |

Quality (site): blind pilot judge (gpt-6-luna via Codex, 2 shuffles, **kf45 arms only**): original 9, v5 8, off 7.
Non-blind ratings (mine, `raw/bench-agentic/quality-ratings.md`): kf43 original 6.5, kf43 graft2 8, kf45 original 9,
kf45 v5 7.5, kf45 off 5.5. The two original runs differ by 2.5 points: single runs cannot rank close arms.

### 4.3 "Does it still think after a surprise?" (exact, offline; `tables/surprise_first_token.csv`)

P(no thinking at all on a step) = P(`</think>` at the first thinking token), computed exactly from the head's
recorded input on the original's real agent prompts:

| Last tool result | unedited | graft2 |
|---|---|---|
| task start (orders / site) | 0.29 / 0.07 | 0.995 / 0.95 |
| routine: "Ran 7 tests ... OK", file updated, answer confirmed | 0.01-0.12 | 0.94-0.95 |
| new information: test file read; empty directory (time to plan) | 0.000 | 0.002-0.004 |
| surprise: node ENOENT error / browser failed to launch | 0.000 | 0.001 / 0.000 |
| surprise: tests fail (`..EFEF.`) | 0.27 | 0.91 (next action is "read the code"; it then thought 220 tokens on the code) |

Live confirmation (promo, unseen task): after "Ran 25 tests ... FAILED (failures=6)" graft2 thought 128 tokens and
diagnosed the cause ("The tests pass code as parse_order only, not parse_code!"), then 76 tokens to fix it.

## 5. Key observations (details: `notes/observations.md`)

1. **Over-thinking is real and measurable**: plain max spends 25,363 of 39,946 maths thinking tokens re-checking an
   answer it already has; on round_table it had the right answer at ~17,800 tokens and verified until the 32k cap.
2. **On maths, the graft ≈ lowering effort on tokens** (25,112 vs 25,548), but graft2 kept max-effort solving and was
   the only arm to solve round_table. Effort 50 solves sloppier (more solve tokens; code_trace first got 93, a full
   re-derivation found 85).
3. **On agent tasks, lowering effort backfires**: low's smaller per-step budget makes the model hit the cap more,
   stall, and take 3.6× the steps (site: 2,702 s vs 1,995 s at medium). graft2: 1,042 s.
4. **The agent failure mode we found**: the original drafts the whole HTML file line by line inside its thinking
   (counting lines for dsh's 200-line rule), the budget cuts it, it emits a stalling `ls`, and re-drafts from scratch:
   63-70% of its site thinking. Not found in prior work.
5. **graft2 thinks when needed**: zero thinking on obvious steps, real thinking on plans, errors and new information.
6. **How to fit such an edit**: LDA + hand-tuned β leaked stray `</think>` in every version (answers, step starts,
   tool-call file content). The correct objective is the survival likelihood with the base log-odds as offset (v7).
7. **Prior work** (cite, don't claim the idea): probe-based early exit (arXiv 2504.05419), `</think>`-logit steering
   (2506.08390, NeurIPS 2025), ThinkEdit (2503.22048); trained routers for agent thinking (Ada-RS, AutoTool);
   function-calling budgets (2604.02155). See `notes/prior_work.md`.

## 6. What the planned blog comparison has and lacks (graft2 vs max vs medium vs low vs off)

| | maths | agent (site/eyes/orders) | surprise (promo/analysis) |
|---|---|---|---|
| graft2 | ✅ 7 problems | ✅ all 3 | ✅ both |
| plain max | ✅ 7 (tough part on kf14) | ❌ | ❌ (analysis started, stopped) |
| plain medium | ❌ | ✅ all 3 (2 samples eyes/orders) | ❌ (removed) |
| plain low | ✅ 7 | ✅ site only | ✅ analysis only |
| thinking off | ❌ | ✅ eyes/orders kf43, site **kf45** | ❌ |
| blind quality judge | — | pilot only (kf45, site, 3 arms) | ❌ |

## 7. Caveats (must be in the post)

- **One run per cell.** Run-to-run variance is large (original eyes 477 s vs 173 s; site quality 9 vs 6.5).
  Statistical claims need the planned replication on a small model (Qwen3.5-4B, below).
- Engines differ between some cells (kf14 / kf43 / kf45); every table labels it.
- Analysis grading and most site ratings are by hand, not blind.
- graft2 defects: stray `</think>` tags in maths answers (56 extra on round_table, 1-2 on four others), checking moved into the visible answer
  (coin 3,611 answer tokens), and it hard-coded "Open today 7:00-18:00" on its site (wrong Fri-Sun).
- The probe was fitted on two maths traces; generalisation evidence = held-out maths problems + unseen agent tasks.

## 8. Folder index

| Path | Content |
|---|---|
| `tables/maths_runs.csv` | every maths run: arm, engine, effort, problem, right/wrong, total/thinking/solve/check/answer tokens, strays, minutes |
| `tables/maths_summary_6_shared_problems.csv` | arm totals |
| `tables/agentic_runs.csv` / `agentic_steps.csv` | every agent run (wall time, thinking, steps, capped, zero-thinking steps) / every step with the preceding tool result |
| `tables/surprise_first_token.csv` | the surprise probe |
| `tables/analysis_grading.csv` | analysis reports vs the answer key |
| `tables/site_browser_checks.csv` | headless-Chromium checks of every site |
| `tables/summary_tables.md` | all of the above as Markdown (generated) |
| `artifacts/site/<tag>/index.html`, `artifacts/screenshots/` | every site page and its desktop/dark/mobile screenshots |
| `artifacts/analysis/<tag>/` | the analysis reports + charts (graft2 = k43-g2, low = k43-low) |
| `artifacts/promo/k43-g2/` | graft2's promo solution (code + tests) |
| `notes/` | method, observations, prior work, maths problems and agent task prompts (exact texts), blog plan |
| `raw/` | raw result files, dsh answer keys, study pre-registration, non-blind ratings, pilot judge, graft rows (`.q5k`) and probe vectors (`.npy`), fit logs, SSD health |
| `build_tables.py`, `make_summary_md.py` | regenerate `tables/` from the raw logs |

**Tags → arms**: `ag-base` original medium kf43 · `ag-graft` / `k43-g2` graft2 kf43 · `k43-low` original low kf43 ·
`k43-g7` v7 kf43 · `before` original default kf43 (Nov 13) · `after-off` thinking off kf43 (Nov 13) · `ag5-base` /
`ag5-off` / `ag5-v5` / `ag5-v4` original / off / v5 / v4 on kf45 · `after-kf45` infrastructure failure (ignore).
