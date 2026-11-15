# Method: the head-row graft, how it was fitted, and how it was checked

## The edit

The model's output layer (`head.weight`, vocab × 5120) turns the final normalised hidden state `h` into logits. The
`</think>` token (id 128822) has logit `l = w·h`. We replace that single row by `w + Δw`:

    l' = w·h + Δw·h

Only this one logit changes, so the new probability of closing the thinking is exact and cheap to compute offline:

    P'(</think>) = sigmoid( o + Δw·h ),   o = l − log(Σ_{other tokens} e^{logit})   (the base model's log-odds)

The row is written back in the file's own quantization (Q5_K) with ggml's quantizer (relative error of the edit after
re-quantization ≈ 4%) and shipped as an "overlay pack": a copy of `head.weight` with one row replaced, every other
tensor pointing at the original files. Runtime cost: zero (same engine, same speed).

## Fit used by graft2 (LDA + β)

- Data: the head's exact inputs `h` recorded while the unmodified model re-read two of its own maths traces
  (digit_sum, coin), on both engine paths (chunked prompt processing and one-token decode, which differ by 1-3 nats).
- Labels: "paragraph ends after the answer was found" (the model is only re-checking) vs everything else.
- Probe: shrinkage linear discriminant (closed form, shrinkage 0.2), scaled so negatives sit at 0 and positives at 1;
  Δw = β·g with β = 14 picked from an exact offline simulation of when the edited model would close.
- Script: `adaptive-experts/delete/graft_refit.py`. Row: `raw/graft-rows/head-row-graft2-14.q5k`, probe `graft2-g.npy`.

## What went wrong with LDA + β (v2-v6) and the correct objective (v7)

LDA separates class *means*; what matters is the *worst* position: one stray `</think>` anywhere corrupts output.
Every LDA version leaked a new failure: strays inside maths answers (graft2), closes at agent step starts (v4),
`</think>` written inside a tool call's file content (v6, real run, corrupted index.html).

v7 fits Δw by maximum likelihood of the behaviour we want, with the exact formula above, on **every position the model
generates** (thinking, visible answers, tool-call content), using the **survival likelihood**, because a run stops at
its first close:

    per episode:  Π_{rows before the stop window} (1 − p)  ×  (1 − Π_{rows in the window} (1 − p))

window = where stopping is right (maths: from the row where the answer is settled through the 3rd paragraph end after
it; agent steps: around the point where the thinking starts drafting the file). All other generated rows are pure
"don't close" (weight 1 each). L2 on Δw, L-BFGS, ~9 s per fit. Mistakes made on the way (all fixed): one positive
label per paragraph end (rewards 300 events, pays early-close risk everywhere); per-group normalised weights;
subsampling negatives (closes between the kept rows); regex answer marks jittering by a few tokens.
Script: `adaptive-experts/delete/graft_v7s.py`; log `raw/logs/graft_v7s.log`.

Held-out gates v7 passed (vs v5 as deployed): worst P(close while still solving) 0.009 (v5 0.19); checking kept on
plain-max traces 1% (v5 35%); strays per answer 0.002 (v5 0.41); strays in tool calls 0 (v6 had 2.66); step-start
closes mean 0.028 (unedited 0.007); decode path = chunk path.

## Evaluation protocol

- Maths: `bench_think.py`, fixed seeds per problem, temperature 1.0, budgets 32k/16k, graded by exact answer match
  (LaTeX-normalised). Phase split (`delete/phase_split.py`): "solve" = thinking before the correct answer first appears
  (code_trace/makespan: where it is *stated* as the result), "check" = thinking after it, "answer" = visible text.
- Agent tasks: dsh harness through a server copy with its own empty KV-slot folder per arm (no cache carry-over);
  automatic checks in `bench-agentic/run.py`; site also checked in headless Chromium (`quality_site.cjs`).
- Surprise probe: exact first-token P(`</think>`) on the original's recorded agent prompts (`delete/surprise.py`).
- Quality: blind pilot with gpt-6-luna through `codex exec` (outputs anonymised A-D, two shuffles, strict JSON
  schema; `raw/judge-pilot/judge.py`); otherwise hand grading against answer keys (`raw/bench-agentic/keys/`).
