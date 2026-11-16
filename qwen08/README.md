# Qwen3.5-0.8B output-head probe graft

This is the frozen **experimental three-row candidate** evaluated in post2.
It starts from unedited Qwen3.5-0.8B; it is not built on RikiyaT's checkpoint.
All 299 original tensors are byte-identical. A separate BF16 output matrix changes
only `</think>`(248069), `Wait`(13784), and ` Wait`(13428).

**Important limitation:** it is not a robust general loop fix. On MATH-500 it
scored 276/500 (55.2%), but 165 outputs reached 32,768 tokens. Of those,163 had
already closed thinking and then continued in the answer. The complete run has
1,348,952 extra closing tags across 407 responses. These are included in the
published outputs and token counts. Do not remove them before benchmarking.

- [Post2](https://probe-graft.vercel.app/2/)
- [Previous post](https://think-graft.vercel.app/)
- [Experimental GGUF](https://huggingface.co/vaibhavrpi/Qwen3.5-0.8B-probe-graft-GGUF)
- [Finished MATH analysis](publication/math500-analysis.json)
- [Capped-answer audit](publication/capped-answer-audit.json)

## Results and interpretation

| Full MATH-500; one selected answer per problem | Correct |
|---|---:|
| Unedited, original upstream evaluation sample 0 |35/500 (7.0%)|
| RikiyaT edit, independent HF rerun sample 0 |261/500 (52.2%)|
| Our output-head candidate, local seed 0 |276/500 (55.2%)|

These are **different serving engines/runs**, not a controlled same-engine gain.
The RikiyaT index is its strongest global index among4, selected at the user's
request. It is fixed for every problem, not best-of-four per question. The
observed difference is 3 percentage points; exploratory paired bootstrap95%CI
[-0.4,6.4]points, descriptive McNemar p=0.110. Neither proves superiority or
equivalence. The original paper's multi-sample baseline average was7.9%; the
7.0% above is the retrieved first-sample complete baseline.

Mean output length is 15,828.59 tokens over our500 outputs. On 20 shared published
reference first-sample texts, ours averages15,340.15 vs approximately 21,638.15
retokenized tokens (-29.1%). That saving is for those 20 prompts, not all 500.
GSM8K and IFEval are separate evaluations; do not infer their final scores from
incomplete progress. No test result was used as a probe-fitting example.

## Run the released model

Use a CUDA build of llama.cpp. The GGUF is approximately2.07GB and fits on an8GB GPU.

```bash
hf download vaibhavrpi/Qwen3.5-0.8B-probe-graft-GGUF \
  Qwen3.5-0.8B-BF16-probe-graft.gguf --local-dir models
llama-server -m models/Qwen3.5-0.8B-BF16-probe-graft.gguf \
  -ngl 99 -fa on -c 278528 -np 8 -b 256 -ub 256 --jinja --port 8200
```

Sampling: temperature 0.6, top-p0.95, top-k20, min-p0, presence1.5,
frequency 0, repeat1.0, seed 0 on **every** request, thinking enabled,
32,768 generated tokens. Sampler order: penalties -> top-k -> top-p -> temperature.
The penalty window is 33,792 tokens and includes the prompt in llama.cpp; vLLM's
presence penalty covers generated tokens. Equal seed numbers across engines do
not imply identical draws. The intended math prompt suffix is exactly:

```text
\nPlease reason step by step, and put your final answer within \boxed{}.
```

IFEval gets its original instruction prompt without that math suffix.

## Environment and independent reproduction

Recorded environment: Ubuntu26.04, RTX4060 Laptop8GB, CUDA13.2.78,
driver595.71.05. Exact package/compiler information is in
`publication/environment.json`. Generation used Python3.14.4, fitting/scoring
Python3.12.12, math-verify0.9.0. `requirements-eval.lock` records the environment;
`pyarrow` is additionally needed to download parquet datasets.

llama.cpp source baseline: `3549c274dc22001d9118713fa56a10085b85de3b`.
`tools/sparse-presence-penalty.patch` changes the penalty loop's iteration strategy,
not its arithmetic. The local100-case bit-exact check and 1,000 sampler checks are
included in `publication/numerical-checks.json`. This is not a batch-invariance proof.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-eval.lock pyarrow
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout 3549c274dc22001d9118713fa56a10085b85de3b
git -C llama.cpp apply ../tools/sparse-presence-penalty.patch
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build llama.cpp/build -j4
uv pip install --python .venv/bin/python -e llama.cpp/gguf-py
.venv/bin/python fetch_data.py
```

Use your own CUDA architecture when appropriate. That and hardware/library
differences can change sampled outputs; bitwise generation portability is not claimed.
Download metadata pins all dataset revisions in `data/dataset-download-manifest.json`.
The fetch script was checked against the experiment's stored data.

To generate a **new** reproduction without overwriting the published results:

```bash
export MODEL="$PWD/models/Qwen3.5-0.8B-BF16-probe-graft.gguf"
export LLAMA_SERVER="$PWD/llama.cpp/build/bin/llama-server"
export PYTHON="$PWD/.venv/bin/python"
./run_benchmark.sh
```

It saves new outputs in `runs/reproduction/`; rerunning it resumes by problem ID,
sample and model identity. It runs only the edited model. Complete unedited and
RikiyaT control evaluations would be additional experiments.

## Rebuild the exact learned edit

The base HF revision is `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`.
Convert that base with the pinned llama.cpp converter to BF16. Then:

```bash
export LLAMA_CPP="$PWD/llama.cpp"
export SRC="$PWD/models/Qwen3.5-0.8B-BF16.gguf"
.venv/bin/python build_head_recovery.py 4
```

This uses the published three learned probes in `loop-repair/`, with strengths
12(re-checking),24(loop-boundary stopping),4(loop-word suppression). Compare
the constructed tensors with `publication/model-manifest.json`. Whole-file hashes
also depend on base GGUF metadata; the released file's SHA256 is recorded there.

To **refit** from recorded fitting traces, expand `results/fit-traces.jsonl.gz`,
capture with the same library, then run `./rebuild_probe.sh`. This can occupy the
GPU; do not run it alongside a benchmark whose serving configuration you are
trying to keep fixed. CPU BLAS thread counts should also be recorded.

Parameter fitting uses80 MATH-training questions in `loop-repair/fit_ids.txt`.
The other16 were repeatedly used for development; they are not a pristine test.
Fit traces inherit a mixture of32k budgets(first4) and 16k(remaining92).
Labels are heuristic answer appearances and repeated-line patterns, not proof
of settled correctness. This is supervised probe fitting without gradient descent
or retraining the base model; "training-free" refers to the latter.

Fresh16 training checks scored 8/16 for the selected repair vs 9/16 for the previous
graft. Failed output-head and FFN variants are retained in summary files. The
first partial benchmark motivated repair, so the project is not a pristine
preregistered confirmatory study. Weights were frozen for the complete run.

## Saved results and cheap-detector diagnostic

`prepare_saved_inputs.py` expands both compressed result files without any model
call. Grade with `score08.py`; a cap is not automatically a failure. It reads
the last box after the last closing tag, or the whole text if thinking never
closed. None of 165 capped MATH outputs contains a correct complete boxed answer
anywhere;164 have no box and one has a wrong box.

The prefix-only8-gram diagnostic uses a4,096-token lookback, repetition density
over 128 tokens, threshold 0.8, and minimum position 512. It flags 165/165 capped
and 206/335 finished outputs. This is **coverage**, not independently measured
precision/recall or proof that terminating at the alert gives the right answer.
No detector-based intervention, calibrated penalty sweep, avg@8 study, or
systematic batch-invariance experiment has been completed in this release.

Prior work includes [RikiyaT's weight edit](https://rikiyat.github.io/blog/weight-space-programming/),
[the LP follow-up](https://nathan.rs/posts/weight-space-lp/),
[probe-guided stopping](https://arxiv.org/abs/2504.05419), and
[NoWait](https://arxiv.org/abs/2506.08343). Both internal edits and output probes
can be state-dependent; the intervention location is the distinction here.
