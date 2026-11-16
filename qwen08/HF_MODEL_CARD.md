---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
language:
- en
tags:
- gguf
- reasoning
- weight-editing
- experimental
---

# Qwen3.5-0.8B: experimental output-head probe graft

A BF16 GGUF built from **unedited Qwen3.5-0.8B**, not RikiyaT's edited model.
A separate output matrix changes only 3 rows: `</think>`, `Wait`, and ` Wait`.
All 299 original tensors remain byte-identical. The three fitted probes are
folded into those rows; inference needs no probe callback or extra forward pass.

**This is a research artifact with a substantial known failure.** On the full
MATH-500 run it scored 276/500 (55.2%), but 165 responses reached 32,768 tokens.
163 of those had already closed thinking and continued in the answer. Extra
`</think>` tags occur in 407 responses, totaling 1,348,952 extra tags. This is not
a robust drop-in loop fix. The original traces and failure audit are published.

## Results

| MATH-500, one selected answer per problem | Accuracy |
|---|---:|
| Unedited original upstream sample 0 |7.0%|
| RikiyaT independent rerun, strongest global index 0 |52.2%|
| This candidate, local seed 0 |55.2%|

Different engines and runs; not a controlled causal comparison. The3-point
difference has exploratory paired-bootstrap95%CI[-0.4,6.4]points, descriptive
McNemar p=0.110. This does not establish superiority or equivalence. RikiyaT's
reference index was selected as the globally strongest of4, fixed for all problems.

Our mean total output length is 15,828.59 over 500 problems. On 20 shared published
first-sample texts:15,340.15 vs approximately 21,638.15 retokenized reference
tokens(-29.1%). That token comparison is limited to20 prompts.

## Use

```bash
hf download vaibhavrpi/Qwen3.5-0.8B-probe-graft-GGUF \
  Qwen3.5-0.8B-BF16-probe-graft.gguf --local-dir .
llama-server -m Qwen3.5-0.8B-BF16-probe-graft.gguf \
  -ngl 99 -fa on -c 278528 -np 8 -b 256 -ub 256 --jinja --port 8200
```

The file is approximately2.07GB. Untying the shared embedding/head adds roughly
0.51GB compared with the base GGUF; transformer compute is unchanged.

Evaluation used temperature 0.6, top-p0.95, top-k20, min-p0, presence penalty1.5,
frequency 0, repetition 1.0,32,768 output tokens, and seed 0 on every request.
Explicit order: penalties -> top-k -> top-p -> temperature. These reproduce the
reference's sampling values, not a claim to be the current general Qwen recommendation.

## Fit and reproducibility

Base HF revision: `2fc06364715b967f1860aea9cf38778875588b17`.
80 MATH-training problems fit the probes;16 separate training questions select
candidate strengths. No MATH-500/GSM8K-test/IFEval examples enter parameter fitting.
The repeatedly used development16 are not an independent test. On a fresh16
training check this candidate scored 8/16 vs previous graft9/16.

The stop row receives12 times the re-checking probe plus24 times a loop-boundary
probe. The two Wait rows receive minus 4 times an all-token loop probe. Labels are
heuristics; the model does not have a guaranteed "answer settled" detector.
There is no hard phase gate, explaining why unwanted closing tags remain possible.

Recorded setup: RTX4060 Laptop8GB, Ubuntu26.04, CUDA13.2.78, driver595.71.05;
llama.cpp base commit `3549c274dc22001d9118713fa56a10085b85de3b` plus an arithmetic-
identical sparse presence-penalty patch. Python and scorer versions, raw results,
fitting scripts, probes, exact parameters, dataset revisions and checks are in
the repository. Same seeds across engines do not guarantee identical draws.

- [Post2](https://probe-graft.vercel.app/2/)
- [Code, raw MATH outputs and fitting recipe](https://github.com/nextgen-user/think-graft/tree/main/qwen08)
- [Previous 4B/DeepSeek post](https://think-graft.vercel.app/)

This differs from an internal feed-forward edit. It is related to published
probe-guided stopping and token-suppression methods; see the post for prior work.
