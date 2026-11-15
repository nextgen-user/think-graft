# DeepSeek V4.1 Flash (763B): the one-row `</think>` edit on a laptop

DeepSeek V4.1 Flash (mixture of experts, IQ2_XXS, 243 GB) running on an RTX 4060 Laptop (8 GB) + 30 GB RAM, with
experts streamed from the SSD through a modified build of [Strata](https://github.com/Niko1221/Strata).

**Name used in these files:** the edited model in the blog post is called **graft2** here (its internal version name).
Other versions that appear in the data: **v4** = graft2 + the model's own answer text as negatives (removes stray
`</think>` tags in answers), **v5** = v4 + agent step starts as negatives, **v7** = fitted with the survival likelihood
instead of LDA + β (`scripts/fit_survival_likelihood.py`). The post reports graft2.

## What's here

| Path | Content |
|---|---|
| `edit/think-row-beta14.q5k` | The edited `</think>` head row (token 128822), Q5_K bytes as written into the head tensor |
| `edit/probe.npy` | The probe `g` (β = 14) |
| `scripts/fit_probe.py` | The graft2 fit: shrinkage LDA on head inputs from two maths traces, both engine paths; exact β simulation; Q5_K re-quantization of the row |
| `scripts/fit_probe_answer_negatives.py` | v4: adds the model's own answer text as negatives (the stray-tag fix) |
| `scripts/fit_survival_likelihood.py` | v7: maximum survival likelihood with the base log-odds as offset, over every generated position |
| `scripts/beta_sweep.py` | Exact offline β sweep |
| `scripts/surprise_first_token.py` | Exact P(skip thinking) on recorded agent prompts (the "does it still think when surprised" table) |
| `scripts/phase_split.py` | Solve / re-check / answer token split of every maths run |
| `scripts/counterfactual_continue.py` | Continue an edited run's thinking with the original model, to test whether skipped checking would have caught an error |
| `scripts/bench_think.py` | The maths benchmark (problems, seeds, budgets, grading) |
| `bench/` | The agent-task harness: task generator (`make_tasks.py`), checks (`run.py`, `quality_site.cjs`, `recheck_site.py`), hidden tests, answer keys, the pre-registered study plan |
| `data/` | Every recorded result: `README.md` (start here), `tables/` (CSV + Markdown), `notes/` (method, observations, prior work, exact task prompts), `artifacts/` (all generated sites with screenshots, analysis reports, promo code), `raw/` (result JSONL, fit logs, judge pilot) |

## Reproducing

The scripts are included **as run**, for reference. They read hidden-state captures produced by capture hooks in my
Strata build (`DS4_DUMP_XN`: the head's exact normalized input at every scored position) and import helpers from that
workbench, so they won't run against upstream Strata without that hook. The method itself is engine-independent; the
[`qwen/`](../qwen/) folder reproduces it end to end with upstream llama.cpp.

To apply the edit to a DeepSeek V4.1 Flash GGUF, replace row 128822 of `output.weight`/`head.weight` with the bytes in
`edit/think-row-beta14.q5k` (Q5_K, one row). Calibrate on the path that runs the model: on this engine, chunked
prefill and one-token decode differed by 1-3 nats on this logit, so the probe was fitted on both.

Sampling for every DeepSeek run: temperature 1.0, top-p 1.0, fixed seed per problem; maths budgets 32k (tough) and
16k (held-out); agent harness at medium effort (2,048 thinking tokens per step).
