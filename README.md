# think-graft: editing one row of an LLM's output layer to stop it overthinking

A training-free way to cut a reasoning model's thinking: add a linear probe to the `</think>` row of the output layer,
so the model's own hidden state decides when its thinking is done. No training, no gradients, no engine or harness
changes; the result is an ordinary weights file.

- **Blog post** (method, results, limitations): https://think-graft.vercel.app
- **Edited Qwen3.5-4B (GGUF, runs in stock llama.cpp):** https://huggingface.co/vaibhavrpi/Qwen3.5-4B-think-graft-GGUF

```
logit(</think>) = w·h  ->  w·h + β·(g·h)        P'(</think>) = σ(o + β·g·h),  o = base log-odds of </think>
```

`g` is a closed-form shrinkage-LDA probe that separates "paragraph ends after the right answer has already appeared
in the thinking" (the model is only re-checking) from every other thinking position. Because only one logit changes,
the edited model's stop probability at every recorded position is exact, so β is chosen by offline simulation, and
any trace of the original model is a valid trace of the edited one up to the point where the edit stops it.

## Results

| Model | Benchmark | Original | With the one-row edit |
|---|---|---|---|
| Qwen3.5-4B (llama.cpp) | MATH-500, 8 held-out problems | 8/8, 34,955 tokens | 8/8, 16,080 tokens (-54%) |
| Qwen3.5-4B | AIME 2025 I-7, I-9, II-8 (32k budget) | 0/3 (out of budget) | 2/3 |
| DeepSeek V4.1 Flash (763B, 2-bit, laptop) | Maths, 7 problems, max effort | 6/7, 41,559 tokens* | 7/7, 25,112 tokens* |
| DeepSeek V4.1 Flash | Agent: build a website (blind rating) | 8.2/10, 13,106 thinking | 9.5/10, 2,062 thinking |
| DeepSeek V4.1 Flash | Still thinks after errors / new info | P(skip) 0.000 | P(skip) <= 0.004 |

\* tokens on the 6 problems both solved. One run per cell; see the post for every number and caveat.

## Layout

| Folder | What |
|---|---|
| [`qwen/`](qwen/) | Full, reproducible pipeline for Qwen3.5-4B on one 8 GB GPU with upstream llama.cpp: trace generation, head-input capture (a small libllama tool), the fit, the held-out β simulation, writing the edited GGUF, live verification, benchmarks, exit-risk analysis. All results as JSONL. |
| [`deepseek/`](deepseek/) | DeepSeek V4.1 Flash: the fit and analysis scripts as run, the edited head row (Q5_K) and probe, the agent-task harness, prompts and answer keys, and every recorded result (tables, sites, screenshots, reports). |
| [`blog/`](blog/) | The blog post (static HTML). |

## Quick start (Qwen3.5-4B)

```bash
git clone https://github.com/nextgen-user/think-graft && cd think-graft/qwen
pip install numpy sympy pyarrow gguf     # or use gguf-py from your llama.cpp checkout
export LLAMA_CPP=~/llama.cpp              # with a CUDA build in build/bin
export QWEN_GGUF=~/models/Qwen3.5-4B-Q8_0.gguf
./run_all.sh fit      # traces + capture + held-out beta report
./run_all.sh edit     # write models/Qwen3.5-4B-Q8_0-think-graft.gguf
./run_all.sh bench    # original vs edited vs thinking off
```

## Related work

The idea that a reasoning model's hidden state knows when the answer is settled, and that a probe can end the
thinking early, is published: Zhang et al. 2025 (arXiv 2504.05419), Sheng et al., NeurIPS 2025 (arXiv 2506.08390),
ROM (arXiv 2603.22016). Trained "think less" variants include ThinkingCap, Thinkless (arXiv 2505.13379) and
BudgetThinker. What this repo adds: folding the probe into the `</think>` row so the edit is just a weights file, and
measurements on Qwen3.5-4B and on a 763B model running on a laptop, including agent tasks.

## License

Apache-2.0 for the code. The edited Qwen3.5-4B weights follow the base model's Apache-2.0 license.

Contact: [LinkedIn](https://www.linkedin.com/in/vaibhav-arduino-913938212/) · vaibhavarduino@gmail.com
