# Qwen3.5-4B: the one-row `</think>` edit, end to end

Everything here runs on one 8 GB GPU with upstream llama.cpp (no patches). `run_all.sh` chains the steps; each step
can also be run on its own.

## Requirements

- llama.cpp with a CUDA build (`$LLAMA_CPP/build/bin/llama-server`, `libllama`), and its `gguf-py`.
- Qwen3.5-4B Q8_0 GGUF (`$QWEN_GGUF`), e.g. from unsloth/Qwen3.5-4B-GGUF.
- Python 3.10+ with numpy, sympy (grading) and pyarrow (only for `fetch_data.sh`).

## Steps

| Step | Script | What it does |
|---|---|---|
| 0 | `fetch_data.sh` | MATH-500 (HuggingFaceH4) and AIME 2025 (MathArena) into `data/` |
| 1 | `scripts/gen.py` | The original model's thinking on the 24 fit problems (`data/fit_ids.txt`), through llama-server with the model's own chat template; keeps prompt and generated token ids |
| 2 | `tools/capt.cpp` + `scripts/capture2.py` | Head inputs (final normed hidden state) at every position: embeddings on, pooling none, causal chunks of 1,024 with the KV cache kept |
| 3 | `scripts/fit.py CAPDIR TRACES` | Labels, shrinkage-LDA probe, and the held-out β report (fit on half the traces, exact simulation on the other half) |
| 4 | `scripts/fit.py CAPDIR TRACES 14 OUT.gguf` | Refit on all traces; write a copy of the GGUF with an added `output.weight` (the tied embedding with the `</think>` row edited, Q8_0) |
| 5 | `scripts/verify.py` | Live check: original vs edited vs offline prediction at chosen positions |
| 6 | `scripts/gen.py`, `scripts/report.py` | Benchmarks (thinking on, edited, thinking off) and the summary table |
| 7 | `scripts/risk.py` | Exit-risk search: on original traces, where would the edited model stop, and what would it answer there? |

## Results in `results/`

| File | Content |
|---|---|
| `fit-traces-original.jsonl` | The 24 fit traces (12k cap) |
| `math500-original.jsonl`, `math500-edited.jsonl` | Held-out MATH-500 runs (32k budget) |
| `aime-medium-{original,edited,thinking-off}.jsonl` | AIME 2025 I-7, I-9, II-8 |
| `aime-hard-original.jsonl` | AIME 2025 I-10, I-11, I-13, II-10 (original only; all hit 32k) |
| `deepseek-problems-edited.jsonl` | The DeepSeek maths set on Qwen (too hard at 32k; the edit did not stop early) |
| `verify-*.json` | Live vs offline `</think>` probabilities |
| `exit-risk-heldout.json` | Exit-risk analysis on 6 held-out traces |
| `hero-trace-intermediate_algebra-2046.json` | Per-token data behind the post's Figure 1 |

Each JSONL row holds the arm, problem id, expected answer, prompt and generated token ids, text, token count and time.
Sampling: temperature 0.6, top-p 0.95, top-k 20, fixed seed per problem (md5 of the problem id). Server: 4 slots,
q8_0 KV cache (needed to fit 4 x 32k in 8 GB; the same for every arm).

## Notes

- Captured hidden states differ by a median 2% between batch sizes (kernel numerics); the 1,024-chunk capture
  reproduces the model's own sampling log-probability (-0.18 nats per token).
- Traces that hit the 12k cap are still used for fitting when the right answer appears in the thinking: everything
  after it is re-checking.
