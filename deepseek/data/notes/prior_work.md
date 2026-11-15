# Prior work (checked Nov 14; cite these, do not claim the core idea as new)

| Work | What it does | Relation |
|---|---|---|
| Zhang et al. 2025, arXiv 2504.05419 | hidden-state probe predicts answer correctness; early exit of reasoning (−24% tokens) | same core idea (probe → stop) |
| Sheng et al., "On Reasoning Strength Planning", NeurIPS 2025, arXiv 2506.08390 | a direction in activations controls reasoning length via the `</think>` logit | closest mechanism (steer `</think>`) |
| ThinkEdit, arXiv 2503.22048 | edits attention output projections to fix too-short thinking | weight edit for reasoning length |
| NoWait (2506.08343), Chain of Draft (2502.18600), DoubtLess | suppress "wait" tokens / terse drafting prompts | decoding-time and prompt alternatives |
| First Try Matters (2510.08308), ROM (2603.22016), NEAT, EAT, SpecExit, LYNX | early-exit / overthinking studies and methods | same problem |
| Sun et al. 2025, arXiv 2507.12379 | probes detect arithmetic errors in hidden states | "doubt" signal idea (not used) |
| Ada-RS (2602.19519), AutoTool (2603.13348), CAST | trained routers deciding when an agent should think | selective thinking for agents, but trained |
| arXiv 2604.02155 | function-calling agents need ≤32 thinking tokens on most simple calls | supports "skip thinking on obvious steps" |
| Arditi et al. 2024, arXiv 2406.11717 (refusal direction) | removes a feature direction from weights | a "cousin": statistics-based weight edit, but removes everywhere |
| "Sparks of Weight-Space Programming" (MIT CSAIL, Nov 2 2026) | an LLM edits another model's weights directly (Qwen loop fix) | different mechanism (no data fit) |

What we found no prior description of (say "not found", not "first"):
- the agent "cap-thrash" (drafting the artefact inside thinking until a per-step budget cuts it, then re-drafting);
- a training-free one-row edit that gives state-dependent thinking in an agent loop (skip on obvious steps, think on
  errors/new information), measured exactly with the first-token probability;
- the survival-likelihood fit with the base log-odds as offset for a single-logit edit, and the stray-`</think>`
  failure modes of mean-based fits (inside answers, inside tool-call file content).
