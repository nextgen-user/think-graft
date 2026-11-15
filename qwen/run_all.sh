#!/bin/bash
# End-to-end reproduction on one 8 GB GPU with upstream llama.cpp (no patches).
#   LLAMA_CPP  = llama.cpp checkout with a CUDA build in $LLAMA_CPP/build/bin (llama-server, libllama)
#   QWEN_GGUF  = Qwen3.5-4B Q8_0 GGUF (e.g. unsloth/Qwen3.5-4B-GGUF)
# usage: ./run_all.sh [fit|edit|bench|all]
set -e; cd "$(dirname "$0")"
LLAMA_CPP=${LLAMA_CPP:-$HOME/llama.cpp}; export LLAMA_CPP; B=$LLAMA_CPP/build/bin
export QWEN_GGUF=${QWEN_GGUF:-models/Qwen3.5-4B-Q8_0.gguf}; OUT=${OUT:-models/Qwen3.5-4B-Q8_0-think-graft.gguf}
export PROBE=${OUT%.gguf}-probe.npy; BETA=${BETA:-14}; STAGE=${1:-all}
up() { for i in $(seq 120); do curl -sf localhost:$1/health >/dev/null && return 0; sleep 2; done; return 1; }
serve() { $B/llama-server -m $1 -ngl 99 -fa on -c 131072 -np 4 -ctk q8_0 -ctv q8_0 --jinja --port 8192 > logs/server.log 2>&1 & S=$!; up 8192; }
mkdir -p logs data results; [ -f data/math500.jsonl ] || ./fetch_data.sh
if [ $STAGE = fit ] || [ $STAGE = all ]; then
  # 1. the original model's own thinking on the 24 fit problems (12k cap)
  serve $QWEN_GGUF; python3 scripts/gen.py 8192 original-fit data/fit-traces.jsonl data/fit_ids.txt on 12000; kill $S; wait $S 2>/dev/null || true
  # 2. head inputs at every position (libllama tool, causal chunks of 1024)
  g++ -O2 -std=c++17 tools/capt.cpp -I$LLAMA_CPP/include -I$LLAMA_CPP/ggml/include -L$B -lllama -lggml-base -Wl,-rpath,$B -o tools/capt
  python3 scripts/capture2.py data/fit-traces.jsonl cap-fit
  # 3. held-out beta report (fit on half, simulate on the other half)
  python3 scripts/fit.py cap-fit data/fit-traces.jsonl | tee logs/beta-report.txt
fi
if [ $STAGE = edit ] || [ $STAGE = all ]; then
  # 4. refit on all traces and write the edited GGUF (one row of an added output.weight)
  python3 scripts/fit.py cap-fit data/fit-traces.jsonl $BETA $OUT
fi
if [ $STAGE = bench ] || [ $STAGE = all ]; then
  # 5. benchmark: original vs edited (and thinking off) on held-out problems, 32k budget
  for arm in edited original off; do
    m=$QWEN_GGUF; think=on; [ $arm = edited ] && m=$OUT; [ $arm = off ] && think=off
    serve $m; python3 scripts/gen.py 8192 $arm results/bench-$arm.jsonl data/medium_ids.txt $think 32000; kill $S; wait $S 2>/dev/null || true
  done
  python3 scripts/report.py results/bench-original.jsonl results/bench-edited.jsonl results/bench-off.jsonl
fi
