#!/bin/bash
# Refit from the saved base traces. Only the80 fit problems enter parameter fitting.
set -euo pipefail
cd "$(dirname "$0")"
: "${SRC:?Set SRC to the pinned BF16 base GGUF}"
: "${LLAMA_CPP:?Set LLAMA_CPP to the pinned llama.cpp checkout}"
PYTHON=${PYTHON:-python3}
"$PYTHON" prepare_saved_inputs.py
./build_capture.sh
"$PYTHON" capture08.py results/fit-traces.jsonl cap-fit-v2
"$PYTHON" loop_repair_fit.py
"$PYTHON" fit_head_recovery.py
"$PYTHON" build_head_recovery.py 4
