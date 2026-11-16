#!/bin/bash
# Start a NEW local evaluation, preserving the published frozen results.
set -euo pipefail
cd "$(dirname "$0")"
: "${MODEL:?Set MODEL to the downloaded experimental edited GGUF}"
: "${LLAMA_SERVER:?Set LLAMA_SERVER to the pinned llama-server binary}"
PORT=${PORT:-8200}
OUT=${OUT:-runs/reproduction}
mkdir -p "$OUT"
PYTHON=${PYTHON:-python3}
export MODEL_ID=$($PYTHON - "$MODEL" <<'PY'
import hashlib,sys
print('reproduction-'+hashlib.file_digest(open(sys.argv[1],'rb'),'sha256').hexdigest()[:12])
PY
)
"$LLAMA_SERVER" -m "$MODEL" -ngl 99 -fa on -c 278528 -np 8 -b 256 -ub 256 --jinja --port "$PORT" > "$OUT/server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<120;i++)); do
    kill -0 "$pid"
    if curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null; then ready=1; break; fi
    sleep 1
done
[[ "$ready" == 1 ]]
for dataset in math500 gsm8k ifeval; do
    FIXED_SEED=0 PAR=8 "$PYTHON" -u gen08.py "$PORT" edited "$OUT/$dataset.jsonl" "$dataset" all 1 32768
    "$PYTHON" score08.py "$OUT/$dataset.jsonl"
done
