#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
: "${LLAMA_CPP:?Set LLAMA_CPP to the pinned llama.cpp checkout}"
BUILD_DIR=${LLAMA_BUILD:-$LLAMA_CPP/build}
c++ -std=c++17 -O2 tools/capture_stats.cpp -I"$LLAMA_CPP/include" -I"$LLAMA_CPP/ggml/include" \
    -L"$BUILD_DIR/bin" -Wl,-rpath,"$BUILD_DIR/bin" -lllama -lggml-base -o tools/capture_stats
