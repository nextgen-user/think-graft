#!/bin/bash
# Blind judging of the study's outputs (codex exec, gpt-6-luna, read-only).  No GPU, production untouched.
export PATH="~/.config/nvm/versions/node/v24.15.0/bin:$PATH"
cd ~/.local/share/deepseek-download/adaptive-experts/bench-agentic && python3 judge.py "$@"
echo "judge done $(date +%H:%M)"
