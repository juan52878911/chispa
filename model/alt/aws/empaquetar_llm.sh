#!/usr/bin/env bash
# llm.tgz: el cliente y el análisis del techo del LLM (solo biblioteca estándar), en la raíz del tarball.
# Uso: bash model/alt/aws/empaquetar_llm.sh [rama-o-sha] > llm.tgz
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
t=$(mktemp -d)
git archive "${1:-HEAD}" model/alt/llm_jevbench.py model/alt/llm_techo.py | tar -x -C "$t"
tar -czf - -C "$t/model/alt" llm_jevbench.py llm_techo.py
