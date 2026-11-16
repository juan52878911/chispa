#!/usr/bin/env bash
# Paso 5 (R2): llm.tgz (el cliente, solo biblioteca estándar) y filas.jsonl (la misma muestra del paso 2), más el user-data. No sube nada.
# Uso: bash model/alt/aws/empaquetar_llm_pequeno.sh SHA DIR_MUESTRA DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; mu=$2; sal=$3; mkdir -p "$sal"
t=$(mktemp -d)
git archive "$sha" model/alt/llm_jevbench.py | tar -x -C "$t"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$sal/llm.tgz" -C "$t/model/alt" llm_jevbench.py 2>/dev/null || COPYFILE_DISABLE=1 tar -czf "$sal/llm.tgz" -C "$t/model/alt" llm_jevbench.py
cp "$mu/filas.jsonl" "$sal/"
tar -tzf "$sal/llm.tgz" | grep -q '\._' && { echo "llm.tgz trae ._*" >&2; exit 1; }
git show "$sha:model/alt/aws/llm_pequeno.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-879381269751/llm-pequeno-$sha#" \
  -e "s#__N_FILAS__#$(wc -l < "$sal/filas.jsonl" | tr -d ' ')#" > "model/alt/aws/lanzados/ud-llm-pequeno-$sha.sh"
wc -c "model/alt/aws/lanzados/ud-llm-pequeno-$sha.sh"; (cd "$sal" && shasum -a 256 llm.tgz filas.jsonl)
