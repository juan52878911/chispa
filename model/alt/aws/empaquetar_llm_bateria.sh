#!/usr/bin/env bash
# Paso 2: llm.tgz (el cliente, solo biblioteca estándar) en la raíz del tarball y el user-data con los recuentos de la muestra.
# Uso: bash model/alt/aws/empaquetar_llm_bateria.sh SHA DIR_MUESTRA DIR_FONDO DIR_SALIDA
#   escribe DIR_SALIDA/{llm.tgz,filas.jsonl,filas-q4.jsonl,fondo.jsonl} y model/alt/aws/lanzados/ud-llm-bateria-SHA.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; mu=$2; fo=$3; sal=$4; mkdir -p "$sal"
t=$(mktemp -d)
git archive "$sha" model/alt/llm_jevbench.py | tar -x -C "$t"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$sal/llm.tgz" -C "$t/model/alt" llm_jevbench.py 2>/dev/null || COPYFILE_DISABLE=1 tar -czf "$sal/llm.tgz" -C "$t/model/alt" llm_jevbench.py
cp "$mu/filas.jsonl" "$mu/filas-q4.jsonl" "$fo/fondo.jsonl" "$sal/"
n() { wc -l < "$1" | tr -d ' '; }
git show "$sha:model/alt/aws/llm_bateria.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-879381269751/llm-bateria-$sha#" \
  -e "s#__N_FILAS__#$(n "$sal/filas.jsonl")#" -e "s#__N_Q4__#$(n "$sal/filas-q4.jsonl")#" -e "s#__N_FONDO__#$(n "$sal/fondo.jsonl")#" \
  > "model/alt/aws/lanzados/ud-llm-bateria-$sha.sh"
tar -tzf "$sal/llm.tgz" | grep -q '\._' && { echo "llm.tgz trae ._*" >&2; exit 1; }
wc -c "model/alt/aws/lanzados/ud-llm-bateria-$sha.sh"; (cd "$sal" && shasum -a 256 llm.tgz filas.jsonl filas-q4.jsonl fondo.jsonl)
