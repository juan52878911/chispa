#!/usr/bin/env bash
# R4: llm.tgz (el cliente del paso 2, solo biblioteca estándar), filas-nuevas.jsonl (`llm_bateria nuevas`) y fondo2.jsonl
# (`llm_bateria fondo2`), más el user-data con el prefijo y los recuentos. No sube nada.
# Uso: bash model/alt/aws/empaquetar_llm_r4.sh SHA FILAS_NUEVAS FONDO2 DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; nu=$2; f2=$3; sal=$4; mkdir -p "$sal"
t=$(mktemp -d)
git archive "$sha" model/alt/llm_jevbench.py | tar -x -C "$t"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$sal/llm.tgz" -C "$t/model/alt" llm_jevbench.py 2>/dev/null || COPYFILE_DISABLE=1 tar -czf "$sal/llm.tgz" -C "$t/model/alt" llm_jevbench.py
cp "$nu" "$sal/filas-nuevas.jsonl"; cp "$f2" "$sal/fondo2.jsonl"
tar -tzf "$sal/llm.tgz" | grep -q '\._' && { echo "llm.tgz trae ._*" >&2; exit 1; }
n() { wc -l < "$1" | tr -d ' '; }
git show "$sha:model/alt/aws/llm_r4.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-879381269751/llm-r4-$sha#" \
  -e "s#__N_NUEVAS__#$(n "$sal/filas-nuevas.jsonl")#" -e "s#__N_FONDO2__#$(n "$sal/fondo2.jsonl")#" > "model/alt/aws/lanzados/ud-llm-r4-$sha.sh"
wc -c "model/alt/aws/lanzados/ud-llm-r4-$sha.sh"; (cd "$sal" && shasum -a 256 llm.tgz filas-nuevas.jsonl fondo2.jsonl)
