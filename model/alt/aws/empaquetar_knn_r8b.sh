#!/usr/bin/env bash
# R8b: codigo.tgz de SHA, vec-i7.tgz (los vectores del i7 con transformers 4.46.3 con los que se fijó la regla de escalado), las entradas
# fijas de R6b (DIR_R6B) y las etiquetas por logprobs (lp_todo = R7 + R7b, lp_r8, lp_muestra = R5, lp_confirma = R7). No sube nada.
# Uso: bash model/alt/aws/empaquetar_knn_r8b.sh SHA DIR_R6B VEC_I7_TGZ LP_TODO LP_R8 LP_MUESTRA LP_CONFIRMA DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; r6b=$2; vec=$3; lpt=$4; lp8=$5; lpm=$6; lpc=$7; sal=$8; mkdir -p "$sal"
export COPYFILE_DISABLE=1
git archive --format=tar.gz "$sha" -- model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py \
  model/l2/largos.py model/l2/modelo.py model/l2/evaluar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py sdk/python \
  ':(exclude)model/alt/resultados*' > "$sal/codigo.tgz"
for f in bateria.tgz nuevas.tgz fondo.jsonl fondo2.jsonl l3_fondo.jsonl l3_fondo2.jsonl l3_bateria.jsonl l3_nuevas.jsonl; do cp "$r6b/$f" "$sal/"; done
cp "$vec" "$sal/vec-i7.tgz"; cp "$lpt" "$sal/lp_todo.jsonl"; cp "$lp8" "$sal/lp_r8.jsonl"; cp "$lpm" "$sal/lp_muestra.jsonl"; cp "$lpc" "$sal/lp_confirma.jsonl"
for x in codigo vec-i7 bateria nuevas; do tar -tzf "$sal/$x.tgz" | grep -q '\._' && { echo "$x.tgz trae ._*" >&2; exit 1; }; done
c8=$(python3 -c "import sys; sys.path.insert(0, '.'); from model.alt.prequential import CRIT_R8B_SHA; print(CRIT_R8B_SHA)")
n() { wc -l < "$1" | tr -d ' '; }
git show "$sha:model/alt/aws/knn_r8b.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/knn-r8b-$sha#" -e "s#__N_FONDO__#$(n "$sal/fondo.jsonl")#" \
  -e "s#__N_FONDO2__#$(n "$sal/fondo2.jsonl")#" -e "s#__CRIT8B_SHA__#$c8#" > "model/alt/aws/lanzados/ud-knn-r8b-$sha.sh"
wc -c "model/alt/aws/lanzados/ud-knn-r8b-$sha.sh"; (cd "$sal" && shasum -a 256 ./*)
