#!/usr/bin/env bash
# R6: el user-data (con el sha de CRIT_R6, los recuentos del fondo y el prefijo del modelo int8) y las entradas, todo en la raíz. No sube nada.
# Uso: bash model/alt/aws/empaquetar_knn_r6.sh SHA DIR_BAT DIR_NUEVAS DIR_VEC_PASO1 DIR_FONDO DIR_FONDO2 L3_FONDO L3_BATERIA DIR_SALIDA [MOVIL_S3]
#   DIR_FONDO: fondo.jsonl y fondo-fuente.jsonl (paso 2); DIR_FONDO2: fondo2.jsonl y fondo2-fuente.jsonl (`llm_bateria fondo2`).
#   l3_fondo2.jsonl y l3_nuevas.jsonl salen de R4 y se suben aparte cuando R4 esté verificado. MOVIL_S3: prefijo con modelo-int8.chispa.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; bat=$2; nu=$3; vec=$4; fo=$5; f2=$6; l3f=$7; l3b=$8; sal=$9; movil=${10:-}; mkdir -p "$sal"
export COPYFILE_DISABLE=1
git archive --format=tar.gz "$sha" -- model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py \
  model/l2/largos.py model/l2/modelo.py model/l2/evaluar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py sdk/python \
  ':(exclude)model/alt/resultados*' > "$sal/codigo.tgz"
( cd "$bat" && tar -czf "$sal/bateria.tgz" ./*.jsonl manifiesto.json )
( cd "$nu" && tar -czf "$sal/nuevas.tgz" ./*.jsonl manifiesto.json )
( cd "$vec" && tar -czf "$sal/vec-paso1.tgz" ./*.npz )
t=$(mktemp -d); git archive "$sha" model/alt/aws/latencia_1vcpu | tar -x -C "$t"; ( cd "$t/model/alt/aws/latencia_1vcpu" && tar -czf "$sal/latencia.tgz" . )
cp "$fo/fondo.jsonl" "$fo/fondo-fuente.jsonl" "$f2/fondo2.jsonl" "$f2/fondo2-fuente.jsonl" "$sal/"; cp "$l3f" "$sal/l3_fondo.jsonl"; cp "$l3b" "$sal/l3_bateria.jsonl"
for x in codigo bateria nuevas vec-paso1 latencia; do tar -tzf "$sal/$x.tgz" | grep -q '\._' && { echo "$x.tgz trae ._*" >&2; exit 1; }; done
c6=$(git show "$sha:model/alt/prequential.py" > "$t/p.py" && cd "$(git rev-parse --show-toplevel)" && python3 -c "import sys; sys.path.insert(0, '.'); from model.alt.prequential import CRIT_R6_SHA; print(CRIT_R6_SHA)")
n() { wc -l < "$1" | tr -d ' '; }
git show "$sha:model/alt/aws/knn_r6.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/knn-r6-$sha#" -e "s#__N_FONDO__#$(n "$sal/fondo.jsonl")#" \
  -e "s#__N_FONDO2__#$(n "$sal/fondo2.jsonl")#" -e "s#__CRIT6_SHA__#$c6#" -e "s#__MOVIL_S3__#$movil#" > "model/alt/aws/lanzados/ud-knn-r6-$sha.sh"
wc -c "model/alt/aws/lanzados/ud-knn-r6-$sha.sh"; (cd "$sal" && shasum -a 256 ./*)
