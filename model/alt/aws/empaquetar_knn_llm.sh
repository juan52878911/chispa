#!/usr/bin/env bash
# Paso 3 (R1): codigo.tgz (model/ y sdk/python, sin informes), bateria.tgz, vec-paso1.tgz y el fondo etiquetado, todo en la raíz,
# más el user-data con el prefijo y el recuento. No sube nada.
# Uso: bash model/alt/aws/empaquetar_knn_llm.sh SHA DIR_BATERIA DIR_VEC_PASO1 DIR_FONDO L3_FONDO DIR_SALIDA
#   DIR_FONDO = la salida de `llm_bateria fondo` (fondo.jsonl, fondo-fuente.jsonl); L3_FONDO = l3_fondo.jsonl de la ronda bf16 del paso 2.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; bat=$2; vec=$3; fo=$4; l3=$5; sal=$6; mkdir -p "$sal"
export COPYFILE_DISABLE=1
git archive --format=tar.gz "$sha" -- model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py \
  model/l2/largos.py model/l2/modelo.py model/l2/evaluar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py sdk/python \
  ':(exclude)model/alt/resultados*' > "$sal/codigo.tgz"
( cd "$bat" && tar -czf "$sal/bateria.tgz" ./*.jsonl manifiesto.json )
( cd "$vec" && tar -czf "$sal/vec-paso1.tgz" ./*.npz )
cp "$fo/fondo.jsonl" "$fo/fondo-fuente.jsonl" "$sal/"; cp "$l3" "$sal/l3_fondo.jsonl"
for t in codigo bateria vec-paso1; do tar -tzf "$sal/$t.tgz" | grep -q '\._' && { echo "$t.tgz trae ._*" >&2; exit 1; }; done
n=$(wc -l < "$sal/fondo.jsonl" | tr -d ' ')
git show "$sha:model/alt/aws/knn_llm.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-879381269751/knn-llm-$sha#" -e "s#__N_FONDO__#$n#" \
  > "model/alt/aws/lanzados/ud-knn-llm-$sha.sh"
wc -c "model/alt/aws/lanzados/ud-knn-llm-$sha.sh"; (cd "$sal" && shasum -a 256 codigo.tgz bateria.tgz vec-paso1.tgz fondo.jsonl l3_fondo.jsonl fondo-fuente.jsonl)
