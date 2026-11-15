#!/usr/bin/env bash
# R6b: reutiliza las entradas fijas de R6 (DIR_R6: bateria, nuevas, vec-paso1, fondos, fuentes y etiquetas JSON), con codigo.tgz de SHA,
# más lp_reetiqueta.jsonl (R7) y lp_bateria.jsonl (R5, 4B por logprobs), y escribe el ud. No sube nada.
# Uso: bash model/alt/aws/empaquetar_knn_r6b.sh SHA DIR_R6 LP_REETIQUETA LP_BATERIA DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; r6=$2; lpr=$3; lpb=$4; sal=$5; mkdir -p "$sal"
export COPYFILE_DISABLE=1
git archive --format=tar.gz "$sha" -- model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py \
  model/l2/largos.py model/l2/modelo.py model/l2/evaluar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py sdk/python \
  ':(exclude)model/alt/resultados*' > "$sal/codigo.tgz"
for f in bateria.tgz nuevas.tgz vec-paso1.tgz fondo.jsonl fondo2.jsonl fondo-fuente.jsonl fondo2-fuente.jsonl l3_fondo.jsonl l3_fondo2.jsonl l3_bateria.jsonl l3_nuevas.jsonl; do
  cp "$r6/$f" "$sal/"; done
cp "$lpr" "$sal/lp_reetiqueta.jsonl"; cp "$lpb" "$sal/lp_bateria.jsonl"
tar -tzf "$sal/codigo.tgz" | grep -q '\._' && { echo "codigo.tgz trae ._*" >&2; exit 1; }
c6=$(python3 -c "import sys; sys.path.insert(0, '.'); from model.alt.prequential import CRIT_R6_SHA; print(CRIT_R6_SHA)")
n() { wc -l < "$1" | tr -d ' '; }
git show "$sha:model/alt/aws/knn_r6b.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/knn-r6b-$sha#" -e "s#__N_FONDO__#$(n "$sal/fondo.jsonl")#" \
  -e "s#__N_FONDO2__#$(n "$sal/fondo2.jsonl")#" -e "s#__CRIT6_SHA__#$c6#" > "model/alt/aws/lanzados/ud-knn-r6b-$sha.sh"
wc -c "model/alt/aws/lanzados/ud-knn-r6b-$sha.sh"; (cd "$sal" && shasum -a 256 ./*)
