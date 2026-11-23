#!/usr/bin/env bash
# R10-c, g5: el 4B por logprobs etiqueta el fondo de la batería v2 (6 tareas dev). Mismo ud que R7b/R9-m (llm_r7b.sh sin tocar; solo cambian el
# prefijo s3://…/r10-c-lp-SHA y N_REET) y el cliente llm_logprobs.py del sha de R5 (f85da01). El fondo se REORDENA para que quepa lo más útil
# (`few_shot_v2 ordenar`: baratas hasta 100 por clase, luego alternando por coste las caras y el resto) y se imprime lo que cabe en la ventana.
# No sube nada.
# Uso: bash model/alt/aws/empaquetar_r10_c_lp.sh SHA SHA_R5 FONDO DIR_SALIDA [VENTANA_S=720]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; r5=$2; fondo=$3; sal=$4; vent=${5:-720}; mkdir -p "$sal"
t=$(mktemp -d)
git archive "$r5" model/alt/llm_logprobs.py model/alt/llm_jevbench.py | tar -x -C "$t"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$sal/llm.tgz" -C "$t/model/alt" llm_logprobs.py llm_jevbench.py 2>/dev/null \
  || COPYFILE_DISABLE=1 tar -czf "$sal/llm.tgz" -C "$t/model/alt" llm_logprobs.py llm_jevbench.py
tar -tzf "$sal/llm.tgz" | grep -q '\._' && { echo "llm.tgz trae ._*" >&2; exit 1; }
git show "$sha:model/alt/few_shot_v2.py" > "$t/few_shot_v2.py"
python3 "$t/few_shot_v2.py" ordenar --fondo "$fondo" --salida "$sal/reetiqueta.jsonl" --ventana "$vent"
f="model/alt/aws/lanzados/ud-r10-c-lp-$sha.sh"
git show "$sha:model/alt/aws/llm_r7b.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/r10-c-lp-$sha#" -e "s#__N_REET__#$(wc -l < "$sal/reetiqueta.jsonl" | tr -d ' ')#" > "$f"
[ "$(wc -c < "$f")" -le 16384 ] || { echo "el ud pasa de 16 KB" >&2; exit 1; }
bash -n "$f"
wc -c "$f"; (cd "$sal" && shasum -a 256 llm.tgz reetiqueta.jsonl)
