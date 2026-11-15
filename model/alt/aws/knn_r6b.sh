#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# R6b: el kNN y la simulación de R6 con el soporte reetiquetado por logprobs (R7) frente al de JSON, en una c7i.2xlarge BAJO DEMANDA
# (solo CPU; tope duro CHISPA_MIN=40 min ≈ 0,24 USD). user-data (DLAMI GPU PyTorch
# AL2023, que arranca sin GPU; rol chispa-pruebas-ec2 sin ListBucket), --instance-initiated-shutdown-behavior terminate. Todo por nombre exacto.
# Lee:     $CHISPA_S3/{codigo.tgz,bateria.tgz,nuevas.tgz,vec-paso1.tgz,fondo.jsonl,fondo2.jsonl,l3_fondo.jsonl,l3_fondo2.jsonl,
#                      fondo-fuente.jsonl,fondo2-fuente.jsonl,l3_bateria.jsonl,l3_nuevas.jsonl,lp_reetiqueta.jsonl,lp_bateria.jsonl}
# (lp_bateria.jsonl = la muestra del paso 2 etiquetada por logprobs en R5; lp_reetiqueta.jsonl = R7)
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
#  1. vectores de e5-small @ 614241f6 (transformers 5.22.2) como en R6;
#  2. `knn_soporte_llm medir --criterio r6b --l3-alt lp_reetiqueta.jsonl` (CRIT6B, sha abajo): por pares, logprobs frente a JSON;
#  3. `prequential medir` con las etiquetas de logprobs encima de las de JSON (descriptivo, mismo CRIT_R6 que R6).
MIN=${CHISPA_MIN:-40}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-8}
E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3; TRANSFORMERS=5.22.2
CRIT6B_SHA=75922ecbf540a318f51a0bda97662a0871a8f19cee768075df7d0e59f6b34bc2
CRIT6_SHA=__CRIT6_SHA__
N_FONDO=__N_FONDO__; N_FONDO2=__N_FONDO2__
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-r6-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-r6}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r6 $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min hilos=$HILOS"
OUT=$BASE/res; FALLO=""; REPO=$BASE/repo
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
lineas() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
subir_y_apagar() {
  trap - EXIT; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  ( cd "$BASE" && tar -czf resultados.tgz res ) && s3cp "$BASE/resultados.tgz" "$CHISPA_S3/resultados.tgz" || echo "no se pudo subir resultados.tgz"
  s3cp "$BASE/log.txt" "$CHISPA_S3/log.txt" || echo "no se pudo subir log.txt"
  if [ "$LOCAL" != 1 ]; then
    echo listo > "$BASE/listo"; s3cp "$BASE/listo" "$CHISPA_S3/listo" || true
    for _ in $(seq $((ESPERA_MIN * 2))); do aws s3 cp --only-show-errors "$CHISPA_S3/verificado" "$BASE/verificado" 2>/dev/null && break; sleep 30; done
    shutdown -h now
  fi
}
trap subir_y_apagar EXIT; trap 'exit 143' TERM INT HUP
cierre() { echo "== listo $(date -u +%FT%TZ) fallo=${FALLO:-no} ($(( $(date +%s) - T0 )) s)"; exit 0; }
( while sleep 20; do [ "$(quedan_min)" -le $((4 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f "model.alt."; break; }; done ) &
sin_tiempo() { [ "$(quedan_min)" -le $((4 + ESPERA_MIN + $1)) ]; }

# --- 1. entorno (CPU)
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] || { fallo "ningún Python de la AMI importa torch"; estado entorno "FALLO sin torch"; cierre; }
  "$PY" -m pip install -q --no-cache-dir "transformers==$TRANSFORMERS" huggingface_hub numpy || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
fi
estado entorno "OK $("$PY" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)" 2>&1 | tail -1); $(nproc) vCPU"

# --- 2. código y entradas; criterios ANTES de calcular nada
echo "== 2. entradas $(date -u +%T)"
mkdir -p "$REPO" "$BASE/bat" "$BASE/nuevas" "$BASE/vec1"
for f in codigo.tgz bateria.tgz nuevas.tgz vec-paso1.tgz fondo.jsonl fondo2.jsonl l3_fondo.jsonl l3_fondo2.jsonl fondo-fuente.jsonl fondo2-fuente.jsonl \
         l3_bateria.jsonl l3_nuevas.jsonl lp_reetiqueta.jsonl lp_bateria.jsonl; do
  s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
tar -xzf "$BASE/codigo.tgz" -C "$REPO" && tar -xzf "$BASE/bateria.tgz" -C "$BASE/bat" && tar -xzf "$BASE/nuevas.tgz" -C "$BASE/nuevas" \
  && tar -xzf "$BASE/vec-paso1.tgz" -C "$BASE/vec1" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
rm -f "$BASE"/bat/._* "$BASE"/nuevas/._* "$BASE"/vec1/._*
[ "$(lineas "$BASE/fondo.jsonl")" = "$N_FONDO" ] && [ "$(lineas "$BASE/fondo2.jsonl")" = "$N_FONDO2" ] || { fallo "recuentos del fondo"; estado entradas "FALLO recuentos"; cierre; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
S4=$(cd "$REPO" && "$PY" -m model.alt.knn_soporte_llm criterio --cual r6b | cut -d' ' -f1); S6=$(cd "$REPO" && "$PY" -m model.alt.prequential criterio | cut -d' ' -f1)
[ "$S4" = "$CRIT6B_SHA" ] && [ "$S6" = "$CRIT6_SHA" ] || { fallo "criterios distintos de los congelados ($S4 / $S6)"; estado entradas "FALLO criterio"; cierre; }
estado entradas "OK fondo $N_FONDO + $N_FONDO2, l3_fondo2 $(lineas "$BASE/l3_fondo2.jsonl"), l3_nuevas $(lineas "$BASE/l3_nuevas.jsonl"), lp_reetiqueta $(lineas "$BASE/lp_reetiqueta.jsonl"); criterios ${S4:0:12} ${S6:0:12}"

# --- 3. vectores (transformers fijado)
echo "== 3. vectores $(date -u +%T)"
VEC=$BASE/vec; FONDOS="$BASE/fondo.jsonl,$BASE/fondo2.jsonl"
if [ "$LOCAL" = 1 ]; then export CHISPA_E5=${CHISPA_LOCAL_E5:?en local hace falta CHISPA_LOCAL_E5}; VMAX=(--max-fondo "${CHISPA_LOCAL_FONDO:-400}")
else
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('intfloat/multilingual-e5-small', revision='$E5_REV', local_dir='$BASE/e5', allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*','LICENSE*'])" \
    || { fallo "descarga de e5"; estado e5 "FALLO"; cierre; }
  export CHISPA_E5=$BASE/e5; VMAX=()
fi
export HF_HUB_OFFLINE=1
T1=$(date +%s)
( cd "$REPO" && "$PY" -m model.alt.knn_soporte_llm vectores --bateria "$BASE/bat,$BASE/nuevas" --fondo "$FONDOS" --salida "$VEC" --hilos "$HILOS" "${VMAX[@]}" ) \
  || { fallo "vectores"; estado vectores "FALLO"; cierre; }
estado vectores "OK en $(( $(date +%s) - T1 )) s"

# --- 4. R6b: kNN con soporte reetiquetado por logprobs frente a solo JSON (por pares)
echo "== 4. r6b $(date -u +%T) quedan $(quedan_min) min"
T1=$(date +%s)
( cd "$REPO" && "$PY" -m model.alt.knn_soporte_llm medir --bateria "$BASE/bat" --vec "$VEC" --fondo "$FONDOS" --l3 "$BASE/l3_fondo.jsonl,$BASE/l3_fondo2.jsonl" \
    --l3-alt "$BASE/lp_reetiqueta.jsonl" --fuente "$BASE/fondo-fuente.jsonl,$BASE/fondo2-fuente.jsonl" --salida "$OUT/knn-r6b.json" --semillas 10 --criterio r6b ) \
  && estado r6b "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json,sys; v=json.load(open(sys.argv[1]))['veredicto_r6b']; print(v['estado'], v['alt_menos_llm'])" "$OUT/knn-r6b.json")" \
  || { fallo "r6b medir"; estado r6b "FALLO"; }

# --- 5. simulación prequential con las etiquetas de logprobs encima (el último registro de un id manda): descriptivo
echo "== 5. preq-lp $(date -u +%T) quedan $(quedan_min) min"
T1=$(date +%s)
( cd "$REPO" && "$PY" -m model.alt.prequential medir --bateria "$BASE/bat,$BASE/nuevas" --vec "$VEC" --l3 "$BASE/l3_bateria.jsonl,$BASE/l3_nuevas.jsonl,$BASE/lp_bateria.jsonl,$BASE/lp_reetiqueta.jsonl" \
    --fondo "$FONDOS" --l3-fondo "$BASE/l3_fondo.jsonl,$BASE/l3_fondo2.jsonl,$BASE/lp_reetiqueta.jsonl" --vec-fondo "$VEC/fondo.npz" --salida "$OUT/prequential-lp.json" --semillas 10 ) \
  && estado preq "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json,sys; v=json.load(open(sys.argv[1]))['veredicto']; print(v.get('estado'))" "$OUT/prequential-lp.json")" \
  || { fallo "prequential lp"; estado preq "FALLO"; }
cierre
