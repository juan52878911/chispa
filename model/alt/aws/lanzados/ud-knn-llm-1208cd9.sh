#!/bin/bash
CHISPA_S3=s3://chispa-pruebas-879381269751/knn-llm-1208cd9
# Paso 3 (R1): kNN-e5 con soporte etiquetado por el LLM (fondo del paso 2, etiquetas de Qwen3-4B bf16) frente al kNN con soporte de oro,
# en una c7i.2xlarge BAJO DEMANDA (solo CPU; tope duro CHISPA_MIN=40 min ≈ 0,24 USD). user-data (DLAMI GPU PyTorch AL2023, que arranca
# igual sin GPU; rol chispa-pruebas-ec2 sin ListBucket), con --instance-initiated-shutdown-behavior terminate. Todo por nombre exacto.
# Lee:     $CHISPA_S3/{codigo.tgz,bateria.tgz,vec-paso1.tgz,fondo.jsonl,l3_fondo.jsonl,fondo-fuente.jsonl}
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
# 1) vectores de e5-small (intfloat/multilingual-e5-small @ 614241f6, MIT) de la batería y del fondo en esta máquina;
# 2) `knn_soporte_llm medir`: curvas por k y por soporte, comparación pareada y veredicto con el criterio congelado (sha256 abajo).
MIN=${CHISPA_MIN:-40}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-8}
E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3; TRANSFORMERS=5.22.2
CRIT_SHA=5e0c38a679aed8f064cbc10e5df271075d2c5a0b22c8ad08ce8a99508aebafb9
N_FONDO=8137
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-knnllm-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-knnllm}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque knn-llm $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min hilos=$HILOS"
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
( while sleep 20; do [ "$(quedan_min)" -le $((4 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f model.alt.knn_soporte_llm; break; }; done ) &

# --- 1. entorno (CPU)
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] || { fallo "ningún Python de la AMI importa torch"; estado entorno "FALLO sin torch"; cierre; }
  "$PY" -m pip install -q --no-cache-dir "transformers==$TRANSFORMERS" huggingface_hub numpy || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
fi
estado entorno "OK $("$PY" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'hilos', torch.get_num_threads())" 2>&1 | tail -1); $(nproc) vCPU"

# --- 2. código y entradas por nombre exacto; imports y criterio ANTES de calcular nada
echo "== 2. entradas $(date -u +%T)"
mkdir -p "$REPO" "$BASE/bat" "$BASE/vec1"
for f in codigo.tgz bateria.tgz vec-paso1.tgz fondo.jsonl l3_fondo.jsonl fondo-fuente.jsonl; do s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
tar -xzf "$BASE/codigo.tgz" -C "$REPO" && tar -xzf "$BASE/bateria.tgz" -C "$BASE/bat" && tar -xzf "$BASE/vec-paso1.tgz" -C "$BASE/vec1" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
rm -f "$BASE"/bat/._* "$BASE"/vec1/._*
[ "$(lineas "$BASE/fondo.jsonl")" = "$N_FONDO" ] && [ "$(lineas "$BASE/l3_fondo.jsonl")" = "$N_FONDO" ] && [ "$(lineas "$BASE/fondo-fuente.jsonl")" = "$N_FONDO" ] \
  || { fallo "recuentos del fondo distintos de $N_FONDO"; estado entradas "FALLO recuentos"; cierre; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
SHA=$(cd "$REPO" && "$PY" -m model.alt.knn_soporte_llm criterio | cut -d' ' -f1)
[ "$SHA" = "$CRIT_SHA" ] || { fallo "el criterio del código ($SHA) no es el congelado ($CRIT_SHA)"; estado entradas "FALLO criterio"; cierre; }
estado entradas "OK $N_FONDO fondo, $(ls "$BASE"/bat/*.jsonl | wc -l) tareas, criterio ${CRIT_SHA:0:12}"

# --- 3. e5 por revisión fijada y vectores
echo "== 3. vectores $(date -u +%T)"
VMAX=(); VSIN=(); VEC=$BASE/vec
if [ "$LOCAL" = 1 ]; then VMAX=(--max-fondo "${CHISPA_LOCAL_FONDO:-300}"); VSIN=(--sin-bateria); mkdir -p "$VEC"; cp "$BASE"/vec1/*.npz "$VEC/"
  export CHISPA_E5=${CHISPA_LOCAL_E5:?en local hace falta CHISPA_LOCAL_E5}
else
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('intfloat/multilingual-e5-small', revision='$E5_REV', local_dir='$BASE/e5', allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*','LICENSE*'])" \
    || { fallo "descarga de e5"; estado e5 "FALLO"; cierre; }
  export CHISPA_E5=$BASE/e5
fi
export HF_HUB_OFFLINE=1
T1=$(date +%s)
( cd "$REPO" && "$PY" -m model.alt.knn_soporte_llm vectores --bateria "$BASE/bat" --fondo "$BASE/fondo.jsonl" --salida "$VEC" --hilos "$HILOS" "${VMAX[@]}" "${VSIN[@]}" ) \
  || { fallo "vectores"; estado vectores "FALLO"; cierre; }
estado vectores "OK en $(( $(date +%s) - T1 )) s"

# --- 4. medición
echo "== 4. medir $(date -u +%T) quedan $(quedan_min) min"
T1=$(date +%s)
( cd "$REPO" && "$PY" -m model.alt.knn_soporte_llm medir --bateria "$BASE/bat" --vec "$VEC" --fondo "$BASE/fondo.jsonl" --l3 "$BASE/l3_fondo.jsonl" \
    --fuente "$BASE/fondo-fuente.jsonl" --vec-oro "$BASE/vec1" --salida "$OUT/knn-llm.json" --semillas 10 ) || { fallo "medir"; estado medir "FALLO"; cierre; }
estado medir "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json,sys; v=json.load(open(sys.argv[1]))['veredicto']; print(v['estado'], v['agrupada_llm'], v['llm_menos_oro'], v['hatecheck_llm'])" "$OUT/knn-llm.json")"
cierre
