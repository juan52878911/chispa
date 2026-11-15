#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# R7: Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f por LOGPROBS (llm_logprobs.py, el mismo código y llm.tgz que R5: etiqueta de un carácter,
# max_tokens=1, top_logprobs=20, relleno por prompt_logprobs) con vLLM 0.33.0 --max-logprobs 20. g5.xlarge BAJO DEMANDA, tope duro
# CHISPA_MIN=50 min (≈ 0,84 USD). user-data (DLAMI GPU PyTorch AL2023, rol chispa-pruebas-ec2 sin ListBucket), terminate al apagar.
# Lee:     $CHISPA_S3/{llm.tgz,filas-confirma.jsonl,reetiqueta.jsonl}
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
# (1) filas-confirma.jsonl: el test de la batería fuera de la muestra del paso 2 (3 535, intercalado por tarea y barajado) -> lp_confirma.jsonl,
#     con un tope de CONF_MIN minutos (CRIT7 se mide sobre las respondidas).
# (2) reetiqueta.jsonl, por prioridad: test de lexboe y fakenews-es, fondo de R1 (primero las tareas baratas, luego massive-esc y massive-es)
#     y fondo2 -> lp_reetiqueta.jsonl hasta el tope; lo etiquetado queda aunque se corte.
MIN=${CHISPA_MIN:-50}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-32}; CONF_MIN=${CHISPA_CONF_MIN:-24}; VLLM=${CHISPA_VLLM:-0.33.0}
REV=cdbee75f17c01a7cc42f958dc650907174af0554
N_CONF=__N_CONF__; N_REET=__N_REET__
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-r7-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-r7}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque llm-r7 $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min hilos=$HILOS"
OUT=$BASE/res; FALLO=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
parar() { pkill -TERM -f "$1" 2>/dev/null; for _ in $(seq 30); do pgrep -f "$1" > /dev/null || return 0; sleep 2; done; pkill -KILL -f "$1" 2>/dev/null; sleep 3; }
subir_y_apagar() {
  trap - EXIT; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  for l in "$BASE"/vllm-*.log; do [ -f "$l" ] && tail -n 200 "$l" > "$OUT/$(basename "$l" .log)-cola.log"; done
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
( while sleep 30; do [ "$(quedan_min)" -le $((5 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f llm_logprobs; pkill -TERM -f vllm.entrypoints; break; }; done ) &
# cliente con tope: corre hasta acabar, hasta que queden $1 minutos o hasta que pasen $2 minutos; lo escrito queda
cliente() {
  local queda=$1 dura=$2; shift 2; local t=$(date +%s)
  "$VPY" "$BASE/codigo/llm_logprobs.py" "$@" & local p=$!
  while kill -0 $p 2>/dev/null; do
    if [ "$(quedan_min)" -le "$queda" ] || [ $(( ($(date +%s) - t) / 60 )) -ge "$dura" ]; then echo "== corte por tiempo"; kill -TERM $p; sleep 5; kill -KILL $p 2>/dev/null; break; fi
    sleep 10
  done; wait $p 2>/dev/null
}
urls() { for _ in $(seq "$2"); do printf -- '--url %s ' "$1"; done; }
validas() { [ -f "$1" ] || { echo 0; return; }; grep -c '"valida": true' "$1"; }
lineas() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

# --- 1. entorno: venv con vLLM fijado (el del paso 2)
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}; VPY=$PY
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] && "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado entorno "FALLO"; cierre; }
  "$PY" -m venv /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" huggingface_hub || { fallo "pip install vllm==$VLLM"; estado entorno "FALLO pip"; cierre; }
  VPY=/opt/vllm/bin/python
  estado entorno "OK vllm $($VPY -c 'import vllm; print(vllm.__version__)' 2>/dev/null); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
fi

# --- 2. entradas por nombre exacto; autoprueba = primera fila de cada tarea de las dos listas
echo "== 2. entradas $(date -u +%T)"
for f in llm.tgz filas-confirma.jsonl reetiqueta.jsonl; do s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
mkdir -p "$BASE/codigo" && tar -xzf "$BASE/llm.tgz" -C "$BASE/codigo" || { fallo "llm.tgz"; cierre; }
[ -f "$BASE/codigo/llm_logprobs.py" ] && [ -f "$BASE/codigo/llm_jevbench.py" ] || { fallo "llm.tgz sin llm_logprobs.py o llm_jevbench.py"; estado entradas "FALLO cliente"; cierre; }
[ "$(lineas "$BASE/filas-confirma.jsonl")" = "$N_CONF" ] && [ "$(lineas "$BASE/reetiqueta.jsonl")" = "$N_REET" ] || { fallo "recuentos distintos de $N_CONF/$N_REET"; estado entradas "FALLO recuentos"; cierre; }
"$PY" -c "
import json,sys
v={}
for p in sys.argv[1:3]:
    for l in open(p):
        r=json.loads(l); v.setdefault(r['conjunto'], l)
open(sys.argv[3],'w').write(''.join(v[k] for k in sorted(v)))" "$BASE/filas-confirma.jsonl" "$BASE/reetiqueta.jsonl" "$BASE/auto.jsonl"
estado entradas "OK $N_CONF confirma, $N_REET reetiqueta, $(lineas "$BASE/auto.jsonl") de autoprueba"
MAX=(); [ "$LOCAL" = 1 ] && MAX=(--max "${CHISPA_LOCAL_MAX:-2}")

# --- 3. modelo y servidor
echo "== 3. modelo $(date -u +%T) quedan $(quedan_min) min"
if [ "$LOCAL" = 1 ]; then URL=${CHISPA_LOCAL_URL:?en local hace falta CHISPA_LOCAL_URL}; MOD=${CHISPA_LOCAL_MODELO:-q4}; estado servidor "OMITIDO (local: $URL)"
else
  URL=http://127.0.0.1:8000; MOD=qwen3-4b; T1=$(date +%s)
  HF_HUB_ENABLE_HF_TRANSFER=0 "$VPY" -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', revision='$REV', local_dir='$BASE/modelo', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json','LICENSE*'])" \
    || { fallo "descarga del modelo"; estado modelo "FALLO"; cierre; }
  export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda 0be529b)
  nohup "$VPY" -m vllm.entrypoints.openai.api_server --model "$BASE/modelo" --served-model-name qwen3-4b --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --max-logprobs 20 --enable-prefix-caching --host 127.0.0.1 --port 8000 > "$BASE/vllm-4b.log" 2>&1 &
  VP=$!
  for _ in $(seq 144); do curl -s -m 3 localhost:8000/v1/models > /dev/null && break; kill -0 $VP 2>/dev/null || break; sleep 5; done
  curl -s -m 3 localhost:8000/v1/models | grep -q qwen3-4b || { tail -5 "$BASE/vllm-4b.log"; fallo "vLLM no arrancó"; estado servidor "FALLO"; cierre; }
  estado servidor "OK Qwen/Qwen3-4B-Instruct-2507@$REV en $(( $(date +%s) - T1 )) s"
fi
"$VPY" "$BASE/codigo/llm_logprobs.py" --filas "$BASE/auto.jsonl" $(urls $URL 6) --modelo $MOD --salida "$OUT/auto-4b.jsonl" --timeout 600 "${MAX[@]}"
NV=$(validas "$OUT/auto-4b.jsonl"); NA=$(lineas "$OUT/auto-4b.jsonl")
[ "$NA" -gt 0 ] && [ $(( NV * 2 )) -ge "$NA" ] || { head -c 600 "$OUT/auto-4b.jsonl"; fallo "autoprueba: $NV de $NA válidas"; estado prueba "FALLO"; cierre; }
estado prueba "OK $NV/$NA válidas"

# --- 4. (1) confirmación en el resto de la batería, con tope CONF_MIN
T1=$(date +%s)
cliente $((5 + ESPERA_MIN + 1)) "$CONF_MIN" --filas "$BASE/filas-confirma.jsonl" $(urls $URL $HILOS) --modelo $MOD --salida "$OUT/lp_confirma.jsonl" --timeout 600 --reintentos 1 "${MAX[@]}"
estado confirma "$(lineas "$OUT/lp_confirma.jsonl") de $N_CONF casos ($(validas "$OUT/lp_confirma.jsonl") válidas) en $(( $(date +%s) - T1 )) s"

# --- 5. (2) reetiquetado del soporte por logprobs hasta el tope
T1=$(date +%s)
cliente $((5 + ESPERA_MIN + 1)) 999 --filas "$BASE/reetiqueta.jsonl" $(urls $URL $HILOS) --modelo $MOD --salida "$OUT/lp_reetiqueta.jsonl" --timeout 600 --reintentos 1 "${MAX[@]}"
estado reetiqueta "$(lineas "$OUT/lp_reetiqueta.jsonl") de $N_REET textos ($(validas "$OUT/lp_reetiqueta.jsonl") válidas) en $(( $(date +%s) - T1 )) s"
cierre
