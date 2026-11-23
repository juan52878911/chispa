#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# Techo del LLM (Qwen3-4B-Instruct-2507, bf16) en los 231 casos de JevBench con salida restringida por esquema, en una g5.xlarge BAJO DEMANDA
# (1,006 USD/h; techo duro de 60 min = 1,01 USD). user-data (< 16 KB; DLAMI GPU PyTorch AL2023, rol rol-ec2-pruebas sin ListBucket), con
# --instance-initiated-shutdown-behavior terminate. Todo por nombre exacto.
# Lee:    $CHISPA_S3/{llm.tgz,jev_filas.jsonl,l3_jevbench_parcial.jsonl,baseline-top1.json}
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
# Modelo: Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f17c01a7cc42f958dc650907174af0554 (Apache-2.0), vLLM 0.33.0. Cliente: llm_jevbench.py --formato vllm
# (esquema con todos los ids obligatorios, contexto 16 k, timeout 600 s, un reintento si la respuesta no es válida).
# Las respuestas de Jev solo se usan para medir (MCA 2.3(b)); nada de esto entra en un entrenamiento.
MIN=${CHISPA_MIN:-60}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-8}
REV=cdbee75f17c01a7cc42f958dc650907174af0554; VLLM=${CHISPA_VLLM:-0.33.0}
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-llm-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-llm}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque llm $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
OUT=$BASE/res; FALLO=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
subir_y_apagar() {
  trap - EXIT; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  [ -f "$BASE/vllm.log" ] && tail -n 200 "$BASE/vllm.log" > "$OUT/vllm-cola.log"
  ( cd "$BASE" && tar -czf resultados.tgz res ) && s3cp "$BASE/resultados.tgz" "$CHISPA_S3/resultados.tgz" || echo "no se pudo subir resultados.tgz"
  s3cp "$BASE/log.txt" "$CHISPA_S3/log.txt" || echo "no se pudo subir log.txt"
  if [ "$LOCAL" != 1 ]; then
    echo listo > "$BASE/listo"; s3cp "$BASE/listo" "$CHISPA_S3/listo" || true
    # sin ListBucket: GetObject de la marca exacta; el orquestador trae y verifica resultados.tgz y entonces sube `verificado`
    for _ in $(seq $((ESPERA_MIN * 2))); do aws s3 cp --only-show-errors "$CHISPA_S3/verificado" "$BASE/verificado" 2>/dev/null && break; sleep 30; done
    shutdown -h now
  fi
}
trap subir_y_apagar EXIT; trap 'exit 143' TERM INT HUP
cierre() { echo "== listo $(date -u +%FT%TZ) fallo=${FALLO:-no} ($(( $(date +%s) - T0 )) s)"; exit 0; }
( while sleep 30; do [ "$(quedan_min)" -le $((6 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f llm_jevbench; pkill -TERM -f "vllm serve"; break; }; done ) &

# --- 1. entorno: venv con vLLM fijado (el que ya sirvió para destilar en esta AMI)
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}; VPY=$PY
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] && "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado entorno "FALLO"; cierre; }
  "$PY" -m venv /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" huggingface_hub \
    || { fallo "pip install vllm==$VLLM"; estado entorno "FALLO pip"; cierre; }
  VPY=/opt/vllm/bin/python
  estado entorno "OK vllm $($VPY -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
fi

# --- 2. entradas por nombre exacto
echo "== 2. entradas $(date -u +%T)"
for f in llm.tgz jev_filas.jsonl l3_jevbench_parcial.jsonl baseline-top1.json; do s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
mkdir -p "$BASE/codigo" && tar -xzf "$BASE/llm.tgz" -C "$BASE/codigo" || { fallo "llm.tgz"; cierre; }
[ "$(wc -l < "$BASE/jev_filas.jsonl")" = 231 ] || { fallo "jev_filas.jsonl no tiene 231 casos"; cierre; }
estado entradas "OK"
if [ "$LOCAL" = 1 ]; then cp "$CHISPA_LOCAL_DATOS/l3_demo.jsonl" "$OUT/l3_jevbench.jsonl"; estado modelo "OMITIDO (local)"; estado servidor "OMITIDO (local; respuestas de juguete)"
else
  MODELO=$BASE/modelo
  HF_HUB_ENABLE_HF_TRANSFER=0 "$VPY" -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', revision='$REV', local_dir='$MODELO', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json','LICENSE*'])" \
    || { fallo "descarga del modelo"; estado modelo "FALLO"; cierre; }
  estado modelo "OK Qwen/Qwen3-4B-Instruct-2507@$REV"
  # --- 3. servidor: contexto 16 k, un solo modelo, sin caché de red; arranque con espera de hasta 12 min
  # la DLAMI no trae nvcc: el muestreador de FlashInfer se compila con JIT y tumba el arranque (ronda d36f8ab); se usa el de PyTorch
  export VLLM_USE_FLASHINFER_SAMPLER=0
  nohup "$VPY" -m vllm.entrypoints.openai.api_server --model "$MODELO" --served-model-name qwen3-4b --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching --host 127.0.0.1 --port 8000 > "$BASE/vllm.log" 2>&1 &
  VPID=$!
  for _ in $(seq 144); do curl -s -m 3 localhost:8000/v1/models > /dev/null && break; kill -0 $VPID 2>/dev/null || break; sleep 5; done
  curl -s -m 3 localhost:8000/v1/models | grep -q qwen3-4b || { tail -5 "$BASE/vllm.log"; fallo "vLLM no arrancó"; estado servidor "FALLO"; cierre; }
  estado servidor "OK vllm $VLLM, ctx 16384"
  # --- 4. autocomprobación: 3 casos; si el servidor rechaza el esquema, se sabe en segundos y no tras 231 casos
  "$VPY" "$BASE/codigo/llm_jevbench.py" --filas "$BASE/jev_filas.jsonl" --url http://127.0.0.1:8000 --modelo qwen3-4b --salida "$OUT/prueba3.jsonl" --formato vllm --max 3 --timeout 600
  [ "$(grep -c '"valida": true' "$OUT/prueba3.jsonl")" -ge 2 ] || { head -c 600 "$OUT/prueba3.jsonl"; fallo "autocomprobación: menos de 2 de 3 respuestas válidas"; estado prueba "FALLO"; cierre; }
  estado prueba "OK $(grep -c '"valida": true' "$OUT/prueba3.jsonl")/3 válidas"
  # --- 5. los 231 casos (todos, con la misma restricción; las 26 válidas de la v1 libre se usan solo para comparar)
  T1=$(date +%s)
  "$VPY" "$BASE/codigo/llm_jevbench.py" --filas "$BASE/jev_filas.jsonl" --url http://127.0.0.1:8000 --modelo qwen3-4b --salida "$OUT/l3_jevbench.jsonl" --formato vllm --timeout 600 --reintentos 1 \
    || { fallo "cliente"; estado llm "FALLO"; }
  estado llm "OK $(wc -l < "$OUT/l3_jevbench.jsonl") casos en $(( $(date +%s) - T1 )) s"
fi
# --- 6. techo con IC95, estricto y permisivo
python3 "$BASE/codigo/llm_techo.py" --filas "$BASE/jev_filas.jsonl" --llm "$OUT/l3_jevbench.jsonl" --base "$BASE/baseline-top1.json" --antes "$BASE/l3_jevbench_parcial.jsonl" --salida "$OUT" > "$OUT/techo.txt" 2>&1 \
  && estado techo "OK $(tr -d '\n ' < "$OUT/llm_techo.json" | cut -c1-260)" || { tail -3 "$OUT/techo.txt"; fallo "llm_techo"; estado techo "FALLO"; }
cierre
