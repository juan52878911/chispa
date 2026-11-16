#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# Paso 2, ronda solo de Q4: Qwen3-4B-Instruct-2507 Q4_K_M (el GGUF de L3) con llama.cpp en el contenedor oficial CUDA, sobre las mismas
# filas-q4.jsonl, con el mismo cliente y método que la ronda bf16 (llm-bateria-4994f3e). g5.xlarge BAJO DEMANDA, tope duro CHISPA_MIN=33 min
# (≈ 0,55 USD). user-data (DLAMI GPU PyTorch AL2023, rol rol-ec2-pruebas sin ListBucket), --instance-initiated-shutdown-behavior terminate.
# Lee:     $CHISPA_S3/{llm.tgz,filas-q4.jsonl}
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
# Imagen fijada por digest (manifiesto linux/amd64 de ghcr.io/ggml-org/llama.cpp:server-cuda12; versión b12375, commit 9d25e00, CUDA 12.8.1).
# Antes de bajar el GGUF: autoprueba `docker run --gpus all IMAGEN --version` y `--list-devices`, que debe listar CUDA0; si no hay docker o el toolkit de NVIDIA, cierra.
MIN=${CHISPA_MIN:-33}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-16}
IMG=ghcr.io/ggml-org/llama.cpp@sha256:e110d1ae4a3572398950e7b64b5abd585a3eab678ae726484f743a12140bdc40
GGUF_URL=https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
GGUF_SHA=3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597
N_Q4=__N_Q4__
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-q4-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-q4}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque llm-q4 $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min hilos=$HILOS"
OUT=$BASE/res; FALLO=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
subir_y_apagar() {
  trap - EXIT; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  [ "$LOCAL" = 1 ] || docker logs q4 2>&1 | tail -n 200 > "$OUT/llama-cola.log"
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
( while sleep 20; do [ "$(quedan_min)" -le $((5 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f llm_jevbench; break; }; done ) &
lineas() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
validas() { [ -f "$1" ] || { echo 0; return; }; grep -c '"valida": true' "$1"; }
urls() { for _ in $(seq "$2"); do printf -- '--url %s ' "$1"; done; }

# --- 1. entorno: python para el cliente (solo biblioteca estándar) y docker con GPU
echo "== 1. entorno $(date -u +%T)"
PY=""; for c in /opt/pytorch/bin/python /usr/bin/python3 python3; do command -v "$c" > /dev/null 2>&1 && { PY=$c; break; }; done
[ -n "$PY" ] || { fallo "sin python"; estado entorno "FALLO"; cierre; }
if [ "$LOCAL" = 1 ]; then estado entorno "OK local ($PY)"
else
  command -v docker > /dev/null || { fallo "sin docker en la AMI"; estado entorno "FALLO sin docker"; cierre; }
  systemctl start docker 2>/dev/null; docker info > /dev/null 2>&1 || { fallo "docker no responde"; estado entorno "FALLO docker"; cierre; }
  T1=$(date +%s)
  docker pull -q "$IMG" > /dev/null || { fallo "docker pull $IMG"; estado entorno "FALLO pull"; cierre; }
  docker run --rm --gpus all "$IMG" --version > "$OUT/version.txt" 2>&1 && docker run --rm --gpus all "$IMG" --list-devices >> "$OUT/version.txt" 2>&1 || { cat "$OUT/version.txt"; fallo "docker run --gpus all (¿nvidia-container-toolkit?)"; estado entorno "FALLO gpus"; cierre; }
  grep -q "CUDA0" "$OUT/version.txt" || { cat "$OUT/version.txt"; fallo "el contenedor no ve la GPU"; estado entorno "FALLO sin CUDA en el contenedor"; cierre; }
  estado entorno "OK docker $(docker --version | cut -d' ' -f3 | tr -d ,), imagen en $(( $(date +%s) - T1 )) s, $(grep -m1 -i 'version' "$OUT/version.txt"); $(grep -m1 'CUDA0' "$OUT/version.txt")"
fi

# --- 2. entradas por nombre exacto
for f in llm.tgz filas-q4.jsonl; do s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
mkdir -p "$BASE/codigo" && tar -xzf "$BASE/llm.tgz" -C "$BASE/codigo" || { fallo "llm.tgz"; cierre; }
[ "$(lineas "$BASE/filas-q4.jsonl")" = "$N_Q4" ] || { fallo "filas-q4.jsonl no tiene $N_Q4 filas"; estado entradas "FALLO recuento"; cierre; }
"$PY" -c "
import json,sys
v={}
for l in open(sys.argv[1]):
    r=json.loads(l); v.setdefault(r['conjunto'], l)
open(sys.argv[2],'w').write(''.join(v[k] for k in sorted(v)))" "$BASE/filas-q4.jsonl" "$BASE/auto.jsonl"
estado entradas "OK $N_Q4 filas, $(lineas "$BASE/auto.jsonl") de autoprueba"
MAX=(); [ "$LOCAL" = 1 ] && MAX=(--max "${CHISPA_LOCAL_MAX:-2}")

# --- 3. GGUF y servidor (16 ranuras, 6 144 tokens por ranura, como en la ronda anterior)
if [ "$LOCAL" = 1 ]; then URL=${CHISPA_LOCAL_URL:?en local hace falta CHISPA_LOCAL_URL}; MOD=${CHISPA_LOCAL_MODELO:-q4}; estado servidor "OMITIDO (local: $URL)"
else
  URL=http://127.0.0.1:8001; MOD=q4; mkdir -p "$BASE/gguf"
  curl -sSfL -o "$BASE/gguf/q4.gguf" "$GGUF_URL" && echo "$GGUF_SHA  $BASE/gguf/q4.gguf" | sha256sum -c --quiet || { fallo "GGUF (descarga o sha256)"; estado gguf "FALLO"; cierre; }
  estado gguf "OK sha256 ${GGUF_SHA:0:16}…"
  docker run -d --name q4 --gpus all -p 127.0.0.1:8001:8080 -v "$BASE/gguf:/models:ro" "$IMG" \
    -m /models/q4.gguf --host 0.0.0.0 --port 8080 -ngl 99 -np 16 -c 98304 -fa on --jinja --alias q4 > /dev/null || { fallo "docker run del servidor"; estado servidor "FALLO"; cierre; }
  for _ in $(seq 60); do curl -s -m 3 localhost:8001/v1/models | grep -q q4 && break; [ "$(docker inspect -f '{{.State.Running}}' q4 2>/dev/null)" = true ] || break; sleep 5; done
  curl -s -m 3 localhost:8001/v1/models | grep -q q4 || { docker logs q4 2>&1 | tail -8; fallo "llama-server no arrancó"; estado servidor "FALLO"; cierre; }
  estado servidor "OK $(docker logs q4 2>&1 | grep -m1 -iE 'offloaded [0-9]+/[0-9]+ layers' | sed 's/.*offloaded/offloaded/')"
fi

# --- 4. autoprueba (una fila por tarea, incluida massive-es) y las filas-q4 con tope de tiempo
"$PY" "$BASE/codigo/llm_jevbench.py" --filas "$BASE/auto.jsonl" $(urls $URL 6) --modelo $MOD --salida "$OUT/auto-q4.jsonl" --formato vllm --timeout 600 "${MAX[@]}"
NV=$(validas "$OUT/auto-q4.jsonl"); NA=$(lineas "$OUT/auto-q4.jsonl")
[ "$NA" -gt 0 ] && [ $(( NV * 2 )) -ge "$NA" ] || { head -c 600 "$OUT/auto-q4.jsonl"; fallo "autoprueba q4: $NV de $NA válidas"; estado prueba "FALLO"; cierre; }
estado prueba "OK q4 $NV/$NA válidas"
T1=$(date +%s)
"$PY" "$BASE/codigo/llm_jevbench.py" --filas "$BASE/filas-q4.jsonl" $(urls $URL $HILOS) --modelo $MOD --salida "$OUT/l3_q4.jsonl" --formato vllm --timeout 600 --reintentos 1 "${MAX[@]}"
estado q4 "$(lineas "$OUT/l3_q4.jsonl") de $N_Q4 casos ($(validas "$OUT/l3_q4.jsonl") válidas) en $(( $(date +%s) - T1 )) s"
cierre
