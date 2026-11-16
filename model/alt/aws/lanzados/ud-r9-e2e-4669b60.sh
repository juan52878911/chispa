#!/bin/bash
CHISPA_S3=s3://chispa-pruebas-879381269751/r9-e2e-4669b60
# R9-e2e: prueba de punta a punta en UNA g5.xlarge BAJO DEMANDA (DLAMI GPU PyTorch AL2023 ami-004851d2e6f24b4b4), tope duro CHISPA_MIN=30 min.
# vLLM 0.33.0 real con Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f (--max-logprobs 20, contexto 16384, como llm_r7b.sh) + el servidor Go
# (server/, compilado aquí con Go fijado por versión y sha256) con l3.modo="logprobs" contra ese vLLM, y el .chispa int8 de movil.
# Rol chispa-pruebas-ec2: s3:GetObject/PutObject por nombre exacto (sin ListBucket). Sin SSH; user-data; terminate al apagar.
# Lee (nombre exacto): $CHISPA_S3/{codigo.tgz,filas.jsonl}; $MOVIL_S3/{modelo-int8.chispa,e5.chispa-tok} (sha256 comprobado).
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
# Fases (cada una escribe a res/ al terminar), por prioridad: (a) paridad L3 con L3 forzado, (d) tope de tiempo, (b) cascada por defecto, (c) cola.
# L3 se fuerza con `"forzar_l3": true` en la config del servidor (solo pruebas; por defecto apagado): salta L0-L2 y acepta L3 con τ=0.
# Versiones fijadas: ver el bloque de abajo. Nada se descarga sin sha256 salvo lo que pip resuelve de vllm==VLLM (sin hashes) y el
# modelo de HF, fijado por commit.
MIN=${CHISPA_MIN:-30}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-3}; VLLM=${CHISPA_VLLM:-0.33.0}
REV=cdbee75f17c01a7cc42f958dc650907174af0554
N_FILAS=3116
GO_VERSION=1.27.1
GO_SHA256=63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445                  # go${GO_VERSION}.linux-amd64.tar.gz (de https://go.dev/dl/); lo sustituye empaquetar_r9_e2e.sh
ORT_VERSION=1.31.1                       # onnxruntime_go v1.38.0 (server/go.mod; README del servidor)
ORT_SHA256=c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9   # onnxruntime-linux-x64-1.31.1.tgz (server/deploy/Dockerfile)
MOVIL_S3=${CHISPA_MOVIL_S3:-s3://chispa-pruebas-879381269751/movil-90e3263c/movil}
MOVIL_SHA=995dd6a7276a7026ce12ad3e04a5324714f62dc05a1c20b11141a580c30db6f4; MOVIL_BYTES=249716640   # model/bench/movil/resultados-20261102-210131/manifiesto.json
TOK_SHA=356d7ef09425121e93a0a595ca5a4591fbebeb8785b0fc5f405a374459cd1ed6; TOK_BYTES=12111120        # e5.chispa-tok, mismo manifiesto
DURA_C=${CHISPA_DURA_C:-180}; CLIENTES_A=${CHISPA_CLIENTES_A:-16}; CLIENTES_C=${CHISPA_CLIENTES_C:-32}
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-r9-s3; mkdir -p "$CHISPA_S3";; esac; MOVIL_S3=${CHISPA_MOVIL_S3:-$CHISPA_S3/movil}
else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-r9}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r9-e2e $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
OUT=$BASE/res; FALLO=""; SRVPID=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
hay_tiempo() { [ "$(quedan_min)" -gt $((ESPERA_MIN + $1)) ]; }
parar_servidor() { [ -n "$SRVPID" ] && { kill "$SRVPID" 2>/dev/null; for _ in $(seq 20); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 0.5; done; kill -KILL "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; }; SRVPID=""; return 0; }
subir_y_apagar() {
  trap - EXIT; parar_servidor; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  for l in "$BASE"/vllm-*.log "$BASE"/servidor-*.log; do [ -f "$l" ] && tail -n 200 "$l" > "$OUT/$(basename "$l" .log)-cola.log"; done
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
# vigilante: a falta de ESPERA_MIN+2 min se cortan cliente y servidores y se sube lo que haya
( while sleep 20; do [ "$(quedan_min)" -le $((ESPERA_MIN + 2)) ] && { echo "== tope: se cierra"; pkill -TERM -f r9_cliente.py; pkill -TERM -f chispa-server; pkill -TERM -f vllm.entrypoints; break; }; done ) &
sha_ok() { if command -v sha256sum >/dev/null; then echo "$1  $2" | sha256sum -c --quiet -; else [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" = "$1" ]; fi; }

# --- 1. entradas por nombre exacto (en paralelo con el entorno)
echo "== 1. entradas $(date -u +%T)"
mkdir -p "$BASE/in" "$BASE/codigo"
for f in codigo.tgz filas.jsonl; do s3cp "$CHISPA_S3/$f" "$BASE/in/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
tar -xzf "$BASE/in/codigo.tgz" -C "$BASE/codigo" || { fallo "codigo.tgz"; cierre; }
NF=$(wc -l < "$BASE/in/filas.jsonl" | tr -d ' ')
[ "$NF" = "$N_FILAS" ] || { fallo "filas.jsonl tiene $NF filas, no $N_FILAS"; estado entradas "FALLO recuento"; cierre; }
[ -f "$BASE/codigo/server/go.mod" ] && [ -f "$BASE/codigo/model/alt/aws/r9_cliente.py" ] || { fallo "codigo.tgz sin server/ o r9_cliente.py"; cierre; }
estado entradas "OK $NF filas"

# --- 2. servidor Go: .chispa y tok (con sha), Go y ORT (con sha), compilación. Corre en segundo plano mientras se instala vLLM.
preparar_go() {
  set -u
  if [ "$LOCAL" = 1 ]; then
    s3cp "$MOVIL_S3/modelo-int8.chispa" "$BASE/modelo-int8.chispa" || return 1
    [ -f "$MOVIL_S3/e5.chispa-tok" ] && s3cp "$MOVIL_S3/e5.chispa-tok" "$BASE/e5.chispa-tok"
    echo "local: sha256 del .chispa y del tok NO comprobado (modelo de prueba)"
    TAGS=""; ORTLIB=""
  else
    s3cp "$MOVIL_S3/modelo-int8.chispa" "$BASE/modelo-int8.chispa" && s3cp "$MOVIL_S3/e5.chispa-tok" "$BASE/e5.chispa-tok" || return 1
    [ "$(wc -c < "$BASE/modelo-int8.chispa" | tr -d ' ')" = "$MOVIL_BYTES" ] && sha_ok "$MOVIL_SHA" "$BASE/modelo-int8.chispa" || { echo "modelo-int8.chispa: tamaño o sha256 distintos"; return 2; }
    [ "$(wc -c < "$BASE/e5.chispa-tok" | tr -d ' ')" = "$TOK_BYTES" ] && sha_ok "$TOK_SHA" "$BASE/e5.chispa-tok" || { echo "e5.chispa-tok: tamaño o sha256 distintos"; return 2; }
    command -v gcc >/dev/null || dnf install -y -q gcc tar gzip || return 3
    [ "${GO_SHA256#__}" != "$GO_SHA256" ] && { echo "GO_SHA256 sin sustituir"; return 3; }
    curl -fsSL -o "$BASE/go.tgz" "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" && sha_ok "$GO_SHA256" "$BASE/go.tgz" || { echo "Go: descarga o sha256"; return 3; }
    tar -xzf "$BASE/go.tgz" -C /usr/local || return 3
    curl -fsSL -o "$BASE/ort.tgz" "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" && sha_ok "$ORT_SHA256" "$BASE/ort.tgz" || { echo "ORT: descarga o sha256"; return 3; }
    mkdir -p /opt/ort && tar -xzf "$BASE/ort.tgz" -C /opt/ort --strip-components=1 || return 3
    export PATH=/usr/local/go/bin:$PATH; TAGS="-tags ort"; ORTLIB=/opt/ort/lib/libonnxruntime.so
  fi
  echo "$TAGS" > "$BASE/go-tags"; echo "$ORTLIB" > "$BASE/ort-lib"
  # los módulos salen de gomod-proxy/ del tgz y se verifican contra server/go.sum
  export GOTOOLCHAIN=local GOFLAGS=-mod=readonly GOSUMDB=off GOPROXY=file://$BASE/codigo/gomod-proxy GOCACHE=$BASE/gocache GOPATH=$BASE/gopath CGO_ENABLED=1
  go version && ( cd "$BASE/codigo/server" && go build $TAGS -trimpath -o "$BASE/chispa-server" ./cmd/chispa-server ) || return 4
}
( preparar_go > "$BASE/preparar-go.log" 2>&1; echo $? > "$BASE/preparar-go.rc" ) &
PGO=$!

# --- 3. vLLM (el mismo arranque que llm_r7b.sh)
echo "== 3. vLLM $(date -u +%T) quedan $(quedan_min) min"
if [ "$LOCAL" = 1 ]; then URL=${CHISPA_LOCAL_URL:?en local hace falta CHISPA_LOCAL_URL}; PY=${CHISPA_PY:-python3}; estado vllm "OMITIDO (local: $URL)"
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] && "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado vllm "FALLO"; cierre; }
  "$PY" -m venv /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" huggingface_hub || { fallo "pip install vllm==$VLLM"; estado vllm "FALLO pip"; cierre; }
  VPY=/opt/vllm/bin/python; PY=python3
  URL=http://127.0.0.1:8000; T1=$(date +%s)
  HF_HUB_ENABLE_HF_TRANSFER=0 "$VPY" -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', revision='$REV', local_dir='$BASE/modelo', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json','LICENSE*'])" \
    || { fallo "descarga del modelo"; estado vllm "FALLO modelo"; cierre; }
  export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda 0be529b)
  nohup "$VPY" -m vllm.entrypoints.openai.api_server --model "$BASE/modelo" --served-model-name qwen3-4b --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --max-logprobs 20 --enable-prefix-caching --host 127.0.0.1 --port 8000 > "$BASE/vllm-4b.log" 2>&1 &
  VP=$!
  for _ in $(seq 144); do curl -s -m 3 localhost:8000/v1/models > /dev/null && break; kill -0 $VP 2>/dev/null || break; sleep 5; done
  curl -s -m 3 localhost:8000/v1/models | grep -q qwen3-4b || { tail -5 "$BASE/vllm-4b.log"; fallo "vLLM no arrancó"; estado vllm "FALLO"; cierre; }
  estado vllm "OK vllm $(/opt/vllm/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null) Qwen3-4B-Instruct-2507@$REV en $(( $(date +%s) - T1 )) s; $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
fi
wait $PGO 2>/dev/null; RC=$(cat "$BASE/preparar-go.rc" 2>/dev/null || echo 9); cat "$BASE/preparar-go.log"
[ "$RC" = 0 ] && [ -x "$BASE/chispa-server" ] || { fallo "servidor Go: preparar_go rc=$RC"; estado servidor "FALLO rc=$RC"; cierre; }
ORTLIB=$(cat "$BASE/ort-lib"); estado servidor "OK Go $GO_VERSION (sha256 $GO_SHA256) ORT $ORT_VERSION; .chispa sha256 ${MOVIL_SHA:0:12} tok ${TOK_SHA:0:12}"

# --- 4. configuraciones y arranque del servidor
SERV=http://127.0.0.1:8090
cfg() {  # cfg NOMBRE FORZAR MAX_CONC TIMEOUT_FILA [TOP_K]
  local topk=""; [ -n "${5:-}" ] && topk=", \"top_k\": $5"
  local forz=""; [ "$2" = true ] && forz='"forzar_l3": true, '
  echo "{${forz}\"l3\": {\"url\": \"$URL/v1\", \"modelo\": \"${CHISPA_MODELO_L3:-qwen3-4b}\", \"modo\": \"logprobs\", \"max_concurrente\": $3, \"timeout\": \"30s\", \"timeout_fila\": \"$4\", \"max_por_peticion\": 8$topk}}" > "$BASE/cfg-$1.json"
}
cfg a true 32 30s 256      # (a) y (d normal): L3 para todo, 60 opciones enteras
cfg dbajo true 4 150ms 256 # (d) timeout_fila muy bajo
cfg b false 8 10s          # (b) cascada por defecto del servidor (top_k por defecto)
cfg c true 8 10s 256       # (c) cola: 8 filas a la vez contra 32 clientes
arrancar() {  # arrancar CONFIG
  parar_servidor
  ARGS=(-addr 127.0.0.1:8090 -modelo "$BASE/modelo-int8.chispa" -config "$BASE/cfg-$1.json" -timeout 120s)
  [ -f "$BASE/e5.chispa-tok" ] && ARGS+=(-tok "$BASE/e5.chispa-tok"); [ -n "$ORTLIB" ] && ARGS+=(-ort-lib "$ORTLIB")
  nohup "$BASE/chispa-server" "${ARGS[@]}" > "$BASE/servidor-$1.log" 2>&1 &
  SRVPID=$!
  for _ in $(seq 120); do curl -s -m 2 $SERV/healthz > /dev/null && return 0; kill -0 $SRVPID 2>/dev/null || break; sleep 1; done
  tail -5 "$BASE/servidor-$1.log"; return 1
}
cli() { "$PY" "$BASE/codigo/model/alt/aws/r9_cliente.py" "$@" --url $SERV --filas "$BASE/in/filas.jsonl" --salida "$OUT" ${CHISPA_N:+--n $CHISPA_N}; }

# --- 5. fases
fase() {  # fase NOMBRE MIN_NECESARIOS CONFIG comando...
  local n=$1 need=$2 c=$3; shift 3; local t=$(date +%s)
  hay_tiempo "$need" || { estado "$n" "OMITIDO por tiempo (quedan $(quedan_min) min)"; return 0; }
  arrancar "$c" || { fallo "servidor no arrancó ($c)"; estado "$n" "FALLO servidor"; return 0; }
  "$@" > "$OUT/$n.out" 2>&1 && estado "$n" "OK en $(( $(date +%s) - t )) s" || { fallo "fase $n"; tail -3 "$OUT/$n.out"; estado "$n" "FALLO"; }
}
echo "== 5. fases $(date -u +%T) quedan $(quedan_min) min"
# autoprueba: una fila debe dar probs de L3 con el servidor forzado antes de gastar la fase (a)
arrancar a || { fallo "servidor no arrancó (a)"; estado a "FALLO servidor"; cierre; }
head -1 "$BASE/in/filas.jsonl" > "$BASE/in/una.jsonl"
"$PY" "$BASE/codigo/model/alt/aws/r9_cliente.py" a --url $SERV --filas "$BASE/in/una.jsonl" --salida "$BASE" --clientes 1 > "$BASE/autoprueba.out" 2>&1 \
  && grep -q '"validas": 1' "$BASE/a-resumen.json" 2>/dev/null || { cat "$BASE/autoprueba.out" | head -c 800; fallo "autoprueba: L3 no contesta"; estado prueba "FALLO"; cierre; }
estado prueba "OK L3 contesta por el servidor"; rm -f "$BASE/a-resumen.json" "$BASE/a-l3.jsonl"
T1=$(date +%s)
cli a --clientes "$CLIENTES_A" > "$OUT/a.out" 2>&1 && estado a "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json;print(json.load(open('$OUT/a-resumen.json'))['agrupada_ponderada'])")" || { fallo "fase a"; estado a "FALLO"; }
T1=$(date +%s); cli d --etiqueta normal --rep 5 > "$OUT/d-normal.out" 2>&1 && estado d-normal "OK en $(( $(date +%s) - T1 )) s" || { fallo "fase d normal"; estado d-normal "FALLO"; }
fase d-bajo 4 dbajo cli d --etiqueta bajo --rep 5
fase b 7 b cli b
dura_c() { local s=$(( ($(quedan_min) - ESPERA_MIN - 3) * 60 )); [ "$s" -gt "$DURA_C" ] && s=$DURA_C; echo "$s"; }
fase c 4 c cli c --clientes "$CLIENTES_C" --dura "$(dura_c)"
cierre
