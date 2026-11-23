#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# R10-caudal: g5.xlarge BAJO DEMANDA (DLAMI GPU PyTorch AL2023 ami-ejemplo), tope duro CHISPA_MIN=30 min. Mismo montaje que r9_e2e.sh
# (vLLM 0.33.0 real con Qwen3-4B-Instruct-2507 @ cdbee75f, servidor Go endurecido compilado aquí con -pruebas y l3.modo="logprobs",
# .chispa int8 y tokenizador de movil-f6acb278/movil con sha256) pero con otras fases, por prioridad:
#   (b) cascada por defecto del servidor con 1 cliente sobre 600 filas intercaladas por tarea: p50/p95 por capa, fracción por capa, exactitud;
#   (e) barrido de caudal con L3 forzado: pares max_concurrente:clientes en el orden de CHISPA_BARRIDO (lo más informativo primero), DURA_P s por punto.
# Preparación en paralelo: pip de vLLM, descarga del modelo (venv aparte con huggingface_hub) y compilación de Go (+ .chispa, ORT).
# Lee (nombre exacto): $CHISPA_S3/{codigo.tgz,filas.jsonl}; $MOVIL_S3/{modelo-int8.chispa,e5.chispa-tok} (sha256 comprobado).
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
# Los percentiles de espera y de latencia de L3 salen de las cabeceras X-Chispa-L3-Espera-Ms / X-Chispa-L3-Ms de cada respuesta (cliente), no del histograma.
MIN=${CHISPA_MIN:-30}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-3}; VLLM=${CHISPA_VLLM:-0.33.0}
REV=cdbee75f17c01a7cc42f958dc650907174af0554
N_FILAS=__N_FILAS__
N_B=${CHISPA_N_B:-600}   # filas de la fase (b), intercaladas por tarea
GO_VERSION=1.27.1
GO_SHA256=__GO_SHA256__                  # go${GO_VERSION}.linux-amd64.tar.gz (de https://go.dev/dl/); lo sustituye empaquetar_r10_caudal.sh
ORT_VERSION=1.31.1                       # onnxruntime_go v1.38.0 (server/go.mod; README del servidor)
ORT_SHA256=c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9   # onnxruntime-linux-x64-1.31.1.tgz (server/deploy/Dockerfile)
MOVIL_S3=${CHISPA_MOVIL_S3:-s3://chispa-pruebas-cuenta-aws/movil-f6acb278/movil}
MOVIL_SHA=995dd6a7276a7026ce12ad3e04a5324714f62dc05a1c20b11141a580c30db6f4; MOVIL_BYTES=249716640   # model/bench/movil/resultados-20261102-210131/manifiesto.json
TOK_SHA=356d7ef09425121e93a0a595ca5a4591fbebeb8785b0fc5f405a374459cd1ed6; TOK_BYTES=12111120        # e5.chispa-tok, mismo manifiesto
DURA_P=${CHISPA_DURA_P:-50}
# pares max_concurrente:clientes; orden = lo más informativo antes (la saturación de la cola y el efecto de más huecos)
BARRIDO=${CHISPA_BARRIDO:-"32:64 16:64 8:64 32:32 16:32 8:32 16:16 8:16 32:16 8:8 16:8 32:8"}
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-r10-s3; mkdir -p "$CHISPA_S3";; esac; MOVIL_S3=${CHISPA_MOVIL_S3:-$CHISPA_S3/movil}
else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-r10}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r10-caudal $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
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
( while sleep 20; do [ "$(quedan_min)" -le $((ESPERA_MIN + 2)) ] && { echo "== tope: se cierra"; pkill -TERM -f r10_cliente.py; pkill -TERM -f chispa-server; pkill -TERM -f vllm.entrypoints; break; }; done ) &
sha_ok() { if command -v sha256sum >/dev/null; then echo "$1  $2" | sha256sum -c --quiet -; else [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" = "$1" ]; fi; }

# --- 1. entradas por nombre exacto (en paralelo con el entorno)
echo "== 1. entradas $(date -u +%T)"
mkdir -p "$BASE/in" "$BASE/codigo"
for f in codigo.tgz filas.jsonl; do s3cp "$CHISPA_S3/$f" "$BASE/in/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
tar -xzf "$BASE/in/codigo.tgz" -C "$BASE/codigo" || { fallo "codigo.tgz"; cierre; }
NF=$(wc -l < "$BASE/in/filas.jsonl" | tr -d ' ')
[ "$NF" = "$N_FILAS" ] || { fallo "filas.jsonl tiene $NF filas, no $N_FILAS"; estado entradas "FALLO recuento"; cierre; }
[ -f "$BASE/codigo/server/go.mod" ] && [ -f "$BASE/codigo/model/alt/aws/r10_cliente.py" ] || { fallo "codigo.tgz sin server/ o r10_cliente.py"; cierre; }
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

# --- 3. vLLM: pip y descarga del modelo en paralelo (la descarga usa un venv aparte con solo huggingface_hub)
echo "== 3. vLLM $(date -u +%T) quedan $(quedan_min) min"
if [ "$LOCAL" = 1 ]; then URL=${CHISPA_LOCAL_URL:?en local hace falta CHISPA_LOCAL_URL}; PY=${CHISPA_PY:-python3}; estado vllm "OMITIDO (local: $URL)"
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] && "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado vllm "FALLO"; cierre; }
  descargar_modelo() {
    "$PY" -m venv /opt/hf && /opt/hf/bin/pip install -q --no-cache-dir huggingface_hub || return 1
    HF_HUB_ENABLE_HF_TRANSFER=0 /opt/hf/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', revision='$REV', local_dir='$BASE/modelo', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json','LICENSE*'])"
  }
  ( descargar_modelo > "$BASE/descarga-modelo.log" 2>&1; echo $? > "$BASE/descarga-modelo.rc" ) &
  PDL=$!
  T1=$(date +%s)
  "$PY" -m venv /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" || { fallo "pip install vllm==$VLLM"; estado vllm "FALLO pip"; cierre; }
  VPY=/opt/vllm/bin/python; PY=python3
  wait $PDL 2>/dev/null; [ "$(cat "$BASE/descarga-modelo.rc" 2>/dev/null || echo 9)" = 0 ] || { tail -3 "$BASE/descarga-modelo.log"; fallo "descarga del modelo"; estado vllm "FALLO modelo"; cierre; }
  URL=http://127.0.0.1:8000
  export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda d36f8ab)
  nohup "$VPY" -m vllm.entrypoints.openai.api_server --model "$BASE/modelo" --served-model-name qwen3-4b --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --max-logprobs 20 --enable-prefix-caching --host 127.0.0.1 --port 8000 > "$BASE/vllm-4b.log" 2>&1 &
  VP=$!
  for _ in $(seq 144); do curl -s -m 3 localhost:8000/v1/models > /dev/null && break; kill -0 $VP 2>/dev/null || break; sleep 5; done
  curl -s -m 3 localhost:8000/v1/models | grep -q qwen3-4b || { tail -5 "$BASE/vllm-4b.log"; fallo "vLLM no arrancó"; estado vllm "FALLO"; cierre; }
  estado vllm "OK vllm $(/opt/vllm/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null) Qwen3-4B-Instruct-2507@$REV (pip+modelo+arranque) en $(( $(date +%s) - T1 )) s; $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
fi
wait $PGO 2>/dev/null; RC=$(cat "$BASE/preparar-go.rc" 2>/dev/null || echo 9); cat "$BASE/preparar-go.log"
[ "$RC" = 0 ] && [ -x "$BASE/chispa-server" ] || { fallo "servidor Go: preparar_go rc=$RC"; estado servidor "FALLO rc=$RC"; cierre; }
ORTLIB=$(cat "$BASE/ort-lib"); estado servidor "OK Go $GO_VERSION (sha256 $GO_SHA256) ORT $ORT_VERSION; .chispa sha256 ${MOVIL_SHA:0:12} tok ${TOK_SHA:0:12}"

# --- 4. configuraciones y arranque del servidor (siempre con -pruebas: el barrido usa forzar_l3)
SERV=http://127.0.0.1:8090
cfg() {  # cfg NOMBRE FORZAR MAX_CONC TIMEOUT_FILA [TOP_K]
  local topk=""; [ -n "${5:-}" ] && topk=", \"top_k\": $5"
  local forz=""; [ "$2" = true ] && forz='"forzar_l3": true, '
  echo "{${forz}\"l3\": {\"url\": \"$URL/v1\", \"modelo\": \"${CHISPA_MODELO_L3:-qwen3-4b}\", \"modo\": \"logprobs\", \"max_concurrente\": $3, \"timeout\": \"30s\", \"timeout_fila\": \"$4\", \"max_por_peticion\": 256$topk}}" > "$BASE/cfg-$1.json"
}
cfg b false 8 10s          # (b) cascada por defecto del servidor (top_k por defecto)
for c in $(for p in $BARRIDO; do echo "${p%%:*}"; done | sort -un); do cfg m$c true "$c" 10s 256; done   # (e) L3 forzado, max_concurrente c
SRVCFG=""
arrancar() {  # arrancar CONFIG (no reinicia si ya corre con esa config)
  [ "$SRVCFG" = "$1" ] && [ -n "$SRVPID" ] && kill -0 "$SRVPID" 2>/dev/null && return 0
  parar_servidor; SRVCFG=""
  ARGS=(-addr 127.0.0.1:8090 -modelo "$BASE/modelo-int8.chispa" -config "$BASE/cfg-$1.json" -timeout 120s -pruebas)
  [ -f "$BASE/e5.chispa-tok" ] && ARGS+=(-tok "$BASE/e5.chispa-tok"); [ -n "$ORTLIB" ] && ARGS+=(-ort-lib "$ORTLIB")
  nohup "$BASE/chispa-server" "${ARGS[@]}" > "$BASE/servidor-$1.log" 2>&1 &
  SRVPID=$!
  for _ in $(seq 120); do curl -s -m 2 $SERV/healthz > /dev/null && { SRVCFG=$1; return 0; }; kill -0 $SRVPID 2>/dev/null || break; sleep 1; done
  tail -5 "$BASE/servidor-$1.log"; return 1
}
cli() { "$PY" "$BASE/codigo/model/alt/aws/r10_cliente.py" "$@" --url $SERV --filas "$BASE/in/filas.jsonl" --salida "$OUT" ${CHISPA_N:+--n $CHISPA_N}; }

# --- 5. fases
echo "== 5. fases $(date -u +%T) quedan $(quedan_min) min"
# autoprueba: unas filas por la cascada (con L3 detrás) antes de gastar la fase (b)
arrancar b || { fallo "servidor no arrancó (b)"; estado prueba "FALLO servidor"; cierre; }
mkdir -p "$BASE/auto"
"$PY" "$BASE/codigo/model/alt/aws/r10_cliente.py" b --url $SERV --filas "$BASE/in/filas.jsonl" --salida "$BASE/auto" --intercalado 6 > "$BASE/autoprueba.out" 2>&1 \
  && grep -q '"n": 6' "$BASE/auto/b-resumen.json" 2>/dev/null && ! grep -q "\"None\"" "$BASE/auto/b-resumen.json" || { head -c 800 "$BASE/autoprueba.out"; fallo "autoprueba: la cascada no contesta"; estado prueba "FALLO"; cierre; }
estado prueba "OK la cascada contesta por el servidor"
T1=$(date +%s)
cli b --intercalado "$N_B" > "$OUT/b.out" 2>&1 && estado b "OK en $(( $(date +%s) - T1 )) s" || { fallo "fase b"; estado b "FALLO"; }
# (e) barrido: un servidor por max_concurrente (se reinicia solo al cambiar); cada punto escribe c-mCONC-cCLIENTES-*.json
for par in $BARRIDO; do
  conc=${par%%:*}; cl=${par##*:}; T1=$(date +%s)
  hay_tiempo 2 || { estado "e-m$conc-c$cl" "OMITIDO por tiempo (quedan $(quedan_min) min)"; continue; }
  arrancar "m$conc" || { fallo "servidor no arrancó (m$conc)"; estado "e-m$conc-c$cl" "FALLO servidor"; continue; }
  cli c --clientes "$cl" --dura "$DURA_P" --etiqueta "m$conc-c$cl" > "$OUT/e-m$conc-c$cl.out" 2>&1 \
    && estado "e-m$conc-c$cl" "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json;d=json.load(open('$OUT/c-m$conc-c$cl-resumen.json'));print('%s req/s p50=%s p95=%s ms' % (d['por_segundo'], d['total_ms_cliente']['p50'], d['total_ms_cliente']['p95']))")" \
    || { fallo "punto $par"; estado "e-m$conc-c$cl" "FALLO"; }
done
cierre
