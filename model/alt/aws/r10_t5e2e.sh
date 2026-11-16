#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# R10-t5e2e: ronda e2e corta en una g5.xlarge BAJO DEMANDA (DLAMI GPU PyTorch AL2023 ami-ejemplo), tope duro CHISPA_MIN=30 min.
# Sirve /v1/decide con L2 = kNN por tarea (bancos de banco-paridad, codificador e5 fp32 en ORT dentro del servidor) y L3 por logprobs (vLLM 0.33.0
# con Qwen3-4B-Instruct-2507 @ cdbee75f), servidor con -pruebas y -tags ort, y pasa las 4244 consultas por HTTP con su texto real.
# Mide, por prioridad: (1) exactitud de la cascada (macro sin hatecheck frente a 0,7233 de Python), fracción escalada por tarea y motivo y p50/p95 por
# capa (codificador e5, kNN, L3); (2) barrido corto de caudal con la mezcla real (16 y 64 clientes, max_concurrente 32); y, en segundo plano durante
# la preparación: paridad de vectores Go/Python del e5 (criterio congelado model/alt/crit-r10-t5b.json) y la versión int8 del e5 (informativa).
# Lee (nombre exacto): $CHISPA_S3/{codigo.tgz,bancos.tgz,consultas.jsonl,e5.onnx}; $MOVIL_S3/{modelo-int8.chispa,e5.chispa-tok} (sha256 comprobado).
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
MIN=${CHISPA_MIN:-30}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-3}; VLLM=${CHISPA_VLLM:-0.33.0}
REV=cdbee75f17c01a7cc42f958dc650907174af0554
N_FILAS=__N_FILAS__
E5_SHA256=__E5_SHA256__   # sha256 de e5.onnx (fp32), lo sustituye el empaquetador
REF_PY=${CHISPA_REF_PY:-0.7233}
GO_VERSION=1.27.1
GO_SHA256=__GO_SHA256__                  # go${GO_VERSION}.linux-amd64.tar.gz (de https://go.dev/dl/); lo sustituye empaquetar_r10_t5e2e.sh
ORT_VERSION=1.31.1                       # onnxruntime_go v1.38.0 (server/go.mod; README del servidor)
ORT_SHA256=c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9   # onnxruntime-linux-x64-1.31.1.tgz (server/deploy/Dockerfile)
MOVIL_S3=${CHISPA_MOVIL_S3:-s3://chispa-pruebas-cuenta-aws/movil-90e3263c/movil}
MOVIL_SHA=995dd6a7276a7026ce12ad3e04a5324714f62dc05a1c20b11141a580c30db6f4; MOVIL_BYTES=249716640   # model/bench/movil/resultados-20261102-210131/manifiesto.json
TOK_SHA=356d7ef09425121e93a0a595ca5a4591fbebeb8785b0fc5f405a374459cd1ed6; TOK_BYTES=12111120        # e5.chispa-tok, mismo manifiesto
DURA_P=${CHISPA_DURA_P:-50}; CARGA="${CHISPA_CARGA:-16 64}"; PAR_N=${CHISPA_PAR_N:-1000}; INT8_N=${CHISPA_INT8_N:-600}; INT8=${CHISPA_INT8:-1}
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-r10t5-s3; mkdir -p "$CHISPA_S3";; esac; MOVIL_S3=${CHISPA_MOVIL_S3:-$CHISPA_S3/movil}
else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-r10t5}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r10-t5e2e $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
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
( while sleep 20; do [ "$(quedan_min)" -le $((ESPERA_MIN + 2)) ] && { echo "== tope: se cierra"; pkill -TERM -f r10_t5_cliente.py; pkill -TERM -f chispa-server; pkill -TERM -f vllm.entrypoints; break; }; done ) &
sha_ok() { if command -v sha256sum >/dev/null; then echo "$1  $2" | sha256sum -c --quiet -; else [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" = "$1" ]; fi; }

# --- 1. entradas por nombre exacto
echo "== 1. entradas $(date -u +%T)"
mkdir -p "$BASE/in" "$BASE/codigo" "$BASE/bancos"
for f in codigo.tgz bancos.tgz consultas.jsonl e5.onnx; do s3cp "$CHISPA_S3/$f" "$BASE/in/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
tar -xzf "$BASE/in/codigo.tgz" -C "$BASE/codigo" || { fallo "codigo.tgz"; cierre; }
tar -xzf "$BASE/in/bancos.tgz" -C "$BASE/bancos" || { fallo "bancos.tgz"; cierre; }
NF=$(wc -l < "$BASE/in/consultas.jsonl" | tr -d ' ')
[ "$NF" = "$N_FILAS" ] || { fallo "consultas.jsonl tiene $NF filas, no $N_FILAS"; estado entradas "FALLO recuento"; cierre; }
if [ "$LOCAL" != 1 ]; then sha_ok "$E5_SHA256" "$BASE/in/e5.onnx" || { fallo "e5.onnx: sha256 distinto de $E5_SHA256"; estado entradas "FALLO sha e5.onnx"; cierre; }; fi
[ -f "$BASE/codigo/server/go.mod" ] && [ -f "$BASE/codigo/model/alt/aws/r10_t5_cliente.py" ] && [ -f "$BASE/bancos/referencia.json" ] || { fallo "codigo.tgz sin server/ o r10_t5_cliente.py, o bancos.tgz sin referencia.json"; cierre; }
estado entradas "OK $NF consultas, $(ls "$BASE"/bancos/banco-*.npy | wc -l | tr -d ' ') bancos, e5.onnx sha256 ${E5_SHA256:0:12}"

# --- 2. servidor Go: .chispa y tok (con sha), Go y ORT (con sha), compilación. Corre en segundo plano mientras se instala vLLM.
preparar_go() {
  set -u
  if [ "$LOCAL" = 1 ]; then
    s3cp "$MOVIL_S3/modelo-int8.chispa" "$BASE/modelo-int8.chispa" || return 1
    [ -f "$MOVIL_S3/e5.chispa-tok" ] && s3cp "$MOVIL_S3/e5.chispa-tok" "$BASE/e5.chispa-tok"
    echo "local: sha256 del .chispa y del tok NO comprobado (modelo de prueba)"
    TAGS=${CHISPA_LOCAL_TAGS:-}; ORTLIB=${CHISPA_LOCAL_ORT:-}
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
  # binario de la prueba de paridad de vectores (necesita -tags ort); se compila aquí para no gastar tiempo luego
  ( cd "$BASE/codigo/server" && go test $TAGS -c -o "$BASE/decide.test" ./decide ) || return 5
}
# cadena del e5 en segundo plano (tras compilar): paridad de vectores del ONNX fp32 y, informativo, int8 (cuantización dinámica de ORT en un venv
# con onnx + onnxruntime, que necesita red). Los resultados van a res/e5-*.txt y e5-int8*.json; ninguno decide nada aquí (el criterio es model/alt/crit-r10-t5b.json).
paridad_e5() {  # paridad_e5 ONNX N SALIDA
  ( cd "$BASE/codigo/server/decide" && CHISPA_PARIDAD_DIR="$BASE/bancos" CHISPA_CONSULTAS_TEXTO="$BASE/in/consultas.jsonl" CHISPA_E5_ONNX="$1" CHISPA_E5_TOK="$BASE/e5.chispa-tok" \
      CHISPA_ORT_LIB="$ORTLIB" CHISPA_E5_N="$2" "$BASE/decide.test" -test.run E5Paridad -test.v -test.timeout 20m ) > "$3" 2>&1
}
cadena_e5() {
  for _ in $(seq 600); do [ -f "$BASE/preparar-go.rc" ] && break; sleep 1; done
  [ "$(cat "$BASE/preparar-go.rc" 2>/dev/null || echo 9)" = 0 ] || { echo "sin binario de paridad"; return 1; }
  [ -x "$BASE/decide.test" ] || return 1
  ORTLIB=$(cat "$BASE/ort-lib" 2>/dev/null)
  paridad_e5 "$BASE/in/e5.onnx" "$PAR_N" "$OUT/e5-paridad-fp32.txt"; grep -E "coseno|kNN con|latencia|PASS|FAIL" "$OUT/e5-paridad-fp32.txt" | head -8
  [ "$INT8" = 1 ] && [ "$LOCAL" != 1 ] || return 0
  # int8 (informativo): venv con onnx y onnxruntime solo para cuantizar
  hay_tiempo 12 || { echo "int8 omitido por tiempo"; return 0; }
  python3 -m venv "$BASE/venvq" && "$BASE/venvq/bin/pip" install -q --no-cache-dir onnx onnxruntime numpy \
    && "$BASE/venvq/bin/python" -c "
from onnxruntime.quantization import QuantType, quantize_dynamic
import hashlib, json, os, sys
quantize_dynamic(sys.argv[1], sys.argv[2], weight_type=QuantType.QInt8)
h = hashlib.sha256(open(sys.argv[2], 'rb').read()).hexdigest()
json.dump({'bytes': os.path.getsize(sys.argv[2]), 'sha256': h, 'fp32_bytes': os.path.getsize(sys.argv[1])}, open(sys.argv[3], 'w'))
print('int8', h)" "$BASE/in/e5.onnx" "$BASE/e5-int8.onnx" "$OUT/e5-int8.json" || { echo "int8: no se pudo cuantizar"; return 0; }
  paridad_e5 "$BASE/e5-int8.onnx" "$INT8_N" "$OUT/e5-paridad-int8.txt"; grep -E "coseno|kNN con|latencia|PASS|FAIL" "$OUT/e5-paridad-int8.txt" | head -8
}
( preparar_go > "$BASE/preparar-go.log" 2>&1; echo $? > "$BASE/preparar-go.rc" ) &
PGO=$!
( cadena_e5 > "$BASE/cadena-e5.log" 2>&1 ) &
PE5=$!

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
  export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda 0be529b)
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

# --- 4. configuración: knn con los bancos de banco-paridad y el e5 fp32; L3 por logprobs con top_k 256 (ve todas las opciones, como en Python) y tau 0
# (la cascada de Python acepta siempre la respuesta de L3: sin tau 0 una L3 insegura escalaría y la respuesta sería la «mejor pista», no la de L3)
SERV=http://127.0.0.1:8090
"$PY" - "$BASE" "$URL" "${CHISPA_MODELO_L3:-qwen3-4b}" <<'PYEOF'
import json, sys
base, url, modelo = sys.argv[1:4]
ref = json.load(open(base + "/bancos/referencia.json"))
u = ref.get("u") or ref["regla"]["u"]
tareas = {}
for l in open(base + "/in/consultas.jsonl", encoding="utf-8"):
    r = json.loads(l)
    t = tareas.setdefault(r["tarea"], {"banco": base + "/bancos/banco-" + r["tarea"], "preguntas": []})
    if r["pregunta"] not in t["preguntas"]:
        t["preguntas"].append(r["pregunta"])
cfg = {"knn": {"umbral": u, "siempre": ref["siempre"], "tareas": tareas, "e5_onnx": base + "/in/e5.onnx"},
       "l3": {"url": url + "/v1", "modelo": modelo, "modo": "logprobs", "tau": 0.0, "max_concurrente": 32, "timeout": "30s", "timeout_fila": "20s", "top_k": 256, "max_por_peticion": 256}}
json.dump(cfg, open(base + "/cfg-knn.json", "w"), ensure_ascii=False)
print("knn: u=%s siempre=%s tareas=%s" % (u, ref["siempre"], sorted(tareas)))
PYEOF
arrancar() {
  parar_servidor
  ARGS=(-addr 127.0.0.1:8090 -modelo "$BASE/modelo-int8.chispa" -config "$BASE/cfg-knn.json" -timeout 120s -pruebas)
  [ -f "$BASE/e5.chispa-tok" ] && ARGS+=(-tok "$BASE/e5.chispa-tok"); [ -n "$ORTLIB" ] && ARGS+=(-ort-lib "$ORTLIB")
  nohup "$BASE/chispa-server" "${ARGS[@]}" > "$BASE/servidor-knn.log" 2>&1 &
  SRVPID=$!
  for _ in $(seq 180); do curl -s -m 2 $SERV/healthz > /dev/null && return 0; kill -0 $SRVPID 2>/dev/null || break; sleep 1; done
  tail -5 "$BASE/servidor-knn.log"; return 1
}
cli() { "$PY" "$BASE/codigo/model/alt/aws/r10_t5_cliente.py" "$@" --url $SERV --filas "$BASE/in/consultas.jsonl" --salida "$OUT" ${CHISPA_N:+--muestra $CHISPA_N}; }

# --- 5. fases
echo "== 5. fases $(date -u +%T) quedan $(quedan_min) min"
arrancar || { fallo "servidor no arrancó"; estado servidor-knn "FALLO"; cierre; }
mkdir -p "$BASE/auto"
"$PY" "$BASE/codigo/model/alt/aws/r10_t5_cliente.py" e2e --url $SERV --filas "$BASE/in/consultas.jsonl" --salida "$BASE/auto" --muestra 12 --clientes 2 > "$BASE/autoprueba.out" 2>&1 \
  && grep -q '"n": 12' "$BASE/auto/e2e-resumen.json" && ! grep -q '"200": 0' "$BASE/auto/e2e-resumen.json" && grep -q '"200"' "$BASE/auto/e2e-resumen.json" || { head -c 900 "$BASE/autoprueba.out"; tail -5 "$BASE/servidor-knn.log"; fallo "autoprueba: el servidor no contesta con kNN"; estado prueba "FALLO"; cierre; }
estado prueba "OK 12 consultas por HTTP con el codificador e5 en el servidor"
T1=$(date +%s)
cli e2e --clientes 16 --ref "$REF_PY" > "$OUT/e2e.out" 2>&1 && estado e2e "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json;d=json.load(open('$OUT/e2e-resumen.json'));print('macro sin hatecheck %s frente a %s (dif %s)' % (d['exactitud']['macro_sin_hatecheck'], d['referencia_python_macro_sin_hatecheck'], d['diferencia_macro_vs_python']))")" || { fallo "fase e2e"; estado e2e "FALLO"; }
for cl in $CARGA; do
  hay_tiempo 2 || { estado "carga-c$cl" "OMITIDO por tiempo (quedan $(quedan_min) min)"; continue; }
  T1=$(date +%s)
  cli carga --clientes "$cl" --dura "$DURA_P" > "$OUT/carga-c$cl.out" 2>&1 \
    && estado "carga-c$cl" "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json;d=json.load(open('$OUT/carga-c$cl-resumen.json'));print('%s req/s total p50=%s p95=%s ms' % (d['por_segundo'], d['latencia_ms']['total_cliente']['p50'], d['latencia_ms']['total_cliente']['p95']))")" \
    || { fallo "carga $cl"; estado "carga-c$cl" "FALLO"; }
done
wait $PE5 2>/dev/null; cat "$BASE/cadena-e5.log" 2>/dev/null
cierre
