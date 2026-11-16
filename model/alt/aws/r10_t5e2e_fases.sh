#!/bin/bash
# R10-t5e2e: fases de la ronda e2e (kNN de L2 con codificador e5 + L3 por logprobs en una g5). Lo carga con `.` el user-data (r10_t5e2e.sh) tras bajar y
# comprobar el sha256 de codigo.tgz; comparte sus variables y funciones (BASE, OUT, estado, fallo, hay_tiempo, cierre, sha_ok, s3cp, los pins de Go/ORT/.chispa/tok).
# Orden: preparar Go + .chispa + tok y compilar (en segundo plano), cadena del e5 (paridad fp32 e int8, informativa), vLLM, configuración, e2e y carga.
# El criterio de la paridad de vectores es model/alt/crit-r10-t5b.json; el macro de referencia de Python es REF_PY.
fases_main() {
DURA_P=${CHISPA_DURA_P:-50}; CARGA="${CHISPA_CARGA:-16 64}"; PAR_N=${CHISPA_PAR_N:-1000}; INT8_N=${CHISPA_INT8_N:-600}; INT8=${CHISPA_INT8:-1}; REF_PY=${CHISPA_REF_PY:-0.7233}
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
}
