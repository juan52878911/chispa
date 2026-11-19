#!/bin/bash
# W1-fs8: fases de la medida (las carga con `.` el user-data w1_fs8.sh tras bajar y comprobar el sha256 de codigo.tgz y dev.tgz; comparte sus variables y funciones: BASE, OUT, LOCAL, MIN,
# REV4, REVE, VLLM, TRANSFORMERS, TORCH, ONNX, ORT_PY, GO_*, ORT_*, N_POR_TAREA, estado, fallo, hay_tiempo, cierre, sha_ok, s3cp, parar_servidor, SRVPID, VP).
# Criterio: model/alt/crit-w1-fs8-e2e.json. Cada fase deja su cifra en $OUT; el veredicto lo calcula model/alt/w1_medir.py (cifras.json) al cerrar.
# Modo local (CHISPA_LOCAL=1, prueba en seco con vLLM falso y bancos fabricados, ver w1_fs8_local.sh): CHISPA_PY, CHISPA_LOCAL_URL, CHISPA_LOCAL_ORT, CHISPA_LOCAL_BANCOS, CHISPA_N.
fases_main() {
CH=$BASE/codigo; MODELOE=$BASE/modeloe; ONX=$BASE/onnx; BAN=$BASE/bancos; ENT=$BASE/ent; SERV=http://127.0.0.1:8090; HILOS=${CHISPA_HILOS:-4}
mkdir -p "$ONX" "$BAN" "$ENT" "$OUT/py"; export PYTHONPATH=$CH
PERMITIR=""; { [ "$LOCAL" = 1 ] || [ -n "${CHISPA_PERMITIR_VERSIONES:-}" ]; } && PERMITIR="--permitir-versiones"

# --- 2. entorno de Python (torch CPU para exportar el ONNX de Qwen3-Embedding) y entradas de la medida
echo "== 2. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}
else
  LIBRES=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
  [ "${LIBRES:-0}" -ge 60 ] || { fallo "volumen pequeno: ${LIBRES:-?} GB libres (hacen falta >= 60)"; estado entorno "FALLO disco"; cierre; }
  dnf install -y -q python3.11 python3.11-pip gcc tar gzip unzip || { fallo "dnf install"; estado entorno "FALLO dnf"; cierre; }
  python3.11 -m venv "$BASE/venv" && "$BASE/venv/bin/pip" install -q --no-cache-dir "torch==${TORCH}+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
    && "$BASE/venv/bin/pip" install -q --no-cache-dir "transformers==$TRANSFORMERS" "onnx==$ONNX" "onnxruntime==$ORT_PY" numpy huggingface_hub safetensors || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
  PY=$BASE/venv/bin/python
fi
"$PY" -c "import model.alt.w1_filas, model.alt.w1_medir, model.alt.lineas_base_v2, model.alt.llm_logprobs, numpy" || { fallo "imports del codigo"; estado entorno "FALLO imports codigo"; cierre; }
NPT=$N_POR_TAREA; [ -n "${CHISPA_N:-}" ] && NPT=$(( CHISPA_N / 6 ))   # CHISPA_N: solo en pruebas (nº total de filas)
CN=$(cd "$CH" && "$PY" -m model.alt.w1_filas --dev "$BASE/dev" --salida "$ENT" --n-por-tarea "$NPT")
case "$CN" in "tareas=6 soporte="*" test="*) ;; *) fallo "w1_filas: $CN"; estado entradas "FALLO filas"; cierre;; esac
[ -f "$ENT/consultas.jsonl" ] && estado entradas "OK dev (test $N_TEST, soporte $N_SOP) -> $CN (n por tarea $NPT)"

# --- 3. Go y ORT en segundo plano (con sha256); los modulos salen de gomod-proxy/ del tgz
preparar_go() {
  set -u
  if [ "$LOCAL" = 1 ]; then TAGS="-tags ort"; ORTLIB=${CHISPA_LOCAL_ORT:-}; [ -n "$ORTLIB" ] || TAGS=""
  else
    [ "${GO_SHA256#__}" != "$GO_SHA256" ] && { echo "GO_SHA256 sin sustituir"; return 3; }
    curl -fsSL -o "$BASE/go.tgz" "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" && sha_ok "$GO_SHA256" "$BASE/go.tgz" || { echo "Go: descarga o sha256"; return 3; }
    tar -xzf "$BASE/go.tgz" -C /usr/local || return 3
    # ORT desde NuGet (el repo de GitHub da 404): el .nupkg es un zip; solo se sacan las dos .so
    curl -fsSL -o "$BASE/ort.nupkg" "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${ORT_VERSION}" && sha_ok "$ORT_NUPKG_SHA256" "$BASE/ort.nupkg" || { echo "ORT: descarga o sha256 del nupkg"; return 3; }
    mkdir -p /opt/ort/lib && unzip -q -j -o "$BASE/ort.nupkg" runtimes/linux-x64/native/libonnxruntime.so runtimes/linux-x64/native/libonnxruntime_providers_shared.so -d /opt/ort/lib || return 3
    sha_ok "$ORT_SO_SHA256" /opt/ort/lib/libonnxruntime.so || { echo "ORT: sha256 de libonnxruntime.so"; return 3; }
    export PATH=/usr/local/go/bin:$PATH; TAGS="-tags ort"; ORTLIB=/opt/ort/lib/libonnxruntime.so
    export GOTOOLCHAIN=local GOFLAGS=-mod=readonly GOSUMDB=off GOPROXY=file://$CH/gomod-proxy GOCACHE=$BASE/gocache GOPATH=$BASE/gopath CGO_ENABLED=1
  fi
  echo "$ORTLIB" > "$BASE/ort-lib"; go version
  cd "$CH/server" && go build $TAGS -trimpath -o "$BASE/chispa-server" ./cmd/chispa-server && go build $TAGS -trimpath -o "$BASE/chispa-banco-qwen" ./cmd/chispa-banco-qwen || return 4
}
( preparar_go > "$BASE/preparar-go.log" 2>&1; echo $? > "$BASE/preparar-go.rc" ) &

# --- 4. vLLM (pip + modelo 4B en la revision fijada) en segundo plano: arranca mientras se exporta el ONNX y se hacen los bancos
URL=${CHISPA_LOCAL_URL:-http://127.0.0.1:8000}
if [ "$LOCAL" != 1 ]; then
  PYG=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PYG=$c; break; }; done
  [ -n "$PYG" ] && "$PYG" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado vllm "FALLO"; cierre; }
  arrancar_vllm() {
    "$PYG" -m venv /opt/hf && /opt/hf/bin/pip install -q --no-cache-dir huggingface_hub || return 1
    HF_HUB_ENABLE_HF_TRANSFER=0 /opt/hf/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', revision='$REV4', local_dir='$BASE/modelo4b', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json','LICENSE*'])" || return 2
    "$PYG" -m venv /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" || return 3
    export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda 0be529b)
    nohup /opt/vllm/bin/python -m vllm.entrypoints.openai.api_server --model "$BASE/modelo4b" --served-model-name qwen3-4b --dtype bfloat16 --max-model-len 16384 \
      --gpu-memory-utilization 0.85 --max-logprobs 20 --enable-prefix-caching --host 127.0.0.1 --port 8000 > "$BASE/vllm-4b.log" 2>&1 &
    echo $! > "$BASE/vllm.pid"
    for _ in $(seq 144); do curl -s -m 3 localhost:8000/v1/models | grep -q qwen3-4b && return 0; kill -0 "$(cat "$BASE/vllm.pid")" 2>/dev/null || return 4; sleep 5; done; return 5
  }
  T1=$(date +%s)
  ( arrancar_vllm > "$BASE/vllm-arranque.log" 2>&1; echo $? > "$BASE/vllm.rc" ) &
fi

# --- 5. Qwen3-Embedding en la revision fijada, tokenizador y ONNX fp32 (sin int8: no viable, L2-qwen3emb)
echo "== 5. modelo de embeddings $(date -u +%T) quedan $(quedan_min) min"
T2=$(date +%s)
if [ "$LOCAL" = 1 ]; then ONNX_LISTO=${CHISPA_LOCAL_BANCOS:+si}
else
  HF_HUB_ENABLE_HF_TRANSFER=0 "$PY" -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-Embedding-0.6B', revision='$REVE', local_dir='$MODELOE', allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','LICENSE*','README.md'])" \
    || { fallo "descarga del modelo de embeddings"; estado modelo "FALLO descarga"; cierre; }
  grep -qi '^license: apache-2.0' "$MODELOE/README.md" || { fallo "la ficha no declara Apache-2.0"; estado modelo "FALLO licencia"; cierre; }
  [ -f "$MODELOE/model.safetensors" ] && [ -f "$MODELOE/tokenizer.json" ] && [ -f "$MODELOE/config.json" ] || { fallo "faltan ficheros del modelo"; estado modelo "FALLO ficheros"; cierre; }
  "$PY" "$CH/runtime/tok/exportar_bpe.py" "$MODELOE" "$ONX/qwen3emb.chispa-bpe" > "$OUT/bpe.json" 2> "$BASE/bpe.err" || { tail -3 "$BASE/bpe.err"; fallo "exportar tokenizador"; estado tokenizador "FALLO"; cierre; }
  "$PY" "$CH/model/alt/exportar_qwen3emb_onnx.py" --modelo "$MODELOE" --revision "$REVE" --salida "$ONX" --sin-int8 --comprobar $PERMITIR > "$BASE/exportar.out" 2> "$BASE/exportar.err" \
    || { tail -5 "$BASE/exportar.err"; fallo "exportar ONNX"; estado onnx "FALLO"; cierre; }
  cp "$ONX/qwen3emb-onnx.json" "$OUT/onnx.json" 2>/dev/null
  estado onnx "OK Qwen3-Embedding@${REVE:0:12} fp32 en $(( $(date +%s) - T2 )) s"
fi
for _ in $(seq 900); do [ -f "$BASE/preparar-go.rc" ] && break; sleep 1; done
[ "$(cat "$BASE/preparar-go.rc" 2>/dev/null || echo 9)" = 0 ] || { tail -15 "$BASE/preparar-go.log"; fallo "servidor Go: preparar_go rc=$(cat "$BASE/preparar-go.rc" 2>/dev/null)"; estado servidor "FALLO"; cierre; }
ORTLIB=$(cat "$BASE/ort-lib"); estado servidor "OK $(head -1 "$BASE/preparar-go.log"); ORT $ORT_VERSION (so sha256 ${ORT_SO_SHA256:0:12})"

# --- 6. bancos con la herramienta Go (8 de oro por clase) y sus <banco>.ejemplos.jsonl (las 8 demostraciones fs8 de L3)
echo "== 6. bancos $(date -u +%T) quedan $(quedan_min) min"
T2=$(date +%s)
if [ -n "${CHISPA_LOCAL_BANCOS:-}" ]; then cp -r "$CHISPA_LOCAL_BANCOS/." "$BAN/"   # prueba en seco: bancos fabricados (no hay ONNX de Qwen3-Embedding)
else
  "$BASE/chispa-banco-qwen" -onnx "$ONX/qwen3emb-fp32.onnx" -tok "$ONX/qwen3emb.chispa-bpe" -soporte "$ENT/soporte.jsonl" -salida "$BAN" -k 8 -ort-lib "$ORTLIB" > "$BASE/bancos.log" 2>&1 \
    || { tail -3 "$BASE/bancos.log"; fallo "bancos"; estado bancos "FALLO"; cierre; }
fi
for t in $("$PY" -c "import json;print(' '.join(sorted(json.load(open('$ENT/tareas.json')))))"); do
  [ -f "$BAN/banco-$t.ejemplos.jsonl" ] && [ -f "$BAN/banco-$t.npy" ] || { fallo "banco de $t sin .npy o .ejemplos.jsonl"; estado bancos "FALLO $t"; cierre; }
  printf '%s\t%s\n' "$t" "$(wc -l < "$BAN/banco-$t.ejemplos.jsonl" | tr -d ' ')" >> "$OUT/ejemplos-por-tarea.tsv"
done
cp "$BAN/bancos.json" "$OUT/bancos.json" 2>/dev/null
estado bancos "OK en $(( $(date +%s) - T2 )) s; ejemplos fs8 por tarea: $(tr '\t\n' '= ' < "$OUT/ejemplos-por-tarea.tsv")"

# --- 7. vLLM listo
if [ "$LOCAL" != 1 ]; then
  for _ in $(seq 600); do [ -f "$BASE/vllm.rc" ] && break; sleep 2; done
  [ "$(cat "$BASE/vllm.rc" 2>/dev/null || echo 9)" = 0 ] || { tail -5 "$BASE/vllm-arranque.log"; tail -5 "$BASE/vllm-4b.log"; fallo "vLLM no arrancó (rc=$(cat "$BASE/vllm.rc" 2>/dev/null))"; estado vllm "FALLO"; cierre; }
  VP=$(cat "$BASE/vllm.pid")
  estado vllm "OK vllm $(/opt/vllm/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null) Qwen3-4B-Instruct-2507@${REV4:0:12} en $(( $(date +%s) - T1 )) s; $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
fi

# --- 8. servidor en modo gpu: config.modo-gpu.json del repo con las 6 tareas, los bancos de Qwen, top_k_knn 0 (L3 ve todas las opciones, como Python), tau 0 y timeout_fila 60 s
"$PY" - "$CH/server/deploy/config.modo-gpu.json" "$ENT/tareas.json" "$BAN" "$ONX" "$URL" "$BASE/cfg-w1.json" <<'PYEOF'
import json, sys
plantilla, tareas, ban, onx, url, salida = sys.argv[1:7]
c = json.load(open(plantilla))
c.pop("api_keys", None)
ts = json.load(open(tareas))
c["knn"]["tareas"] = {t: {"banco": "%s/banco-%s" % (ban, t), "preguntas": v["preguntas"]} for t, v in sorted(ts.items())}
if "qwen_onnx" in c["knn"]:
    c["knn"]["qwen_onnx"], c["knn"]["qwen_tok"] = onx + "/qwen3emb-fp32.onnx", onx + "/qwen3emb.chispa-bpe"
c["l3"].update({"url": url + "/v1", "modelo": "qwen3-4b", "tau": 0.0, "top_k_knn": 0, "timeout": "60s", "timeout_fila": "60s"})
json.dump(c, open(salida, "w"), ensure_ascii=False, indent=1)
print("cfg: modo %s, %d tareas, ejemplos %s, top_k_knn %s" % (c["knn"]["modo"], len(ts), c["l3"].get("ejemplos"), c["l3"]["top_k_knn"]))
PYEOF
[ -f "$BASE/cfg-w1.json" ] || { fallo "config del servidor"; estado servidor-gpu "FALLO config"; cierre; }
[ -n "${CHISPA_LOCAL_BANCOS:-}" ] && "$PY" -c "
import json;p='$BASE/cfg-w1.json';c=json.load(open(p));[c['knn'].pop(k,None) for k in ('qwen_onnx','qwen_tok','qwen_sesiones')];c['knn']['codificador']='e5-small';json.dump(c,open(p,'w'))"   # prueba en seco: sin codificador
ARGS=(-addr 127.0.0.1:8090 -config "$BASE/cfg-w1.json" -timeout 120s); [ -n "$ORTLIB" ] && ARGS+=(-ort-lib "$ORTLIB")
nohup "$BASE/chispa-server" "${ARGS[@]}" > "$BASE/servidor-w1.log" 2>&1 &
SRVPID=$!
for _ in $(seq 300); do curl -s -m 2 $SERV/healthz > /dev/null && break; kill -0 $SRVPID 2>/dev/null || break; sleep 1; done
curl -s -m 2 $SERV/healthz > /dev/null || { tail -8 "$BASE/servidor-w1.log"; fallo "el servidor no arrancó"; estado servidor-gpu "FALLO"; cierre; }
grep -qi 'zero-shot' "$BASE/servidor-w1.log" && { grep -i 'zero-shot' "$BASE/servidor-w1.log" | head -3; fallo "alguna tarea sin ejemplos.jsonl: L3 sería zero-shot"; estado servidor-gpu "FALLO zero-shot"; cierre; }
cli() { "$PY" "$CH/model/alt/aws/r10_t5_cliente.py" "$@" --url $SERV --filas "$ENT/consultas.jsonl" --salida "$OUT"; }

# --- 9. e2e por HTTP: autoprueba de 12 filas (todas por L3 y con 200) y la medida con 4 clientes
echo "== 9. e2e por el servidor $(date -u +%T) quedan $(quedan_min) min"
cli e2e --etiqueta auto --clientes 2 --muestra 12 --ref 0 > "$BASE/autoprueba.out" 2>&1 \
  && grep -q '"n": 12' "$OUT/e2e-auto-resumen.json" && "$PY" -c "import json,sys;d=json.load(open('$OUT/e2e-auto-resumen.json'));sys.exit(0 if d['http']=={'200':12} and d['capas'].get('L3')==12 else 1)" \
  || { head -c 700 "$BASE/autoprueba.out"; tail -5 "$BASE/servidor-w1.log"; fallo "autoprueba: el servidor no contesta 12 de 12 por L3"; estado e2e-servidor "FALLO autoprueba"; cierre; }
rm -f "$OUT"/e2e-auto-*
T3=$(date +%s)
cli e2e --etiqueta w1 --clientes "$HILOS" --timeout-fila-ms 60000 --ref 0 > "$OUT/e2e-w1.out" 2>&1 \
  && estado e2e-servidor "OK en $(( $(date +%s) - T3 )) s: $("$PY" -c "import json;d=json.load(open('$OUT/e2e-w1-resumen.json'));l=d['latencia_ms'];print('n=%s http=%s capas=%s total p50=%s p95=%s ms; L3 p50=%s' % (d['n'], d['http'], d['capas'], l['total_cliente']['p50'], l['total_cliente']['p95'], l['l3_ms']['p50']))")" \
  || { tail -3 "$OUT/e2e-w1.out"; fallo "e2e por el servidor"; estado e2e-servidor "FALLO"; }
parar_servidor

# --- 10. el fs8 de Python sobre las MISMAS filas y las mismas demostraciones (soporte de 8 por clase, semilla de siempre), contra el mismo vLLM
echo "== 10. fs8 de Python $(date -u +%T) quedan $(quedan_min) min"
urls() { for _ in $(seq "$2"); do printf -- '--url %s ' "$1"; done; }
T3=$(date +%s); NP=0
for t in $("$PY" -c "import json;print(' '.join(sorted(json.load(open('$ENT/tareas.json')))))"); do
  hay_tiempo 2 || { fallo "sin tiempo para $t en Python"; continue; }
  ( cd "$CH" && "$PY" -m model.alt.llm_logprobs --filas "$ENT/test/$t.jsonl" $(urls "$URL" "$HILOS") --modelo qwen3-4b --salida "$OUT/py/$t.jsonl" --timeout 600 --reintentos 1 \
      --ejemplos "$ENT/soporte/$t.jsonl" --k 8 ) > "$BASE/py-$t.log" 2>&1 || { tail -3 "$BASE/py-$t.log"; fallo "llm_logprobs $t"; }
  NP=$(( NP + $(wc -l < "$OUT/py/$t.jsonl" 2>/dev/null || echo 0) ))
done
estado fs8-python "$NP filas en $(( $(date +%s) - T3 )) s"
cierre
}
