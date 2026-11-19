#!/bin/bash
# W1c fase B: las carga con `.` el user-data w1_fs8b.sh tras comprobar el sha256 de codigo.tgz y dev.tgz (comparte BASE, OUT, LOCAL, MIN, REV4, VLLM, GO_*, ORT_*, S3A, FASEA_SHA256, RESERVA_PY, estado, fallo,
# hay_tiempo, cierre, sha_ok, s3cp, parar_servidor, SRVPID, VP). Criterio: model/alt/crit-w1-fs8-e2e.json; el veredicto lo calcula model/alt/w1_medir.py (cifras.json y cifras-parcial.json) al cerrar.
# Modo local (CHISPA_LOCAL=1, prueba en seco con vLLM falso y bancos de la fase A fabricados, ver w1_fs8ab_local.sh): CHISPA_PY, CHISPA_LOCAL_URL, CHISPA_LOCAL_ORT, CHISPA_N.
fases_main() {
CH=$BASE/codigo; FA=$BASE/faseA; ONX=$FA/onnx; BAN=$FA/bancos; ENT=$FA/ent; SERV=http://127.0.0.1:8090; HILOS=${CHISPA_HILOS:-4}
mkdir -p "$OUT/py" "$BASE/chunks" "$BASE/in"; export PYTHONPATH=$CH

# --- 2. Python con numpy (el ONNX ya viene hecho de la fase A: no hace falta torch)
echo "== 2. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}
else
  LIBRES=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
  [ "${LIBRES:-0}" -ge 60 ] || { fallo "volumen pequeno: ${LIBRES:-?} GB libres (hacen falta >= 60)"; estado entorno "FALLO disco"; cierre; }
  dnf install -y -q python3.11 python3.11-pip gcc tar gzip unzip || { fallo "dnf install"; estado entorno "FALLO dnf"; cierre; }
  python3.11 -m venv "$BASE/venv" && "$BASE/venv/bin/pip" install -q --no-cache-dir numpy || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
  PY=$BASE/venv/bin/python
fi
"$PY" -c "import model.alt.w1_filas, model.alt.w1_medir, model.alt.w1_demos, model.alt.lineas_base_v2, model.alt.lineas_base_v2_pred, model.alt.llm_logprobs, numpy" || { fallo "imports del codigo"; estado entorno "FALLO imports codigo"; cierre; }
NPT=$N_POR_TAREA; [ -n "${CHISPA_N:-}" ] && NPT=$(( CHISPA_N / 6 ))   # CHISPA_N: solo en pruebas (nº total de filas)

# --- 3. Go y ORT en segundo plano (con sha256), solo el servidor; los modulos salen de gomod-proxy/ del tgz
preparar_go() {
  set -u
  if [ "$LOCAL" = 1 ]; then TAGS="-tags ort"; ORTLIB=${CHISPA_LOCAL_ORT:-}; [ -n "$ORTLIB" ] || TAGS=""
  else
    [ "${GO_SHA256#__}" != "$GO_SHA256" ] && { echo "GO_SHA256 sin sustituir"; return 3; }
    curl -fsSL -o "$BASE/go.tgz" "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" && sha_ok "$GO_SHA256" "$BASE/go.tgz" || { echo "Go: descarga o sha256"; return 3; }
    tar -xzf "$BASE/go.tgz" -C /usr/local || return 3
    curl -fsSL -o "$BASE/ort.nupkg" "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${ORT_VERSION}" && sha_ok "$ORT_NUPKG_SHA256" "$BASE/ort.nupkg" || { echo "ORT: descarga o sha256 del nupkg"; return 3; }
    mkdir -p /opt/ort/lib && unzip -q -j -o "$BASE/ort.nupkg" runtimes/linux-x64/native/libonnxruntime.so runtimes/linux-x64/native/libonnxruntime_providers_shared.so -d /opt/ort/lib || return 3
    sha_ok "$ORT_SO_SHA256" /opt/ort/lib/libonnxruntime.so || { echo "ORT: sha256 de libonnxruntime.so"; return 3; }
    export PATH=/usr/local/go/bin:$PATH; TAGS="-tags ort"; ORTLIB=/opt/ort/lib/libonnxruntime.so
    export GOTOOLCHAIN=local GOFLAGS=-mod=readonly GOSUMDB=off GOPROXY=file://$CH/gomod-proxy GOCACHE=$BASE/gocache GOPATH=$BASE/gopath CGO_ENABLED=1
  fi
  echo "$ORTLIB" > "$BASE/ort-lib"; go version
  cd "$CH/server" && go build $TAGS -trimpath -o "$BASE/chispa-server" ./cmd/chispa-server || return 4
}
( preparar_go > "$BASE/preparar-go.log" 2>&1; echo $? > "$BASE/preparar-go.rc" ) &

# --- 4. vLLM + Qwen3-4B (revision fijada) en segundo plano: venv que ve el torch de la DLAMI y `uv pip` (si la version fijada de vLLM exige otro torch, uv instala el suyo en el venv; se deja escrito cual)
URL=${CHISPA_LOCAL_URL:-http://127.0.0.1:8000}
if [ "$LOCAL" != 1 ]; then
  PYG=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PYG=$c; break; }; done
  [ -n "$PYG" ] && "$PYG" -c "import torch; assert torch.cuda.is_available()" || { fallo "sin torch o sin GPU"; estado vllm "FALLO"; cierre; }
  arrancar_vllm() {
    "$PYG" -m venv --system-site-packages /opt/vllm && /opt/vllm/bin/pip install -q --no-cache-dir uv huggingface_hub || return 1
    echo "torch antes: $(/opt/vllm/bin/python -c 'import torch; print(torch.__version__)' 2>&1 | tail -1)"
    ( HF_HUB_ENABLE_HF_TRANSFER=0 /opt/vllm/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', revision='$REV4', local_dir='$BASE/modelo4b', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json','LICENSE*'])" ) & HFP=$!
    /opt/vllm/bin/uv pip install -q --python /opt/vllm/bin/python "vllm==$VLLM" || { echo "uv falló: pip"; /opt/vllm/bin/pip install -q --no-cache-dir "vllm==$VLLM" || { kill $HFP 2>/dev/null; return 3; }; }
    echo "torch después: $(/opt/vllm/bin/python -c 'import torch; print(torch.__version__, torch.__file__)' 2>&1 | tail -1)"
    wait $HFP || return 2
    export VLLM_USE_FLASHINFER_SAMPLER=0   # la DLAMI no trae nvcc (ronda 0be529b)
    nohup /opt/vllm/bin/python -m vllm.entrypoints.openai.api_server --model "$BASE/modelo4b" --served-model-name qwen3-4b --dtype bfloat16 --max-model-len 16384 \
      --gpu-memory-utilization 0.85 --max-logprobs 20 --enable-prefix-caching --host 127.0.0.1 --port 8000 > "$BASE/vllm-4b.log" 2>&1 &
    echo $! > "$BASE/vllm.pid"
    for _ in $(seq 180); do curl -s -m 3 localhost:8000/v1/models | grep -q qwen3-4b && return 0; kill -0 "$(cat "$BASE/vllm.pid")" 2>/dev/null || return 4; sleep 5; done; return 5
  }
  T1=$(date +%s)
  ( arrancar_vllm > "$BASE/vllm-arranque.log" 2>&1; echo $? > "$BASE/vllm.rc" ) &
fi

# --- 5. la fase A: faseA.json y faseA.tar por nombre exacto, sha256 del tar, de cada fichero, y que A usó ESTE codigo.tgz, dev.tgz y criterio; las entradas se regeneran de dev.tgz y se comparan con las de A
echo "== 5. fase A $(date -u +%T) quedan $(quedan_min) min"
T2=$(date +%s)
s3cp "$S3A/faseA.json" "$BASE/in/faseA.json" && s3cp "$S3A/faseA.tar" "$BASE/in/faseA.tar" || { fallo "faseA: descarga"; estado faseA "FALLO descarga"; cierre; }
mkdir -p "$FA"; tar -xf "$BASE/in/faseA.tar" -C "$FA" || { fallo "faseA.tar: no se extrae"; estado faseA "FALLO tar"; cierre; }
"$PY" - "$BASE/in/faseA.json" "$BASE/in/faseA.tar" "$FA" "$COD_SHA256" "$DEV_SHA256" "$CRIT_SHA" "$FASEA_SHA256" "$LOCAL" "$NPT" <<'PYEOF' || { fallo "faseA: sha256 o procedencia"; estado faseA "FALLO sha"; cierre; }
import hashlib, json, os, sys
js, tar, fa, cod, dev, crit, pin, local, npt = sys.argv[1:10]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
j = json.load(open(js))
def malo(m):
    print("faseA: " + m); sys.exit(1)
if j.get("fase") != "A": malo("no es una fase A")
if j["tar_sha256"] != sha(tar): malo("sha256 de faseA.tar distinto del de faseA.json")
if not pin.startswith("__") and pin != "-" and pin != j["tar_sha256"]: malo("sha256 de faseA.tar distinto del fijado en el ud")
for k, v in (("codigo_sha256", cod), ("dev_sha256", dev), ("criterio_sha256", crit)):
    if j[k] != v: malo("%s de A (%s) no es el de B (%s)" % (k, j[k][:12], v[:12]))
if local != "1" and not j["onnx_real"]: malo("A no exportó el ONNX real")
if j["n_por_tarea"] != int(npt): malo("n por tarea distinto (%s y %s)" % (j["n_por_tarea"], npt))
for n, s in j["ficheros"].items():
    if not os.path.isfile(os.path.join(fa, n)) or sha(os.path.join(fa, n)) != s: malo("fichero %s ausente o con otro sha256" % n)
hay = {os.path.relpath(os.path.join(d, f), fa) for d, _, fs in os.walk(fa) for f in fs} - {".sha256"}
if hay != set(j["ficheros"]): malo("ficheros de más o de menos: %s" % sorted(hay ^ set(j["ficheros"]))[:5])
print("faseA ok: %d ficheros, tar %s" % (len(hay), j["tar_sha256"][:12]))
PYEOF
CN=$(cd "$CH" && "$PY" -m model.alt.w1_filas --dev "$BASE/dev" --salida "$BASE/ent-chk" --n-por-tarea "$NPT")
diff -rq "$BASE/ent-chk" "$ENT" > "$BASE/ent-diff.txt" 2>&1 || { head -3 "$BASE/ent-diff.txt"; fallo "las entradas regeneradas de dev.tgz no son las de la fase A"; estado faseA "FALLO ent"; cierre; }
[ -f "$ENT/consultas.jsonl" ] && [ -f "$ENT/muestra.jsonl" ] || { fallo "faseA sin consultas o muestra"; estado faseA "FALLO ent"; cierre; }
estado faseA "OK sha256 del tar, de $(wc -l < "$FA/.sha256" | tr -d ' ') ficheros y procedencia; entradas iguales a las de dev.tgz ($CN) en $(( $(date +%s) - T2 )) s"
for _ in $(seq 900); do [ -f "$BASE/preparar-go.rc" ] && break; sleep 1; done
[ "$(cat "$BASE/preparar-go.rc" 2>/dev/null || echo 9)" = 0 ] || { tail -15 "$BASE/preparar-go.log"; fallo "servidor Go: preparar_go rc=$(cat "$BASE/preparar-go.rc" 2>/dev/null)"; estado servidor "FALLO"; cierre; }
ORTLIB=$(cat "$BASE/ort-lib"); estado servidor "OK $(head -1 "$BASE/preparar-go.log"); ORT $ORT_VERSION (so sha256 ${ORT_SO_SHA256:0:12})"
for t in $("$PY" -c "import json;print(' '.join(sorted(json.load(open('$ENT/tareas.json')))))"); do
  [ -f "$BAN/banco-$t.ejemplos.jsonl" ] && [ -f "$BAN/banco-$t.npy" ] && grep -q '"fila"' "$BAN/banco-$t.ejemplos.jsonl" || { fallo "banco de $t incompleto"; estado bancos "FALLO $t"; cierre; }
done

# --- 6. vLLM listo
if [ "$LOCAL" != 1 ]; then
  for _ in $(seq 600); do [ -f "$BASE/vllm.rc" ] && break; sleep 2; done
  [ "$(cat "$BASE/vllm.rc" 2>/dev/null || echo 9)" = 0 ] || { tail -5 "$BASE/vllm-arranque.log"; tail -5 "$BASE/vllm-4b.log"; fallo "vLLM no arrancó (rc=$(cat "$BASE/vllm.rc" 2>/dev/null))"; estado vllm "FALLO"; cierre; }
  VP=$(cat "$BASE/vllm.pid")
  estado vllm "OK vllm $(/opt/vllm/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null) Qwen3-4B-Instruct-2507@${REV4:0:12} en $(( $(date +%s) - T1 )) s; $(grep '^torch' "$BASE/vllm-arranque.log" | tr '\n' ';'); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
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
c["l3"].pop("demos_fijas", None)   # las demostraciones son las 8 más cercanas por consulta (el valor por defecto del modo gpu)
json.dump(c, open(salida, "w"), ensure_ascii=False, indent=1)
print("cfg: modo %s, %d tareas, ejemplos %s, top_k_knn %s" % (c["knn"]["modo"], len(ts), c["l3"].get("ejemplos"), c["l3"]["top_k_knn"]))
PYEOF
[ -f "$BASE/cfg-w1.json" ] || { fallo "config del servidor"; estado servidor-gpu "FALLO config"; cierre; }
[ "$LOCAL" = 1 ] && "$PY" -c "
import json;p='$BASE/cfg-w1.json';c=json.load(open(p));[c['knn'].pop(k,None) for k in ('qwen_onnx','qwen_tok','qwen_sesiones')];c['knn']['codificador']='e5-small';json.dump(c,open(p,'w'))"   # prueba en seco: sin codificador
ARGS=(-addr 127.0.0.1:8090 -config "$BASE/cfg-w1.json" -timeout 120s -demos-log "$OUT/demos-log.jsonl"); [ -n "$ORTLIB" ] && ARGS+=(-ort-lib "$ORTLIB")
[ "$LOCAL" = 1 ] && ARGS+=(-pruebas -codificador-prueba 8)   # prueba en seco: vectores inventados por hash (los de w1_banco_falso)
nohup "$BASE/chispa-server" "${ARGS[@]}" > "$BASE/servidor-w1.log" 2>&1 &
SRVPID=$!
for _ in $(seq 300); do curl -s -m 2 $SERV/healthz > /dev/null && break; kill -0 $SRVPID 2>/dev/null || break; sleep 1; done
curl -s -m 2 $SERV/healthz > /dev/null || { tail -8 "$BASE/servidor-w1.log"; fallo "el servidor no arrancó"; estado servidor-gpu "FALLO"; cierre; }
grep -q 'demostraciones fs8 por consulta' "$BASE/servidor-w1.log" || { grep -i 'fs8\|ejemplos' "$BASE/servidor-w1.log" | head -3; fallo "el servidor no usa demostraciones por consulta (kNN)"; estado servidor-gpu "FALLO demos kNN"; cierre; }
grep -qi 'zero-shot' "$BASE/servidor-w1.log" && { grep -i 'zero-shot' "$BASE/servidor-w1.log" | head -3; fallo "alguna tarea sin ejemplos.jsonl: L3 sería zero-shot"; estado servidor-gpu "FALLO zero-shot"; cierre; }
cli() { local f=$1; shift; "$PY" "$CH/model/alt/aws/r10_t5_cliente.py" "$@" --url $SERV --filas "$f" --salida "$OUT"; }
TAREAS=$("$PY" -c "import json;print(' '.join(sorted(json.load(open('$ENT/tareas.json')))))")
"$PY" - "$ENT/consultas.jsonl" "$BASE/chunks" <<'PYEOF'
import json, sys
por = {}
for l in open(sys.argv[1], encoding="utf-8"):
    if l.strip():
        por.setdefault(json.loads(l)["tarea"], []).append(l)
for t, ls in por.items():
    open("%s/%s.jsonl" % (sys.argv[2], t), "w", encoding="utf-8").writelines(ls)
PYEOF

# --- 7. e2e por HTTP: autoprueba de 12 filas (todas por L3 y con 200) y la medida con 4 clientes, UNA tarea cada vez (el cliente escribe sus filas al acabar: así una tarea cortada no pierde las anteriores)
echo "== 7. e2e por el servidor $(date -u +%T) quedan $(quedan_min) min"
cli "$ENT/consultas.jsonl" e2e --etiqueta auto --clientes 2 --muestra 12 --ref 0 > "$BASE/autoprueba.out" 2>&1 \
  && grep -q '"n": 12' "$OUT/e2e-auto-resumen.json" && "$PY" -c "import json,sys;d=json.load(open('$OUT/e2e-auto-resumen.json'));sys.exit(0 if d['http']=={'200':12} and d['capas'].get('L3')==12 else 1)" \
  || { head -c 700 "$BASE/autoprueba.out"; tail -5 "$BASE/servidor-w1.log"; fallo "autoprueba: el servidor no contesta 12 de 12 por L3"; estado e2e-servidor "FALLO autoprueba"; cierre; }
rm -f "$OUT"/e2e-auto-*
T3=$(date +%s); OK_T=""
for t in $TAREAS; do
  [ "$(quedan_min)" -gt "$RESERVA_PY" ] || { fallo "e2e: sin tiempo para $t (quedan $(quedan_min) min <= reserva $RESERVA_PY del fs8 de Python): cifras parciales"; continue; }
  T4=$(date +%s)
  if cli "$BASE/chunks/$t.jsonl" e2e --etiqueta "w1-$t" --clientes "$HILOS" --timeout-fila-ms 60000 --ref 0 > "$OUT/e2e-w1-$t.out" 2>&1 && [ -s "$OUT/e2e-w1-$t-filas.jsonl" ]; then
    OK_T="$OK_T $t"; estado "e2e-$t" "OK $(wc -l < "$OUT/e2e-w1-$t-filas.jsonl" | tr -d ' ') filas en $(( $(date +%s) - T4 )) s"
  else tail -3 "$OUT/e2e-w1-$t.out"; rm -f "$OUT/e2e-w1-$t-filas.jsonl"; fallo "e2e de $t"; estado "e2e-$t" "FALLO"; fi
done
for t in $OK_T; do cat "$OUT/e2e-w1-$t-filas.jsonl"; done > "$OUT/e2e-w1-filas.jsonl"
estado e2e-servidor "tareas servidas:${OK_T:- ninguna} en $(( $(date +%s) - T3 )) s; $("$PY" -c "
import json,collections
f=[json.loads(l) for l in open('$OUT/e2e-w1-filas.jsonl') if l.strip()]
print('n=%d http=%s capas=%s' % (len(f), dict(collections.Counter(str(r['http']) for r in f)), dict(collections.Counter(str(r['capa']) for r in f))))")"
parar_servidor

# --- 8. selección de demostraciones: la de Python (demos_fila) sobre los vectores de la muestra que dejó la fase A (vq/) frente al registro del servidor; las filas no servidas salen sin_registro
echo "== 8. selección de demostraciones $(date -u +%T) quedan $(quedan_min) min"
T3=$(date +%s)
if [ "$LOCAL" = 1 ]; then CMP=(--hash-dim 8); else CMP=(--vectores "$FA/vq"); fi
( cd "$CH" && "$PY" -m model.alt.w1_demos comparar --log "$OUT/demos-log.jsonl" --muestra "$ENT/muestra.jsonl" --bancos "$BAN" "${CMP[@]}" -k 8 --salida "$OUT/seleccion.json" ) > "$OUT/seleccion.txt" 2>&1 \
  && estado seleccion "$(cat "$OUT/seleccion.txt") en $(( $(date +%s) - T3 )) s" || { tail -3 "$OUT/seleccion.txt"; fallo "comparar selección"; estado seleccion "FALLO"; }

# --- 9. el fs8-kNN de Python sobre las filas de las tareas servidas y las demostraciones que eligió el servidor (ids del registro), contra el mismo vLLM; si el tope aprieta, las ya hechas quedan para cifras-parcial.json
echo "== 9. fs8-kNN de Python $(date -u +%T) quedan $(quedan_min) min"
( cd "$CH" && "$PY" -m model.alt.w1_demos escribir --log "$OUT/demos-log.jsonl" --test "$ENT/test" --salida "$OUT/demos" --salida-test "$OUT/test-demos" ) > "$BASE/escribir-demos.out" 2>&1 || fallo "demos del registro: $(tail -2 "$BASE/escribir-demos.out")"
urls() { for _ in $(seq "$2"); do printf -- '--url %s ' "$1"; done; }
T3=$(date +%s); NP=0
for t in $OK_T; do
  hay_tiempo 2 || { fallo "sin tiempo para $t en Python"; continue; }
  [ -s "$OUT/test-demos/$t.jsonl" ] || { fallo "$t: el servidor no registró ninguna fila"; continue; }
  T4=$(date +%s)
  ( cd "$CH" && "$PY" -m model.alt.llm_logprobs --filas "$OUT/test-demos/$t.jsonl" $(urls "$URL" "$HILOS") --modelo qwen3-4b --salida "$OUT/py/$t.jsonl" --timeout 600 --reintentos 1 \
      --ejemplos "$ENT/soporte/$t.jsonl" --demos "$OUT/demos/$t.jsonl" --k 8 ) > "$BASE/py-$t.log" 2>&1 || { tail -3 "$BASE/py-$t.log"; fallo "llm_logprobs $t"; }
  NP=$(( NP + $(wc -l < "$OUT/py/$t.jsonl" 2>/dev/null || echo 0) )); estado "py-$t" "$(wc -l < "$OUT/py/$t.jsonl" 2>/dev/null || echo 0) filas en $(( $(date +%s) - T4 )) s"
done
estado fs8-knn-python "$NP filas en $(( $(date +%s) - T3 )) s"
cierre
}
