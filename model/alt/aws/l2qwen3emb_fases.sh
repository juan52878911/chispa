#!/bin/bash
# L2-qwen3emb: fases de la medida (las carga con `.` el user-data l2qwen3emb.sh tras bajar y comprobar el sha256 de codigo.tgz y dev.tgz; comparte sus variables y
# funciones: BASE, OUT, LOCAL, MIN, REV, TRANSFORMERS, TORCH, ONNX, ORT_PY, GO_*, ORT_*, N_TEST, N_SOP, estado, fallo, hay_tiempo, cierre, sha_ok, s3cp, parar_servidor, SRVPID).
# Criterio congelado: model/alt/crit-l2-qwen3emb.json. Cada fase deja su cifra en $OUT; las veredictos los calcula model/alt/qwen3emb_medir.py (cifras.json) al cerrar.
# Modo local (CHISPA_LOCAL=1, pruebas con el Qwen3 en miniatura de model/alt/qwen3emb_mini.py): CHISPA_PY, CHISPA_LOCAL_MODELO, CHISPA_LOCAL_ORT, CHISPA_N (muestra de filas), CHISPA_PERMITIR_VERSIONES.
fases_main() {
CH=$BASE/codigo; MODELO=$BASE/modelo; ONX=$BASE/onnx; REF=$BASE/ref; BAN=$BASE/bancos; ENT=$BASE/ent; SERV=http://127.0.0.1:8090
mkdir -p "$ONX" "$REF" "$BAN" "$ENT"; export PYTHONPATH=$CH
PERMITIR=""; { [ "$LOCAL" = 1 ] || [ -n "${CHISPA_PERMITIR_VERSIONES:-}" ]; } && PERMITIR="--permitir-versiones"
LECTURAS=${CHISPA_N:-}   # solo en pruebas: nº de filas del e2e

# --- 2. entorno: se comprueba que importa todo ANTES de bajar nada
echo "== 2. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}
else
  LIBRES=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
  [ "${LIBRES:-0}" -ge 30 ] || { fallo "volumen pequeno: ${LIBRES:-?} GB libres (hacen falta >= 30)"; estado entorno "FALLO disco"; cierre; }
  dnf install -y -q python3.11 python3.11-pip gcc tar gzip || { fallo "dnf install"; estado entorno "FALLO dnf"; cierre; }
  python3.11 -m venv "$BASE/venv" && "$BASE/venv/bin/pip" install -q --no-cache-dir "torch==${TORCH}+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
    && "$BASE/venv/bin/pip" install -q --no-cache-dir "transformers==$TRANSFORMERS" "onnx==$ONNX" "onnxruntime==$ORT_PY" numpy huggingface_hub safetensors || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
  PY=$BASE/venv/bin/python
  # el modelo fp32 (2,4 GB) y su exportacion pasan de la RAM comoda de 8 GB: un swap de 12 GB evita el OOM
  fallocate -l 12G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile || echo "AVISO: sin swap"
fi
"$PY" -c "
import torch, transformers, onnx, onnxruntime, numpy, huggingface_hub, safetensors, tokenizers
if '$LOCAL' != '1':
    assert transformers.__version__ == '$TRANSFORMERS', transformers.__version__
    assert torch.__version__.startswith('$TORCH'), torch.__version__
    assert onnx.__version__ == '$ONNX' and onnxruntime.__version__ == '$ORT_PY'
" || { fallo "imports o versiones"; estado entorno "FALLO imports"; cierre; }
"$PY" -c "import model.alt.qwen3emb_medir, model.alt.aws.qwen3emb_filas, model.alt.lineas_base_v2, model.alt.fusion_knn_l2" || { fallo "imports del codigo"; estado entorno "FALLO imports codigo"; cierre; }
estado entorno "OK $("$PY" -c "import sys, torch, transformers, onnxruntime; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'onnxruntime', onnxruntime.__version__)" 2>&1 | tail -1); $(nproc) vCPU; $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); $(free -g | awk '/Mem:/{print $2}') GB RAM"
CN=$(cd "$CH" && "$PY" -m model.alt.lineas_base_v2 filas --dev "$BASE/dev" --salida "$BASE/t")
[ "$CN" = "tareas=6 test=$N_TEST soporte=$N_SOP" ] || { fallo "recuentos de dev distintos: $CN"; estado entradas "FALLO recuentos $CN"; cierre; }
estado entradas "OK dev: $CN"

# --- 3. servidor Go, Go y ORT en segundo plano (con sha256); los modulos salen de gomod-proxy/ del tgz
preparar_go() {
  set -u
  if [ "$LOCAL" = 1 ]; then TAGS="-tags ort"; ORTLIB=${CHISPA_LOCAL_ORT:?en local hace falta CHISPA_LOCAL_ORT}
  else
    [ "${GO_SHA256#__}" != "$GO_SHA256" ] && { echo "GO_SHA256 sin sustituir"; return 3; }
    curl -fsSL -o "$BASE/go.tgz" "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" && sha_ok "$GO_SHA256" "$BASE/go.tgz" || { echo "Go: descarga o sha256"; return 3; }
    tar -xzf "$BASE/go.tgz" -C /usr/local || return 3
    curl -fsSL -o "$BASE/ort.tgz" "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" && sha_ok "$ORT_SHA256" "$BASE/ort.tgz" || { echo "ORT: descarga o sha256"; return 3; }
    mkdir -p /opt/ort && tar -xzf "$BASE/ort.tgz" -C /opt/ort --strip-components=1 || return 3
    export PATH=/usr/local/go/bin:$PATH; TAGS="-tags ort"; ORTLIB=/opt/ort/lib/libonnxruntime.so
    export GOTOOLCHAIN=local GOFLAGS=-mod=readonly GOSUMDB=off GOPROXY=file://$CH/gomod-proxy GOCACHE=$BASE/gocache GOPATH=$BASE/gopath CGO_ENABLED=1
  fi
  echo "$ORTLIB" > "$BASE/ort-lib"; go version
  cd "$CH/server" && go build $TAGS -trimpath -o "$BASE/chispa-server" ./cmd/chispa-server && go build $TAGS -trimpath -o "$BASE/chispa-banco-qwen" ./cmd/chispa-banco-qwen \
    && go test $TAGS -c -o "$BASE/decide.test" ./decide || return 4
}
( preparar_go > "$BASE/preparar-go.log" 2>&1; echo $? > "$BASE/preparar-go.rc" ) &

# --- 4. modelo en la revision fijada, tokenizador exportado, ONNX fp32 e int8
echo "== 4. modelo $(date -u +%T) quedan $(quedan_min) min"
T1=$(date +%s)
if [ "$LOCAL" = 1 ]; then MODELO=${CHISPA_LOCAL_MODELO:?en local hace falta CHISPA_LOCAL_MODELO}
else
  HF_HUB_ENABLE_HF_TRANSFER=0 "$PY" -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-Embedding-0.6B', revision='$REV', local_dir='$MODELO', allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','LICENSE*','README.md'])" \
    || { fallo "descarga del modelo"; estado modelo "FALLO descarga"; cierre; }
  grep -qi '^license: apache-2.0' "$MODELO/README.md" || { fallo "la ficha no declara Apache-2.0"; estado modelo "FALLO licencia"; cierre; }
fi
[ -f "$MODELO/model.safetensors" ] && [ -f "$MODELO/tokenizer.json" ] && [ -f "$MODELO/config.json" ] || { fallo "faltan ficheros del modelo"; estado modelo "FALLO ficheros"; cierre; }
"$PY" -c "import hashlib,json,sys;d=sys.argv[1];f=lambda n:hashlib.sha256(open(d+'/'+n,'rb').read()).hexdigest();json.dump({'modelo':'Qwen/Qwen3-Embedding-0.6B','revision':'$REV','model.safetensors':f('model.safetensors'),'tokenizer.json':f('tokenizer.json'),'config.json':f('config.json')},open(sys.argv[2],'w'),indent=1)" "$MODELO" "$OUT/modelo.json"
estado modelo "OK revision ${REV:0:12}, safetensors sha256 $(sed -n 's/.*"model.safetensors": "\(.\{12\}\).*/\1/p' "$OUT/modelo.json") en $(( $(date +%s) - T1 )) s"
"$PY" "$CH/runtime/tok/exportar_bpe.py" "$MODELO" "$ONX/qwen3emb.chispa-bpe" > "$OUT/bpe.json" 2> "$BASE/bpe.err" || { tail -3 "$BASE/bpe.err"; fallo "exportar tokenizador"; estado tokenizador "FALLO"; cierre; }
estado tokenizador "OK $(cat "$OUT/bpe.json")"
T1=$(date +%s)
"$PY" "$CH/model/alt/exportar_qwen3emb_onnx.py" --modelo "$MODELO" --revision "$REV" --salida "$ONX" --comprobar $PERMITIR > "$BASE/exportar.out" 2> "$BASE/exportar.err" \
  || { tail -5 "$BASE/exportar.err"; fallo "exportar ONNX"; estado onnx "FALLO"; cierre; }
cp "$ONX/qwen3emb-onnx.json" "$OUT/onnx.json"; estado onnx "OK fp32 e int8 en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json;d=json.load(open('$OUT/onnx.json'));print({k:v['coseno_min_ultimo_token'] for k,v in d.get('comprobacion',{}).items()})")"

# --- 5. referencia de transformers (fp32, CPU, lotes con relleno a la izquierda) sobre el fixture de 200 textos
echo "== 5. referencia $(date -u +%T) quedan $(quedan_min) min"
T1=$(date +%s)
"$PY" "$CH/model/alt/qwen3emb_referencia.py" --modelo "$MODELO" --fixture "$CH/runtime/tok/testdata/qwen/fixture_paridad.jsonl" --salida "$REF" --revision "$REV" --lote 8 > "$BASE/referencia.out" 2> "$BASE/referencia.err" \
  || { tail -5 "$BASE/referencia.err"; fallo "referencia de transformers"; estado referencia "FALLO"; cierre; }
cp "$REF/referencia.json" "$OUT/referencia.json"; estado referencia "OK $(cat "$REF/referencia.json") en $(( $(date +%s) - T1 )) s"

# --- 6. paridad Go: tokens 200/200 y coseno por texto (fp32 decide; int8 informativo)
for _ in $(seq 900); do [ -f "$BASE/preparar-go.rc" ] && break; sleep 1; done
[ "$(cat "$BASE/preparar-go.rc" 2>/dev/null || echo 9)" = 0 ] || { tail -15 "$BASE/preparar-go.log"; fallo "servidor Go: preparar_go rc=$(cat "$BASE/preparar-go.rc" 2>/dev/null)"; estado servidor "FALLO"; cierre; }
ORTLIB=$(cat "$BASE/ort-lib"); estado servidor "OK $(head -1 "$BASE/preparar-go.log"); ORT $ORT_VERSION (sha256 ${ORT_SHA256:0:12})"
paridad() {  # paridad CONDICION ONNX
  ( cd "$CH/server/decide" && CHISPA_QWEN_TOK="$ONX/qwen3emb.chispa-bpe" CHISPA_QWEN_REF="$REF" CHISPA_QWEN_ONNX="$2" CHISPA_QWEN_CONDICION="$1" CHISPA_QWEN_SALIDA="$OUT" CHISPA_ORT_LIB="$ORTLIB" \
      "$BASE/decide.test" -test.run 'QwenTokensReal|QwenCosenoReal' -test.v -test.timeout 30m ) > "$OUT/paridad-$1.txt" 2>&1
}
paridad fp32 "$ONX/qwen3emb-fp32.onnx"; grep -E "tokens:|coseno|PASS|FAIL" "$OUT/paridad-fp32.txt" | head -6
if [ -f "$OUT/paridad-fp32.json" ]; then
  estado paridad "fp32: $("$PY" -c "import json;d=json.load(open('$OUT/paridad-fp32.json'));print('tokens %s/%s, coseno min %.6f, p50 codificador %.0f ms' % (d['tokens']['iguales'], d['tokens']['n'], d['coseno']['min'], d['latencia_codificador_ms']['p50']))")"
else fallo "paridad fp32 sin resultado"; estado paridad "FALLO"; fi
hay_tiempo 10 && { paridad int8 "$ONX/qwen3emb-int8.onnx"; grep -E "coseno" "$OUT/paridad-int8.txt" | head -2; }

# --- 7. bancos de las 6 tareas dev con la herramienta Go (8 de oro por clase, como muestrear(ys,8,0) de R11), uno por ONNX
echo "== 7. bancos $(date -u +%T) quedan $(quedan_min) min"
( cd "$CH" && "$PY" -m model.alt.aws.qwen3emb_filas filas --dev "$BASE/dev" --salida "$ENT" ) > "$BASE/filas.out" 2>&1 || { cat "$BASE/filas.out"; fallo "filas"; estado bancos "FALLO filas"; cierre; }
cp "$ENT/tareas.json" "$OUT/tareas.json"; echo "$(cat "$BASE/filas.out")" > "$OUT/filas.txt"
bancos() {  # bancos MODO
  T1=$(date +%s); mkdir -p "$BAN/$1"
  "$BASE/chispa-banco-qwen" -onnx "$ONX/qwen3emb-$1.onnx" -tok "$ONX/qwen3emb.chispa-bpe" -soporte "$ENT/soporte.jsonl" -salida "$BAN/$1" -ort-lib "$ORTLIB" > "$BASE/bancos-$1.log" 2>&1 \
    && estado "bancos-$1" "OK $(cat "$BASE/filas.out") en $(( $(date +%s) - T1 )) s" || { tail -3 "$BASE/bancos-$1.log"; fallo "bancos $1"; estado "bancos-$1" "FALLO"; return 1; }
  cp "$BAN/$1/bancos.json" "$OUT/bancos-$1.json"
}
bancos fp32 || cierre

# --- 8. servidor con L3 apagado y sin escalado (umbral 0, sin «siempre»): contesta solo L2; e2e por HTTP con un cliente (latencia por consulta limpia)
echo "== 8. e2e $(date -u +%T) quedan $(quedan_min) min"
: > "$OUT/rss.tsv"
arrancar() {  # arrancar MODO VECINOS
  parar_servidor
  ( cd "$CH" && "$PY" -m model.alt.aws.qwen3emb_filas config --tareas "$ENT/tareas.json" --bancos "$BAN/$1" --onnx "$ONX/qwen3emb-$1.onnx" --tok "$ONX/qwen3emb.chispa-bpe" --vecinos "$2" --salida "$BASE/cfg-$1-v$2.json" ) || return 1
  nohup "$BASE/chispa-server" -addr 127.0.0.1:8090 -config "$BASE/cfg-$1-v$2.json" -timeout 120s -ort-lib "$ORTLIB" > "$BASE/servidor-$1-v$2.log" 2>&1 &
  SRVPID=$!
  for _ in $(seq 300); do curl -s -m 2 $SERV/healthz > /dev/null && return 0; kill -0 $SRVPID 2>/dev/null || break; sleep 1; done
  tail -5 "$BASE/servidor-$1-v$2.log"; return 1
}
cli() { "$PY" "$CH/model/alt/aws/r10_t5_cliente.py" "$@" --url $SERV --filas "$ENT/consultas.jsonl" --salida "$OUT"; }
e2e() {  # e2e MODO VECINOS
  local c="$1-v$2"; T1=$(date +%s)
  arrancar "$1" "$2" || { fallo "servidor $c no arranco"; estado "e2e-$c" "FALLO servidor"; return 1; }
  mkdir -p "$BASE/auto"
  cli e2e --etiqueta "auto-$c" --clientes 1 --muestra 12 > "$BASE/autoprueba.out" 2>&1 && grep -q '"n": 12' "$OUT/e2e-auto-$c-resumen.json" && grep -q '"200"' "$OUT/e2e-auto-$c-resumen.json" \
    || { head -c 600 "$BASE/autoprueba.out"; tail -5 "$BASE/servidor-$c.log"; fallo "autoprueba $c: el servidor no contesta con kNN"; estado "e2e-$c" "FALLO autoprueba"; return 1; }
  rm -f "$OUT/e2e-auto-$c-"*
  cli e2e --etiqueta "$c" --clientes 1 ${LECTURAS:+--muestra $LECTURAS} > "$OUT/e2e-$c.out" 2>&1 || { tail -3 "$OUT/e2e-$c.out"; fallo "e2e $c"; estado "e2e-$c" "FALLO"; return 1; }
  printf '%s\t%s\t%s\n' "$c" "$(awk '/VmHWM/{print $2}' /proc/$SRVPID/status)" "$(awk '/VmRSS/{print $2}' /proc/$SRVPID/status)" >> "$OUT/rss.tsv"
  estado "e2e-$c" "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json;d=json.load(open('$OUT/e2e-$c-resumen.json'));l=d['latencia_ms'];print('n=%s http=%s total p50=%s p95=%s ms; codificador p50=%s p95=%s ms' % (d['n'], d['http'], l['total_cliente']['p50'], l['total_cliente']['p95'], l['cod_ms']['p50'], l['cod_ms']['p95']))")"
  grep -q 'Aviso\|panic' "$BASE/servidor-$c.log" && grep -m3 'Aviso\|panic' "$BASE/servidor-$c.log"
  return 0
}
DUR=0
for combo in "fp32 32" "fp32 8" "int8 32" "int8 8"; do
  set -- $combo
  if [ "$1" = int8 ] && [ ! -d "$BAN/int8" ]; then hay_tiempo $((DUR / 60 * 2 + 8)) && bancos int8 || { estado "e2e-$1-v$2" "OMITIDO (bancos int8)"; continue; }; fi
  hay_tiempo $(( DUR / 60 + 4 )) || { estado "e2e-$1-v$2" "OMITIDO por tiempo (quedan $(quedan_min) min)"; continue; }
  t0=$(date +%s); e2e "$1" "$2"; d=$(( $(date +%s) - t0 )); [ "$d" -gt "$DUR" ] && DUR=$d
done
parar_servidor

# --- 9. RSS, tamanos (sin docker en el nodo: la imagen se estima con lo que copia server/deploy/Dockerfile.producto) y cierre
"$PY" - "$OUT" "$BASE" "$ONX" "$BAN" "$ORTLIB" <<'PYEOF'
import json, os, sys
out, base, onx, ban, lib = sys.argv[1:6]
rss = {}
for l in open(out + "/rss.tsv"):
    c, hwm, rs = l.split("\t")
    rss[c] = {"VmHWM_kB": int(hwm), "VmRSS_kB": int(rs)}
json.dump(rss, open(out + "/rss.json", "w"), indent=1)
sz = lambda p: os.path.getsize(p) if os.path.exists(p) else None
du = lambda d: sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) if os.path.isdir(d) else None
t = {"servidor_bytes": sz(base + "/chispa-server"), "banco_qwen_bytes": sz(base + "/chispa-banco-qwen"), "libonnxruntime_bytes": sz(lib), "tokenizador_bytes": sz(onx + "/qwen3emb.chispa-bpe"),
     "onnx_fp32_bytes": (sz(onx + "/qwen3emb-fp32.onnx") or 0) + (sz(onx + "/qwen3emb-fp32.onnx.data") or 0), "onnx_int8_bytes": (sz(onx + "/qwen3emb-int8.onnx") or 0) + (sz(onx + "/qwen3emb-int8.onnx.data") or 0),
     "bancos_fp32_bytes": du(ban + "/fp32"), "bancos_int8_bytes": du(ban + "/int8"), "e5_onnx_fp32_bytes_segun_server_README": 470_000_000}
t["imagen_estimada_bytes"] = {m: sum(t[k] or 0 for k in ("servidor_bytes", "libonnxruntime_bytes", "tokenizador_bytes", "onnx_%s_bytes" % m, "bancos_%s_bytes" % m)) for m in ("fp32", "int8")}
t["nota_imagen"] = "estimacion: no hay docker en el nodo; suma de lo que copia server/deploy/Dockerfile.producto (servidor, libonnxruntime, ONNX, tokenizador, bancos) SIN la capa base"
json.dump(t, open(out + "/tamanos.json", "w"), indent=1)
print(json.dumps(t))
PYEOF
cierre
}
