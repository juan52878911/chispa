#!/bin/bash
# W1c fase A: las carga con `.` el user-data w1_fs8a.sh tras comprobar el sha256 de codigo.tgz y dev.tgz (comparte BASE, OUT, LOCAL, MIN, REVE, TORCH..., estado, fallo, cierre, sha_ok, s3cp, quedan_min).
# Deja en $BASE/faseA/ (y sube como faseA.tar + faseA.json): onnx/ (qwen3emb-fp32.onnx[.data], qwen3emb.chispa-bpe, qwen3emb-onnx.json), bancos/ (banco-<t>.npy/.jsonl/.ejemplos.jsonl, bancos.json),
# vq/ (vectores de la muestra de selección, banco-<t>.npy), ent/ (consultas.jsonl, soporte.jsonl, muestra.jsonl, tareas.json: la fase B los regenera de dev.tgz y comprueba su sha256).
# Modo local (prueba en seco): CHISPA_PY, CHISPA_LOCAL_BANCOS (bancos fabricados con w1_banco_falso: no hay ONNX ni vq), CHISPA_N.
fases_main() {
CH=$BASE/codigo; ONX=$BASE/faseA/onnx; BAN=$BASE/faseA/bancos; ENT=$BASE/faseA/ent; VQ=$BASE/faseA/vq; MODELOE=$BASE/modeloe
mkdir -p "$ONX" "$BAN" "$ENT" "$VQ"; export PYTHONPATH=$CH
PERMITIR=""; { [ "$LOCAL" = 1 ] || [ -n "${CHISPA_PERMITIR_VERSIONES:-}" ]; } && PERMITIR="--permitir-versiones"

echo "== 2. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}
else
  LIBRES=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
  [ "${LIBRES:-0}" -ge 30 ] || { fallo "volumen pequeno: ${LIBRES:-?} GB libres (hacen falta >= 30)"; estado entorno "FALLO disco"; cierre; }
  dnf install -y -q python3.11 python3.11-pip gcc tar gzip unzip || { fallo "dnf install"; estado entorno "FALLO dnf"; cierre; }
  python3.11 -m venv "$BASE/venv" && "$BASE/venv/bin/pip" install -q --no-cache-dir "torch==${TORCH}+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
    && "$BASE/venv/bin/pip" install -q --no-cache-dir "transformers==$TRANSFORMERS" "onnx==$ONNX" "onnxruntime==$ORT_PY" numpy huggingface_hub safetensors || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
  PY=$BASE/venv/bin/python
  # el modelo fp32 (2,4 GB) y su exportacion pasan de la RAM comoda de 8 GB: un swap de 12 GB evita el OOM (como l2qwen3emb_fases.sh)
  fallocate -l 12G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile || echo "AVISO: sin swap"
  estado entorno "OK dnf + pip en $(( $(date +%s) - T0 )) s"
fi
"$PY" -c "import model.alt.w1_filas, model.alt.w1_demos, numpy" || { fallo "imports del codigo"; estado entorno "FALLO imports codigo"; cierre; }
NPT=$N_POR_TAREA; [ -n "${CHISPA_N:-}" ] && NPT=$(( CHISPA_N / 6 ))   # CHISPA_N: solo en pruebas
CN=$(cd "$CH" && "$PY" -m model.alt.w1_filas --dev "$BASE/dev" --salida "$ENT" --n-por-tarea "$NPT")
case "$CN" in "tareas=6 soporte="*" test="*) ;; *) fallo "w1_filas: $CN"; estado entradas "FALLO filas"; cierre;; esac
( cd "$CH" && "$PY" -m model.alt.w1_demos muestra --consultas "$ENT/consultas.jsonl" --por-tarea 10 --salida "$ENT/muestra.jsonl" ) > "$BASE/muestra.out" 2>&1 && [ -s "$ENT/muestra.jsonl" ] || { fallo "muestra de selección"; estado entradas "FALLO muestra"; cierre; }
estado entradas "OK dev (test $N_TEST, soporte $N_SOP) -> $CN (n por tarea $NPT); muestra $(wc -l < "$ENT/muestra.jsonl" | tr -d ' ') filas"

# --- 3. Go y ORT (con sha256), solo la herramienta de bancos; los modulos salen de gomod-proxy/ del tgz
preparar_go() {
  set -u
  if [ "$LOCAL" = 1 ]; then TAGS=""; ORTLIB=""
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
  cd "$CH/server" && go build $TAGS -trimpath -o "$BASE/chispa-banco-qwen" ./cmd/chispa-banco-qwen || return 4
}
( preparar_go > "$BASE/preparar-go.log" 2>&1; echo $? > "$BASE/preparar-go.rc" ) &

# --- 4. Qwen3-Embedding en la revision fijada, tokenizador y ONNX fp32 (sin int8: no viable, L2-qwen3emb)
echo "== 4. modelo de embeddings $(date -u +%T) quedan $(quedan_min) min"
T2=$(date +%s); ONNX_REAL=0
if [ -z "${CHISPA_LOCAL_BANCOS:-}" ]; then
  HF_HUB_ENABLE_HF_TRANSFER=0 "$PY" -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-Embedding-0.6B', revision='$REVE', local_dir='$MODELOE', allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','LICENSE*','README.md'])" \
    || { fallo "descarga del modelo de embeddings"; estado modelo "FALLO descarga"; cierre; }
  grep -qi '^license: apache-2.0' "$MODELOE/README.md" || { fallo "la ficha no declara Apache-2.0"; estado modelo "FALLO licencia"; cierre; }
  [ -f "$MODELOE/model.safetensors" ] && [ -f "$MODELOE/tokenizer.json" ] && [ -f "$MODELOE/config.json" ] || { fallo "faltan ficheros del modelo"; estado modelo "FALLO ficheros"; cierre; }
  "$PY" "$CH/runtime/tok/exportar_bpe.py" "$MODELOE" "$ONX/qwen3emb.chispa-bpe" > "$OUT/bpe.json" 2> "$BASE/bpe.err" || { tail -3 "$BASE/bpe.err"; fallo "exportar tokenizador"; estado tokenizador "FALLO"; cierre; }
  "$PY" "$CH/model/alt/exportar_qwen3emb_onnx.py" --modelo "$MODELOE" --revision "$REVE" --salida "$ONX" --sin-int8 --comprobar $PERMITIR > "$BASE/exportar.out" 2> "$BASE/exportar.err" \
    || { tail -5 "$BASE/exportar.err"; fallo "exportar ONNX"; estado onnx "FALLO"; cierre; }
  cp "$ONX/qwen3emb-onnx.json" "$OUT/onnx.json" 2>/dev/null; ONNX_REAL=1
  estado onnx "OK Qwen3-Embedding@${REVE:0:12} fp32 en $(( $(date +%s) - T2 )) s"
fi
for _ in $(seq 900); do [ -f "$BASE/preparar-go.rc" ] && break; sleep 1; done
[ "$(cat "$BASE/preparar-go.rc" 2>/dev/null || echo 9)" = 0 ] || { tail -15 "$BASE/preparar-go.log"; fallo "chispa-banco-qwen: preparar_go rc=$(cat "$BASE/preparar-go.rc" 2>/dev/null)"; estado go "FALLO"; cierre; }
ORTLIB=$(cat "$BASE/ort-lib"); estado go "OK $(head -1 "$BASE/preparar-go.log"); ORT $ORT_VERSION (so sha256 ${ORT_SO_SHA256:0:12})"

# --- 5. bancos con la herramienta Go (8 de oro por clase) y sus <banco>.ejemplos.jsonl: el oro completo con `fila` y las 8 fijas marcadas con `fs`
echo "== 5. bancos $(date -u +%T) quedan $(quedan_min) min"
T2=$(date +%s)
if [ -n "${CHISPA_LOCAL_BANCOS:-}" ]; then cp -r "$CHISPA_LOCAL_BANCOS/." "$BAN/"
else
  "$BASE/chispa-banco-qwen" -onnx "$ONX/qwen3emb-fp32.onnx" -tok "$ONX/qwen3emb.chispa-bpe" -soporte "$ENT/soporte.jsonl" -salida "$BAN" -k 8 -ort-lib "$ORTLIB" > "$BASE/bancos.log" 2>&1 \
    || { tail -3 "$BASE/bancos.log"; fallo "bancos"; estado bancos "FALLO"; cierre; }
fi
for t in $("$PY" -c "import json;print(' '.join(sorted(json.load(open('$ENT/tareas.json')))))"); do
  [ -f "$BAN/banco-$t.ejemplos.jsonl" ] && [ -f "$BAN/banco-$t.npy" ] || { fallo "banco de $t sin .npy o .ejemplos.jsonl"; estado bancos "FALLO $t"; cierre; }
  grep -q '"fila"' "$BAN/banco-$t.ejemplos.jsonl" || { fallo "banco de $t: .ejemplos.jsonl no está en formato pool (sin fila)"; estado bancos "FALLO pool $t"; cierre; }
  printf '%s\t%s\t%s\n' "$t" "$(wc -l < "$BAN/banco-$t.ejemplos.jsonl" | tr -d ' ')" "$(grep -c '"fs"' "$BAN/banco-$t.ejemplos.jsonl")" >> "$OUT/ejemplos-por-tarea.tsv"
done
cp "$BAN/bancos.json" "$OUT/bancos.json" 2>/dev/null
estado bancos "OK en $(( $(date +%s) - T2 )) s; pool y fijas por tarea: $(awk '{printf "%s=%s/%s ", $1, $2, $3}' "$OUT/ejemplos-por-tarea.tsv")"

# --- 6. vectores de la muestra de selección (60 consultas) con la misma ruta que el servidor
if [ "$ONNX_REAL" = 1 ]; then
  T2=$(date +%s)
  "$BASE/chispa-banco-qwen" -onnx "$ONX/qwen3emb-fp32.onnx" -tok "$ONX/qwen3emb.chispa-bpe" -soporte "$ENT/muestra.jsonl" -salida "$VQ" -k 0 -ort-lib "$ORTLIB" > "$BASE/vq.log" 2>&1 \
    || { tail -3 "$BASE/vq.log"; fallo "vectores de la muestra"; estado vq "FALLO"; cierre; }
  estado vq "OK $(ls "$VQ"/*.npy | wc -l | tr -d ' ') ficheros en $(( $(date +%s) - T2 )) s"
fi

# --- 7. paquete: faseA.tar (sin comprimir: son float32) y faseA.json con el sha256 de todo; se sube primero el tar, luego el json y al final `listo` (lo escribe subir_y_apagar)
echo "== 7. paquete $(date -u +%T) quedan $(quedan_min) min"
T2=$(date +%s)
rm -f "$BASE"/faseA/.sha256; ( cd "$BASE/faseA" && find . -type f ! -name .sha256 | sort | xargs sha256sum | sed 's#  \./#  #' > "$BASE/faseA.sha256" ) && mv "$BASE/faseA.sha256" "$BASE/faseA/.sha256"
( cd "$BASE" && tar -cf faseA.tar -C faseA . ) || { fallo "tar de la fase A"; estado paquete "FALLO"; cierre; }
"$PY" - "$BASE" "$COD_SHA256" "$DEV_SHA256" "$CRIT_SHA" "$REVE" "$ONNX_REAL" "$NPT" <<'PYEOF'
import hashlib, json, sys, time
base, cod, dev, crit, reve, onnx_real, npt = sys.argv[1:8]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
ficheros = {}
for l in open(base + "/faseA/.sha256"):
    s, n = l.rstrip("\n").split("  ", 1)
    ficheros[n] = s
json.dump({"fase": "A", "tar": "faseA.tar", "tar_sha256": sha(base + "/faseA.tar"), "codigo_sha256": cod, "dev_sha256": dev, "criterio_sha256": crit, "onnx_real": onnx_real == "1",
           "qwen3_embedding_rev": reve, "n_por_tarea": int(npt), "ficheros": ficheros}, open(base + "/faseA.json", "w"), indent=1)
PYEOF
[ -f "$BASE/faseA.json" ] || { fallo "faseA.json"; estado paquete "FALLO json"; cierre; }
s3cp "$BASE/faseA.tar" "$CHISPA_S3/faseA.tar" && s3cp "$BASE/faseA.json" "$CHISPA_S3/faseA.json" || { fallo "subida de faseA"; estado paquete "FALLO subida"; cierre; }
cp "$BASE/faseA.json" "$OUT/faseA.json"
estado paquete "OK faseA.tar $(du -m "$BASE/faseA.tar" | cut -f1) MB, sha256 $("$PY" -c "import json;print(json.load(open('$BASE/faseA.json'))['tar_sha256'][:16])") en $(( $(date +%s) - T2 )) s"
cierre
}
