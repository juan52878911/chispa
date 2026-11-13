#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# P3: L2x (una pasada por pregunta, e5-small descongelado) en una g5.xlarge spot. user-data (< 16 KB; AMI DLAMI GPU PyTorch AL2023, root,
# rol rol-ec2-pruebas con s3:Get/PutObject y SIN ListBucket), lanzada con --instance-initiated-shutdown-behavior terminate.
# Todo se baja POR NOMBRE EXACTO (nunca `aws s3 cp --recursive`, que pide ListBucket). Claves que LEE (bajo META_S3 = s3://chispa-pruebas-cuenta-aws/meta-20261027/meta):
#   $CHISPA_S3/codigo.tgz · $CHISPA_S3/bateria-no-vista.tgz · $META_S3/{tsjev/train-*,tsjev/dev-*} (nombres de model/meta/resultados/conversion-20261027.json)
#   $META_S3/synzs/{train-00000..N-1,dev-00000}.jsonl.gz · s3://chispa-pruebas-cuenta-aws/blandas/e2deac55256e3314/{blandas.jsonl.gz,destilacion.json}
# Claves que ESCRIBE: $CHISPA_S3/{log.txt,listo,resultados.tgz,ckpt/mejor-l2x.tgz,onnx/l2x_int8.onnx,onnx/exportacion.json}; espera $CHISPA_S3/terminar.
# El código lleva sdk/python (el módulo `chispa`) y PYTHONPATH lo incluye; el paso 2 comprueba los imports ANTES de bajar datos.
# CHISPA_LOCAL=1: sin pip, descargas ni shutdown; CPU, 2 pasos, datos de juguete de CHISPA_LOCAL_DATOS (bateria-no-vista/, jevbench.jsonl, demo.jsonl).
LOCAL=${CHISPA_LOCAL:-0}
MIN=${CHISPA_MIN:-360}                    # techo duro: 6 h (g5.xlarge spot ~0,52 USD/h; bajo demanda 1,006 USD/h)
NSYN=${CHISPA_SYNZS_FICHEROS:-8}          # ficheros de train de synzs (~78 000 filas cada uno)
EPOCAS=${CHISPA_EPOCAS:-3}; TOPE=${CHISPA_TOPE_TAREA:-5000}; LOTE=${CHISPA_LOTE:-32}
META_S3=${CHISPA_META_S3:-s3://chispa-pruebas-cuenta-aws/meta-20261027/meta}
BL_S3=${CHISPA_BLANDAS_S3:-s3://chispa-pruebas-cuenta-aws/blandas/e2deac55256e3314}
ESPERA_MIN=${CHISPA_ESPERA_MIN:-20}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-l2x-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-l2x}
mkdir -p "$BASE"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque l2x $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
export HF_HOME="${HF_HOME:-$HOME/hf}"; mkdir -p "$HF_HOME"
REPO=$BASE/repo DATOS=$BASE/datos OUT=$BASE/resultados CK=$BASE/ck JEV=$BASE/jevbench RAIZ=$BASE/datos/raiz
mkdir -p "$OUT" "$REPO" "$RAIZ/datos"; FALLO=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
subir_y_apagar() {
  trap - EXIT; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  ( cd "$BASE" && tar -czf resultados.tgz resultados ) && s3cp "$BASE/resultados.tgz" "$CHISPA_S3/resultados.tgz" || echo "no se pudo subir resultados.tgz"
  [ -d "$CK/mejor" ] && ( cd "$CK" && tar -czf "$BASE/mejor-l2x.tgz" mejor ) && { s3cp "$BASE/mejor-l2x.tgz" "$CHISPA_S3/ckpt/mejor-l2x.tgz" || echo "no se pudo subir el checkpoint"; }
  [ -f "$BASE/onnx/l2x_int8.onnx" ] && s3cp "$BASE/onnx/l2x_int8.onnx" "$CHISPA_S3/onnx/l2x_int8.onnx" && s3cp "$BASE/onnx/exportacion.json" "$CHISPA_S3/onnx/exportacion.json"
  s3cp "$BASE/log.txt" "$CHISPA_S3/log.txt" || echo "no se pudo subir log.txt"
  if [ "$LOCAL" != 1 ]; then
    echo listo > "$BASE/listo"; s3cp "$BASE/listo" "$CHISPA_S3/listo" || true
    [ -z "$FALLO" ] && for _ in $(seq $((ESPERA_MIN * 2))); do aws s3 cp --only-show-errors "$CHISPA_S3/terminar" "$BASE/terminar" 2>/dev/null && break; sleep 30; done
    shutdown -h now
  fi
}
trap subir_y_apagar EXIT; trap 'exit 143' TERM INT HUP
cierre() { echo "== listo $(date -u +%FT%TZ) fallo=${FALLO:-no} ($(( $(date +%s) - T0 )) s)"; exit 0; }
( while sleep 60; do [ "$(quedan_min)" -le $((12 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f "model.alt.entrenar"; pkill -TERM -f "model.alt.p0_medir"; break; }; done ) &
VIG=$!

# --- 1. entorno
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}; DEV=cpu; DISP=cpu
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python /usr/bin/python3; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] || { fallo "ningún Python de la AMI importa torch"; estado entorno "FALLO sin torch"; cierre; }
  "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "torch no ve la GPU"; estado entorno "FALLO sin GPU"; cierre; }
  "$PY" -m pip install -q --no-cache-dir -U transformers huggingface_hub pyyaml pyarrow numpy onnx onnxruntime || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
  DEV=cuda; DISP=cuda
fi
estado entorno "OK dispositivo $DEV"

# --- 2. código (con sdk/python) e imports ANTES de bajar nada
echo "== 2. código $(date -u +%T)"
s3cp "$CHISPA_S3/codigo.tgz" "$BASE/codigo.tgz" && tar -xzf "$BASE/codigo.tgz" -C "$REPO" || { fallo "codigo.tgz"; estado codigo "FALLO"; cierre; }
export PYTHONPATH="$REPO:$REPO/sdk/python"
( cd "$REPO" && "$PY" -I -c "import sys; sys.path[:0]=['$REPO','$REPO/sdk/python']; import chispa, model.alt.l2x, model.alt.entrenar, model.alt.p0_medir, model.alt.l2x_onnx, model.l2.datos, model.evaluacion.jevbench, model.evaluacion.chispa_eval; print('imports OK')" ) \
  || { fallo "imports del código (¿falta sdk/python?)"; estado codigo "FALLO imports"; cierre; }
estado codigo "OK imports"
cd "$REPO" || cierre

# --- 3. datos
echo "== 3. datos $(date -u +%T)"
BAT=$DATOS/bateria; mkdir -p "$BAT"
if [ "$LOCAL" = 1 ]; then cp "$CHISPA_LOCAL_DATOS"/bateria-no-vista/*.jsonl "$BAT/"; cp "$CHISPA_LOCAL_DATOS/demo.jsonl" "$RAIZ/datos/demo.jsonl"; JEVARG=$CHISPA_LOCAL_DATOS/jevbench.jsonl
  cp "$CHISPA_LOCAL_DATOS/manifiesto-demo.json" "$BASE/manifiesto.json"
  MAN=$BASE/manifiesto.json; EXTRA="--pasos 2 --epocas 1 --lote 4 --max-tok 96 --max-tokens 4096 --tope-tarea 0 --max-dev 8"; estado datos "OK local"
else
  s3cp "$CHISPA_S3/bateria-no-vista.tgz" "$BASE/bateria.tgz" && tar -xzf "$BASE/bateria.tgz" -C "$BAT" || { fallo "bateria-no-vista.tgz"; estado datos "FALLO bateria"; cierre; }
  "$PY" -I - <<'PYEOF' || { fallo "descarga de HF"; estado hf "FALLO"; cierre; }
import sys
from huggingface_hub import snapshot_download as s
bad = []
def t(*a, **k):
    try: s(*a, **k)
    except Exception as e: bad.append(a[0]); print("aviso:", a[0], e)
t("intfloat/multilingual-e5-small", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
for d in ("bgonzalezbustamante/toxicity-protests-ES", "bgonzalezbustamante/toxicity-Constitution-ES", "mrm8488/go_emotions-es-mt", "tanaos/synthetic-spam-detection-dataset-spanish",
          "tasksource/it-support-tickets", "FelipeGuerra/Colombian_Spanish_Cyberbullying_Dataset_1", "FelipeGuerra/Colombian_Spanish_Cyberbullying_Dataset_2"):
    t(d, repo_type="dataset")
print("fallos:", bad or "ninguno"); sys.exit(1 if bad else 0)
PYEOF
  git clone -q https://github.com/fstandhartinger/jevbench "$JEV" || { fallo "git clone JevBench"; estado jevbench "FALLO"; cierre; }
  JEVARG=$JEV/datasets/public
  mkdir -p "$DATOS/tsjev" "$DATOS/synzs"; NF=0
  for f in $("$PY" -I -c 'import json,sys; print("\n".join(f for f in json.load(open(sys.argv[1]))["ficheros"] if f.startswith(("tsjev/train-", "tsjev/dev-"))))' model/meta/resultados/conversion-20261027.json); do
    s3cp "$META_S3/$f" "$DATOS/$f" || { NF=$((NF+1)); echo "no se pudo bajar $f"; }; done
  for i in $(seq 0 $((NSYN-1))); do f=$(printf 'synzs/train-%05d.jsonl.gz' $i); s3cp "$META_S3/$f" "$DATOS/$f" || NF=$((NF+1)); done
  s3cp "$META_S3/synzs/dev-00000.jsonl.gz" "$DATOS/synzs/dev-00000.jsonl.gz" || NF=$((NF+1))
  [ "$NF" = 0 ] || { fallo "$NF ficheros del meta sin bajar"; estado meta "FALLO"; cierre; }
  cat "$DATOS"/synzs/*.jsonl.gz > "$RAIZ/datos/synzs.jsonl.gz"   # gzip multi-miembro
  "$PY" -m model.datos.convertir --salida "$DATOS/conv" --fuentes toxicidad_protestas,toxicidad_constitucion,goemociones,spam_tanaos,soporte_ti,ciberacoso > "$OUT/convertir.log" 2>&1 \
    || { tail -8 "$OUT/convertir.log"; fallo "convertir"; estado datos "FALLO convertir"; cierre; }
  for par in toxicidad_protestas:toxicidad_protestas toxicidad_constitucion:toxicidad_constitucion goemociones:goemociones_es spam_tanaos:spam_tanaos soporte_ti:soporte_ti ciberacoso:ciberacoso; do
    [ -s "$DATOS/conv/${par%%:*}.jsonl" ] || { fallo "sin filas de ${par%%:*}"; estado datos "FALLO"; cierre; }; gzip -n -c "$DATOS/conv/${par%%:*}.jsonl" > "$RAIZ/datos/${par#*:}.jsonl.gz"; done
  "$PY" -m model.alt.manifiesto filtrar --entrada "$DATOS/tsjev" --salida "$RAIZ/datos/tsjev-admitido.jsonl.gz" | tee "$OUT/tsjev-filtrado.txt" || { fallo "filtrar tsjev"; estado tsjev "FALLO"; cierre; }
  MAN=model/alt/manifiesto-l2x.json
  "$PY" -m model.alt.manifiesto validar > /dev/null || { fallo "manifiesto"; estado manifiesto "FALLO"; cierre; }
  s3cp "$BL_S3/blandas.jsonl.gz" "$BASE/blandas.jsonl.gz" && "$PY" -m model.alt.destilar --datos "$RAIZ" --manifiesto "$MAN" --aplicar "$BASE/blandas.jsonl.gz" --resumen "$OUT/blandas.json" > "$OUT/blandas.log" 2>&1 \
    || { tail -3 "$OUT/blandas.log"; fallo "aplicar blandas"; estado blandas "FALLO"; cierre; }
  estado datos "OK $(cat "$OUT/tsjev-filtrado.txt") · blandas $(cat "$OUT/blandas.json")"
  EXTRA="--epocas $EPOCAS --lote $LOTE --max-tok 256 --lr 3e-5 --bf16 --recomputar --tope-tarea $TOPE"
  "$PY" -m model.alt.entrenar --sondeo "$OUT/sondeo.json" --datos "$RAIZ" --manifiesto "$MAN" --arquitectura l2x --salida "$CK/sondeo" --max-tok 256 --max-tokens 16384 --bf16 --recomputar --tope-tarea $TOPE > "$OUT/sondeo.log" 2>&1 \
    && estado sondeo "OK $(cut -c1-160 "$OUT/sondeo.json")" || { tail -4 "$OUT/sondeo.log"; fallo "sondeo de memoria"; estado sondeo "FALLO"; cierre; }
fi

# --- 4. entrenar y 5. evaluar (mismas filas y arnés que P0: p0_medir pred --modelo l2x) y exportar a ONNX
echo "== 4. entrenar $(date -u +%T), quedan $(quedan_min) min"
MINS=$(( $(quedan_min) - 40 - ESPERA_MIN )); [ "$MINS" -ge 5 ] || { fallo "sin tiempo"; cierre; }
"$PY" -m model.alt.entrenar --datos "$RAIZ" --manifiesto "$MAN" --sellar --arquitectura l2x --salida "$CK" --codificador e5-small --semilla 0 --minutos "$MINS" $EXTRA \
  || { fallo "entrenar"; estado entrenar "FALLO"; cierre; }
estado entrenar "OK $(tr -d '\n ' < "$CK/historial.json" | cut -c1-240)"
"$PY" -m model.alt.p0_medir pred --modelo l2x --ckpt "$CK/mejor" --bateria "$BAT" --jevbench "$JEVARG" --salida "$OUT/pred" --dispositivo "$DISP" --hilos 4 > "$OUT/pred.log" 2>&1 \
  && estado evaluar "OK" || { tail -4 "$OUT/pred.log"; fallo "evaluar"; estado evaluar "FALLO"; }
"$PY" -m model.alt.l2x_onnx exportar --ckpt "$CK/mejor" --salida "$BASE/onnx" > "$OUT/onnx.log" 2>&1 \
  && estado onnx "OK $(tr -d '\n ' < "$BASE/onnx/exportacion.json" | cut -c1-200)" || estado onnx "AVISO la exportación falló (se rehace en el PC de pruebas desde el checkpoint)"
kill "$VIG" 2>/dev/null
cierre
