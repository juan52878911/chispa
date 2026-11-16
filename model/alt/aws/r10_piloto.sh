#!/bin/bash
CHISPA_S3=__CHISPA_S3__
CRIT_SHA=__CRIT_SHA__
# R10, PILOTO de destilación 4B -> biencoder de Chispa (L2). UNA g5.xlarge BAJO DEMANDA (DLAMI GPU PyTorch AL2023), root, rol con s3:Get/PutObject
# por nombre exacto (sin ListBucket). Tope duro CHISPA_MIN=88 min (shutdown -h +88 al arrancar, instancia con terminate): ~1,5 USD.
# Lee: $CHISPA_S3/{codigo.tgz,dev.tgz (las 6 *.jsonl DEV de v2)} · $PARTIDA_S3 (ajuste-80ab8ef/ckpt/mejor.pt) · $META_S3/tsjev/{train,dev}-* (nombres de
#   model/meta/resultados/conversion-20261027.json) · $BL_S3/{blandas.jsonl.gz,destilacion.json} (blandas de Qwen3-4B ya hechas; sin vLLM ni synzs).
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz,ckpt/final.pt}; espera $CHISPA_S3/verificado (<= ESPERA_MIN) antes de apagar.
# Fases: 1 entorno · 2 código, imports y sha del criterio · 3 datos (e5 @614241f6, fuentes HF, tsjev, blandas aplicadas, DEV, partida) · 4 base int8 en CPU
#   (en segundo plano) · 5 entrenar en GPU hasta agotar el tiempo · 6 destilado int8 en CPU y veredicto. Todo en int8 (perfil movil-int8 de model/l2/exportar.py).
# CHISPA_LOCAL=1: sin pip, descargas ni shutdown; datos, e5 y checkpoint sintéticos diminutos, CPU, 12 pasos. CHISPA_LOCAL_S3 = carpeta con codigo.tgz.
LOCAL=${CHISPA_LOCAL:-0}
MIN=${CHISPA_MIN:-88}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-5}; MAX_ENTRENAR=${CHISPA_MIN_ENTRENAR:-50}
S3=s3://chispa-pruebas-879381269751
META_S3=${CHISPA_META_S3:-$S3/meta-20261027/meta}; BL_S3=${CHISPA_BLANDAS_S3:-$S3/blandas/e2deac55256e3314}
PARTIDA_S3=${CHISPA_PARTIDA_S3:-$S3/ajuste-80ab8ef/ckpt/mejor.pt}; E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=${CHISPA_LOCAL_S3:?CHISPA_LOCAL_S3 = carpeta con codigo.tgz};; esac; case $CRIT_SHA in __*) CRIT_SHA=${CHISPA_CRIT_SHA:?};; esac
else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-r10}
mkdir -p "$BASE"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r10-piloto $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
export HF_HOME="${HF_HOME:-$HOME/hf}"; mkdir -p "$HF_HOME"
REPO=$BASE/repo; OUT=$BASE/resultados; CK=$BASE/ck; RAIZ=$BASE/raiz; DEV=$BASE/dev; PART=$BASE/partida.pt
mkdir -p "$OUT" "$REPO" "$RAIZ/datos" "$DEV" "$CK"; FALLO=""; BASEPID=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
subir_y_apagar() {
  trap - EXIT; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  [ -f "$CK/historial.json" ] && cp "$CK/historial.json" "$CK/datos.json" "$OUT/" 2>/dev/null
  ( cd "$BASE" && tar -czf resultados.tgz resultados ) && s3cp "$BASE/resultados.tgz" "$CHISPA_S3/resultados.tgz" || echo "no se pudo subir resultados.tgz"
  [ -f "$CK/final.pt" ] && { s3cp "$CK/final.pt" "$CHISPA_S3/ckpt/final.pt" || echo "no se pudo subir final.pt"; }
  s3cp "$BASE/log.txt" "$CHISPA_S3/log.txt" || echo "no se pudo subir log.txt"
  if [ "$LOCAL" != 1 ]; then
    echo listo > "$BASE/listo"; s3cp "$BASE/listo" "$CHISPA_S3/listo" || true
    for _ in $(seq $((ESPERA_MIN * 2))); do aws s3 cp --only-show-errors "$CHISPA_S3/verificado" "$BASE/verificado" 2>/dev/null && break; sleep 30; done
    shutdown -h now
  fi
}
trap subir_y_apagar EXIT; trap 'exit 143' TERM INT HUP
cierre() { echo "== listo $(date -u +%FT%TZ) fallo=${FALLO:-no} ($(( $(date +%s) - T0 )) s)"; exit 0; }
( while sleep 60; do [ "$(quedan_min)" -le $((10 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f "destilar_biencoder"; pkill -TERM -f "eval_r10_piloto"; break; }; done ) &
VIG=$!

# --- 1. entorno
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python /usr/bin/python3; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] || { fallo "ningún Python de la AMI importa torch"; estado entorno "FALLO sin torch"; cierre; }
  "$PY" -c "import torch; assert torch.cuda.is_available()" || { fallo "torch no ve la GPU"; estado entorno "FALLO sin GPU"; cierre; }
  "$PY" -m pip install -q --no-cache-dir -U transformers==5.22.2 huggingface_hub pyyaml pyarrow numpy onnx onnxruntime || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
fi
estado entorno "OK $("$PY" -c "import torch, transformers, onnxruntime; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'ort', onnxruntime.__version__)" 2>/dev/null)"

# --- 2. código, imports y sha del criterio ANTES de bajar datos
echo "== 2. código $(date -u +%T)"
s3cp "$CHISPA_S3/codigo.tgz" "$BASE/codigo.tgz" && tar -xzf "$BASE/codigo.tgz" -C "$REPO" || { fallo "codigo.tgz"; estado codigo "FALLO"; cierre; }
find "$REPO" -name '._*' -delete; cd "$REPO" || cierre
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "import model.alt.destilar_biencoder, model.alt.eval_r10_piloto, model.alt.destilar, model.l2.exportar, model.l2.evaluar, model.evaluacion.chispa_eval, model.datos.convertir; print('imports OK')" \
  || { fallo "imports del código"; estado codigo "FALLO imports"; cierre; }
[ "$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" model/alt/crit-r10-piloto.json)" = "$CRIT_SHA" ] \
  || { fallo "sha256 del criterio distinto de $CRIT_SHA"; estado criterio "FALLO"; cierre; }
estado codigo "OK imports; criterio sha256 $CRIT_SHA"

# --- 3. datos
echo "== 3. datos $(date -u +%T)"
VOC=(); TRAIN_EXTRA=()
if [ "$LOCAL" = 1 ]; then
  "$PY" -m model.alt.destilar_biencoder sintetico --salida "$BASE/sint" > "$OUT/sintetico.log" 2>&1 || { fallo "sintético"; estado datos "FALLO"; cierre; }
  export CHISPA_E5=$BASE/sint/e5; export HF_HUB_OFFLINE=1
  cp -R "$BASE/sint/raiz/." "$RAIZ/"; cp "$BASE/sint/dev/"*.jsonl "$DEV/"; cp "$BASE/sint/partida.pt" "$PART"; cp "$BASE/sint/blandas.jsonl.gz" "$BASE/blandas.jsonl.gz"
  VOC=(--vocab "$BASE/sint/vocab-diminuto.json")
  TRAIN_EXTRA=(--pasos 12 --lote 8 --tope-tarea 0 --dev-max 20 --vias 4 --umbral-dup 0.9999 --log-cada 4 --device cpu); MINS=0
else
  export CHISPA_E5=$("$PY" -I - <<PYEOF 2>"$OUT/hf-e5.log" | tail -1
from huggingface_hub import snapshot_download
print(snapshot_download("intfloat/multilingual-e5-small", revision="$E5_REV", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"]))
PYEOF
)
  [ -f "$CHISPA_E5/config.json" ] || { tail -3 "$OUT/hf-e5.log"; fallo "descarga de e5"; estado hf "FALLO e5"; cierre; }
  "$PY" -I - <<'PYEOF' || { fallo "descarga de HF"; estado hf "FALLO"; cierre; }
import sys
from huggingface_hub import snapshot_download as s
bad = []
for d in ("bgonzalezbustamante/toxicity-protests-ES", "bgonzalezbustamante/toxicity-Constitution-ES", "mrm8488/go_emotions-es-mt", "tanaos/synthetic-spam-detection-dataset-spanish",
          "tasksource/it-support-tickets", "FelipeGuerra/Colombian_Spanish_Cyberbullying_Dataset_1", "FelipeGuerra/Colombian_Spanish_Cyberbullying_Dataset_2"):
    try: s(d, repo_type="dataset")
    except Exception as e: bad.append(d); print("aviso:", d, e)
print("fallos:", bad or "ninguno"); sys.exit(1 if bad else 0)
PYEOF
  mkdir -p "$BASE/tsjev"; NF=0
  for f in $("$PY" -I -c 'import json,sys; print("\n".join(f for f in json.load(open(sys.argv[1]))["ficheros"] if f.startswith(("tsjev/train-", "tsjev/dev-"))))' model/meta/resultados/conversion-20261027.json); do
    s3cp "$META_S3/$f" "$BASE/$f" || { NF=$((NF+1)); echo "no se pudo bajar $f"; }; done
  [ "$NF" = 0 ] || { fallo "$NF ficheros de tsjev sin bajar"; estado tsjev "FALLO"; cierre; }
  "$PY" -m model.datos.convertir --salida "$BASE/conv" --fuentes toxicidad_protestas,toxicidad_constitucion,goemociones,spam_tanaos,soporte_ti,ciberacoso > "$OUT/convertir.log" 2>&1 \
    || { tail -8 "$OUT/convertir.log"; fallo "convertir"; estado datos "FALLO convertir"; cierre; }
  for par in toxicidad_protestas:toxicidad_protestas toxicidad_constitucion:toxicidad_constitucion goemociones:goemociones_es spam_tanaos:spam_tanaos soporte_ti:soporte_ti ciberacoso:ciberacoso; do
    [ -s "$BASE/conv/${par%%:*}.jsonl" ] || { fallo "sin filas de ${par%%:*}"; estado datos "FALLO"; cierre; }; gzip -n -c "$BASE/conv/${par%%:*}.jsonl" > "$RAIZ/datos/${par#*:}.jsonl.gz"; done
  "$PY" -m model.alt.manifiesto filtrar --entrada "$BASE/tsjev" --salida "$RAIZ/datos/tsjev-admitido.jsonl.gz" | tee "$OUT/tsjev-filtrado.txt" || { fallo "filtrar tsjev"; estado tsjev "FALLO"; cierre; }
  s3cp "$BL_S3/blandas.jsonl.gz" "$BASE/blandas.jsonl.gz" && s3cp "$BL_S3/destilacion.json" "$OUT/destilacion-origen.json" || { fallo "blandas de S3"; estado blandas "FALLO"; cierre; }
  s3cp "$CHISPA_S3/dev.tgz" "$BASE/dev.tgz" && mkdir -p "$BASE/dev-raw" && tar -xzf "$BASE/dev.tgz" -C "$BASE/dev-raw" || { fallo "dev.tgz"; estado dev "FALLO"; cierre; }
  find "$BASE/dev-raw" -name '*.jsonl' -not -name '._*' -exec cp {} "$DEV/" \;
  s3cp "$PARTIDA_S3" "$PART" || { fallo "checkpoint de partida"; estado partida "FALLO"; cierre; }
  export HF_HUB_OFFLINE=1; MINS=$(( $(quedan_min) - 35 )); [ "$MINS" -gt "$MAX_ENTRENAR" ] && MINS=$MAX_ENTRENAR
  [ "$MINS" -ge 10 ] || { fallo "sin tiempo para entrenar ($MINS min)"; estado entrenar "FALLO sin tiempo"; cierre; }
  TRAIN_EXTRA=(--minutos "$MINS" --epocas 2 --lote 32)
fi
"$PY" -m model.alt.manifiesto validar > /dev/null || { fallo "manifiesto"; estado manifiesto "FALLO"; cierre; }
MAN=model/alt/manifiesto-entrenamiento.json
"$PY" -m model.alt.destilar --datos "$RAIZ" --manifiesto "$MAN" --aplicar "$BASE/blandas.jsonl.gz" --resumen "$OUT/blandas.json" > "$OUT/blandas.log" 2>&1 \
  || { tail -3 "$OUT/blandas.log"; fallo "aplicar blandas"; estado blandas "FALLO"; cierre; }
NDEV=$(ls "$DEV"/*.jsonl 2>/dev/null | wc -l | tr -d " "); [ "$NDEV" = 6 ] || { fallo "dev: $NDEV ficheros y deben ser 6"; estado dev "FALLO"; cierre; }
estado datos "OK blandas $(cat "$OUT/blandas.json") · dev $NDEV tareas · quedan $(quedan_min) min"

# --- 4. base en int8, en CPU y en segundo plano mientras la GPU entrena
echo "== 4. base int8 $(date -u +%T)"
EV=(-m model.alt.eval_r10_piloto medir --dev "$DEV" --crit model/alt/crit-r10-piloto.json --sha-crit "$CRIT_SHA" ${VOC[@]+"${VOC[@]}"})
nice "$PY" "${EV[@]}" --ckpt "$PART" --etiqueta base --salida "$OUT/res-base.json" --hilos 2 > "$OUT/eval-base.log" 2>&1 &
BASEPID=$!

# --- 5. entrenar (destilar_biencoder: KL(T=2)+CE oro+contrastivo, 20 % de episodios con prototipos de 8, penalidad a e5 original)
echo "== 5. entrenar $(date -u +%T), quedan $(quedan_min) min"
"$PY" -m model.alt.destilar_biencoder entrenar --datos "$RAIZ" --manifiesto "$MAN" --partida "$PART" --salida "$CK" --dev-dir "$DEV" "${TRAIN_EXTRA[@]}" \
  > "$OUT/entrenar.log" 2>&1; RC=$?
tail -4 "$OUT/entrenar.log"; { [ "$RC" = 0 ] || [ "$RC" = 143 ]; } && [ -f "$CK/final.pt" ] || { fallo "entrenar (rc $RC)"; estado entrenar "FALLO"; cierre; }
estado entrenar "OK rc=$RC (143 = tope de tiempo) $(grep -c '^paso' "$OUT/entrenar.log") líneas de paso · $(grep '^datos:' "$OUT/entrenar.log" | cut -c1-300)"

# --- 6. destilado en int8 y veredicto
echo "== 6. destilado int8 $(date -u +%T), quedan $(quedan_min) min"
wait "$BASEPID" || { tail -4 "$OUT/eval-base.log"; fallo "medir base"; estado base "FALLO"; cierre; }
estado base "OK $(grep -h macro-F1 "$OUT/eval-base.log" | tr '\n' ' ')"
"$PY" "${EV[@]}" --ckpt "$CK/final.pt" --etiqueta destilado --salida "$OUT/res-destilado.json" --hilos 4 > "$OUT/eval-destilado.log" 2>&1 \
  || { tail -4 "$OUT/eval-destilado.log"; fallo "medir destilado"; estado destilado "FALLO"; cierre; }
estado destilado "OK $(grep -h macro-F1 "$OUT/eval-destilado.log" | tr '\n' ' ')"
"$PY" -m model.alt.eval_r10_piloto comparar --base "$OUT/res-base.json" --destilado "$OUT/res-destilado.json" --crit model/alt/crit-r10-piloto.json --sha-crit "$CRIT_SHA" \
  --salida "$OUT/informe" > "$OUT/comparar.log" 2>&1 || { tail -4 "$OUT/comparar.log"; fallo "comparar"; estado veredicto "FALLO"; cierre; }
estado veredicto "$("$PY" -c "import json; r=json.load(open('$OUT/informe/r10-piloto.json')); print(r['veredicto'], 'dzs=%.2f dk8=%.2f' % (r['seis']['zeroshot']['delta'], r['seis']['k8']['delta']), 'cambia_sin_sinteticas=%s' % r['veredicto_cambia_sin_sinteticas'])")"
kill "$VIG" 2>/dev/null
cierre
