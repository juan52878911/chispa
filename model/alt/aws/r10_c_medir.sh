#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# R10-c (c7i, CPU): ¿mejora el few-shot real el soporte etiquetado por el LLM? 6 tareas DEV de la batería v2 (nunca las selladas).
# c7i.2xlarge BAJO DEMANDA, tope duro CHISPA_MIN=30 min. user-data (DLAMI, rol chispa-pruebas-ec2 sin ListBucket), terminate al apagar.
# Lee:     $CHISPA_S3/{codigo.tgz,dev.tgz,oro32.jsonl,fondo.jsonl,fondo-fuente.jsonl,lp_reetiqueta.jsonl}
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
#  1. vectores de e5-small @ 614241f6 con transformers 5.22.2 (`knn_soporte_llm vectores`) de oro32, test de dev y fondo etiquetado;
#  2. `few_shot_v2 medir`: kNN y prototipos con oro8, oro32, oro8+LLM, oro32+LLM y LLM solo. El criterio (crit-r10-c.json) se comprueba ANTES.
MIN=${CHISPA_MIN:-30}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-8}
E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3; TRANSFORMERS=5.22.2
CRIT_SHA=__CRIT_SHA__
N_FONDO=__N_FONDO__; N_DEV=6
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-r10c-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-r10c}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r10-c $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min hilos=$HILOS"
OUT=$BASE/res; FALLO=""; REPO=$BASE/repo
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
lineas() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
subir_y_apagar() {
  trap - EXIT; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
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
( while sleep 20; do [ "$(quedan_min)" -le $((4 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f "model.alt."; break; }; done ) &

# --- 1. entorno (CPU)
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] || { fallo "ningún Python de la AMI importa torch"; estado entorno "FALLO sin torch"; cierre; }
  "$PY" -m pip install -q --no-cache-dir "transformers==$TRANSFORMERS" huggingface_hub numpy || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
fi
estado entorno "OK $("$PY" -c "import numpy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"); $(nproc) vCPU"

# --- 2. entradas; el criterio congelado ANTES de calcular nada
echo "== 2. entradas $(date -u +%T)"
mkdir -p "$REPO" "$BASE/dev"
for f in codigo.tgz dev.tgz oro32.jsonl fondo.jsonl fondo-fuente.jsonl lp_reetiqueta.jsonl; do
  s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
tar -xzf "$BASE/codigo.tgz" -C "$REPO" && tar -xzf "$BASE/dev.tgz" -C "$BASE/dev" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
rm -f "$BASE"/dev/._*
[ "$(lineas "$BASE/fondo.jsonl")" = "$N_FONDO" ] || { fallo "recuento del fondo distinto de $N_FONDO"; estado entradas "FALLO recuentos"; cierre; }
[ "$(ls "$BASE"/dev/*.jsonl | wc -l | tr -d ' ')" = "$N_DEV" ] && ! ls "$BASE"/dev | grep -qi sellad || { fallo "dev.tgz no trae exactamente las $N_DEV tareas de desarrollo"; estado entradas "FALLO dev"; cierre; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"; CRIT=$REPO/model/alt/crit-r10-c.json
SC=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$CRIT")
[ "$SC" = "$CRIT_SHA" ] || { fallo "criterio distinto del congelado ($SC)"; estado entradas "FALLO criterio"; cierre; }
estado entradas "OK criterio ${SC:0:12}; fondo $(lineas "$BASE/fondo.jsonl"), oro32 $(lineas "$BASE/oro32.jsonl"), lp $(lineas "$BASE/lp_reetiqueta.jsonl") ($(grep -c '"valida": true' "$BASE/lp_reetiqueta.jsonl") válidas)"
cd "$REPO" && "$PY" -m model.alt.few_shot_v2 armar-dev32 --dev "$BASE/dev" --oro32 "$BASE/oro32.jsonl" --salida "$BASE/dev32" \
  && "$PY" -m model.alt.few_shot_v2 filtrar-fondo --fondo "$BASE/fondo.jsonl" --lp "$BASE/lp_reetiqueta.jsonl" --salida "$BASE/fondo_lp.jsonl" || { fallo "preparar entradas"; estado entradas "FALLO preparar"; cierre; }

# --- 3. vectores (transformers fijado) de oro32, test de dev y fondo etiquetado
echo "== 3. vectores $(date -u +%T) quedan $(quedan_min) min"
VEC=$BASE/vec; T1=$(date +%s)
if [ "$LOCAL" = 1 ]; then
  "$PY" -m model.alt.few_shot_v2 vectores-falsos --bateria "$BASE/dev32" --fondo "$BASE/fondo_lp.jsonl" --salida "$VEC" || { fallo "vectores falsos"; estado vectores "FALLO"; cierre; }; PF=(--permitir-falso)
else
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('intfloat/multilingual-e5-small', revision='$E5_REV', local_dir='$BASE/e5', allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*','LICENSE*'])" \
    || { fallo "descarga de e5"; estado e5 "FALLO"; cierre; }
  export CHISPA_E5=$BASE/e5 HF_HUB_OFFLINE=1; PF=()
  "$PY" -m model.alt.knn_soporte_llm vectores --bateria "$BASE/dev32" --fondo "$BASE/fondo_lp.jsonl" --salida "$VEC" --hilos "$HILOS" --transformers "$TRANSFORMERS" || { fallo "vectores"; estado vectores "FALLO"; cierre; }
fi
estado vectores "OK en $(( $(date +%s) - T1 )) s ($(lineas "$BASE/fondo_lp.jsonl") textos de fondo)"

# --- 4. medición con el criterio congelado
echo "== 4. medir $(date -u +%T) quedan $(quedan_min) min"
T1=$(date +%s)
"$PY" -m model.alt.few_shot_v2 medir --bateria "$BASE/dev32" --vec "$VEC" --fondo "$BASE/fondo_lp.jsonl" --fondo-fuente "$BASE/fondo-fuente.jsonl" --lp "$BASE/lp_reetiqueta.jsonl" \
  --crit "$CRIT" --crit-sha "$CRIT_SHA" --salida "$OUT/medicion.json" "${PF[@]}" \
  && estado medir "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['veredicto'], m['delta']['knn/oro32+llm']['delta_puntos'])" "$OUT/medicion.json")" \
  || { fallo "medir"; estado medir "FALLO"; }
cierre
