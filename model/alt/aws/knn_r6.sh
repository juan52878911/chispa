#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# R6 (y la medida de R4), en una c7i.2xlarge BAJO DEMANDA (solo CPU; tope duro CHISPA_MIN=55 min ≈ 0,33 USD). user-data (DLAMI GPU PyTorch
# AL2023, que arranca sin GPU; rol rol-ec2-pruebas sin ListBucket), --instance-initiated-shutdown-behavior terminate. Todo por nombre exacto.
# Lee:     $CHISPA_S3/{codigo.tgz,bateria.tgz,nuevas.tgz,vec-paso1.tgz,fondo.jsonl,fondo2.jsonl,l3_fondo.jsonl,l3_fondo2.jsonl,
#                      fondo-fuente.jsonl,fondo2-fuente.jsonl,l3_bateria.jsonl,l3_nuevas.jsonl,latencia.tgz}
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
# Pasos, por prioridad:
#  1. vectores de e5-small @ 614241f6 (transformers 5.22.2) de la batería, las tareas nuevas, el fondo y el fondo2;
#  2. R4: `knn_soporte_llm medir --criterio r4` con fondo+fondo2 (CRIT4, sha abajo);
#  3. R6: `prequential medir` (CRIT_R6, sha abajo) en las 6 tareas de la batería y las 2 nuevas;
#  4. vectores de la batería con transformers 4.46.3 y tokenizers 0.20.3 (las versiones del paso 1), con el torch de la AMI y con
#     torch 2.5.1+cpu, frente a los del paso 1: cierra la sospecha de R1 (hatecheck-es difería hasta 0,057);
#  5. latencia de L0-L2 int8 con 1 vCPU y 1,8 GB (`docker run --cpus=1 --memory=1800m`, sustituto de Lambda; nada en Lambda ni IAM).
MIN=${CHISPA_MIN:-55}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-8}
E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3; TRANSFORMERS=5.22.2
CRIT4_SHA=a273e9271013035e5725af0fbec6580d4dcd6c995da3b98465b083fa0c0950ce
CRIT6_SHA=__CRIT6_SHA__
N_FONDO=__N_FONDO__; N_FONDO2=__N_FONDO2__; MOVIL_S3=__MOVIL_S3__   # vacío: la latencia queda en `omitidos`
MOVIL_SHA=995dd6a7276a7026ce12ad3e04a5324714f62dc05a1c20b11141a580c30db6f4   # model/bench/movil/resultados-20261102-210131/manifiesto.json
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-r6-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-r6}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r6 $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min hilos=$HILOS"
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
( while sleep 20; do [ "$(quedan_min)" -le $((4 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f "model.alt."; pkill -TERM -f "docker run"; break; }; done ) &
sin_tiempo() { [ "$(quedan_min)" -le $((4 + ESPERA_MIN + $1)) ]; }

# --- 1. entorno (CPU)
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}
else
  PY=""; for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] || { fallo "ningún Python de la AMI importa torch"; estado entorno "FALLO sin torch"; cierre; }
  "$PY" -m pip install -q --no-cache-dir "transformers==$TRANSFORMERS" huggingface_hub numpy || { fallo "pip install"; estado entorno "FALLO pip"; cierre; }
fi
estado entorno "OK $("$PY" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)" 2>&1 | tail -1); $(nproc) vCPU"

# --- 2. código y entradas; criterios ANTES de calcular nada
echo "== 2. entradas $(date -u +%T)"
mkdir -p "$REPO" "$BASE/bat" "$BASE/nuevas" "$BASE/vec1" "$BASE/lat"
for f in codigo.tgz bateria.tgz nuevas.tgz vec-paso1.tgz fondo.jsonl fondo2.jsonl l3_fondo.jsonl l3_fondo2.jsonl fondo-fuente.jsonl fondo2-fuente.jsonl \
         l3_bateria.jsonl l3_nuevas.jsonl latencia.tgz; do
  s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
tar -xzf "$BASE/codigo.tgz" -C "$REPO" && tar -xzf "$BASE/bateria.tgz" -C "$BASE/bat" && tar -xzf "$BASE/nuevas.tgz" -C "$BASE/nuevas" \
  && tar -xzf "$BASE/vec-paso1.tgz" -C "$BASE/vec1" && tar -xzf "$BASE/latencia.tgz" -C "$BASE/lat" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
rm -f "$BASE"/bat/._* "$BASE"/nuevas/._* "$BASE"/vec1/._*
[ "$(lineas "$BASE/fondo.jsonl")" = "$N_FONDO" ] && [ "$(lineas "$BASE/fondo2.jsonl")" = "$N_FONDO2" ] || { fallo "recuentos del fondo"; estado entradas "FALLO recuentos"; cierre; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
S4=$(cd "$REPO" && "$PY" -m model.alt.knn_soporte_llm criterio --cual r4 | cut -d' ' -f1); S6=$(cd "$REPO" && "$PY" -m model.alt.prequential criterio | cut -d' ' -f1)
[ "$S4" = "$CRIT4_SHA" ] && [ "$S6" = "$CRIT6_SHA" ] || { fallo "criterios distintos de los congelados ($S4 / $S6)"; estado entradas "FALLO criterio"; cierre; }
estado entradas "OK fondo $N_FONDO + $N_FONDO2, l3_fondo2 $(lineas "$BASE/l3_fondo2.jsonl"), l3_nuevas $(lineas "$BASE/l3_nuevas.jsonl"); criterios ${S4:0:12} ${S6:0:12}"

# --- 3. vectores (transformers fijado)
echo "== 3. vectores $(date -u +%T)"
VEC=$BASE/vec; FONDOS="$BASE/fondo.jsonl,$BASE/fondo2.jsonl"
if [ "$LOCAL" = 1 ]; then export CHISPA_E5=${CHISPA_LOCAL_E5:?en local hace falta CHISPA_LOCAL_E5}; VMAX=(--max-fondo "${CHISPA_LOCAL_FONDO:-400}")
else
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('intfloat/multilingual-e5-small', revision='$E5_REV', local_dir='$BASE/e5', allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*','LICENSE*'])" \
    || { fallo "descarga de e5"; estado e5 "FALLO"; cierre; }
  export CHISPA_E5=$BASE/e5; VMAX=()
fi
export HF_HUB_OFFLINE=1
T1=$(date +%s)
( cd "$REPO" && "$PY" -m model.alt.knn_soporte_llm vectores --bateria "$BASE/bat,$BASE/nuevas" --fondo "$FONDOS" --salida "$VEC" --hilos "$HILOS" "${VMAX[@]}" ) \
  || { fallo "vectores"; estado vectores "FALLO"; cierre; }
estado vectores "OK en $(( $(date +%s) - T1 )) s"

# --- 4. R4: kNN con el fondo ampliado
echo "== 4. r4 $(date -u +%T) quedan $(quedan_min) min"
T1=$(date +%s)
( cd "$REPO" && "$PY" -m model.alt.knn_soporte_llm medir --bateria "$BASE/bat" --vec "$VEC" --fondo "$FONDOS" --l3 "$BASE/l3_fondo.jsonl,$BASE/l3_fondo2.jsonl" \
    --fuente "$BASE/fondo-fuente.jsonl,$BASE/fondo2-fuente.jsonl" --vec-oro "$BASE/vec1" --salida "$OUT/knn-r4.json" --semillas 10 --criterio r4 ) \
  && estado r4 "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json,sys; v=json.load(open(sys.argv[1]))['veredicto_r4']; print(v['estado'], v['agrupada_llm'])" "$OUT/knn-r4.json")" \
  || { fallo "r4 medir"; estado r4 "FALLO"; }

# --- 5. R6: simulación prequential
echo "== 5. r6 $(date -u +%T) quedan $(quedan_min) min"
T1=$(date +%s)
( cd "$REPO" && "$PY" -m model.alt.prequential medir --bateria "$BASE/bat,$BASE/nuevas" --vec "$VEC" --l3 "$BASE/l3_bateria.jsonl,$BASE/l3_nuevas.jsonl" \
    --fondo "$FONDOS" --l3-fondo "$BASE/l3_fondo.jsonl,$BASE/l3_fondo2.jsonl" --vec-fondo "$VEC/fondo.npz" --salida "$OUT/prequential.json" --semillas 10 ) \
  && estado r6 "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json,sys; v=json.load(open(sys.argv[1]))['veredicto']; print(v.get('estado'))" "$OUT/prequential.json")" \
  || { fallo "r6 medir"; estado r6 "FALLO"; }

# --- 6. diferencia de vectores frente al paso 1 con las versiones del paso 1
echo "== 6. versiones $(date -u +%T) quedan $(quedan_min) min"
if [ "$LOCAL" = 1 ] || sin_tiempo 8; then estado versiones "OMITIDO"
else
  for var in torch-ami torch251; do
    V=/opt/v-$var; "$PY" -m venv --system-site-packages "$V" && "$V/bin/pip" install -q --no-cache-dir transformers==4.46.3 tokenizers==0.20.3 numpy \
      || { estado versiones "$var FALLO pip"; continue; }
    if [ "$var" = torch251 ]; then "$V/bin/pip" install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 || { estado versiones "$var FALLO torch"; continue; }; fi
    ( cd "$REPO" && "$V/bin/python" -m model.alt.knn_soporte_llm vectores --bateria "$BASE/bat" --fondo "$BASE/fondo.jsonl" --max-fondo 1 --salida "$BASE/vec-$var" --hilos "$HILOS" ) \
      || { estado versiones "$var FALLO vectores"; continue; }
    "$PY" - "$BASE/vec1" "$BASE/vec-$var" "$VEC" "$var" "$("$V/bin/python" -c 'import torch,transformers;print(torch.__version__, transformers.__version__)')" >> "$OUT/versiones.jsonl" <<'EOF'
import json, sys, numpy as np
from pathlib import Path
a, b, c, var, ver = sys.argv[1:6]
out = {"variante": var, "versiones": ver, "max_dif_frente_a_paso1": {}, "max_dif_frente_a_5_22_2": {}}
for f in sorted(Path(a).glob("*.npz")):
    x, y, z = np.load(f), np.load(Path(b) / f.name), np.load(Path(c) / f.name)
    out["max_dif_frente_a_paso1"][f.stem] = round(float(max(np.abs(x["Xte"] - y["Xte"]).max(), np.abs(x["Xtr"] - y["Xtr"]).max())), 6)
    out["max_dif_frente_a_5_22_2"][f.stem] = round(float(max(np.abs(z["Xte"] - y["Xte"]).max(), np.abs(z["Xtr"] - y["Xtr"]).max())), 6)
print(json.dumps(out))
EOF
    estado versiones "$var $(tail -1 "$OUT/versiones.jsonl" | cut -c1-300)"
  done
fi

# --- 7. latencia L0-L2 int8 con 1 vCPU (docker como sustituto de Lambda)
echo "== 7. latencia $(date -u +%T) quedan $(quedan_min) min"
if [ "$LOCAL" = 1 ] || sin_tiempo 6; then estado latencia "OMITIDO"
else
  systemctl start docker 2>/dev/null
  T1=$(date +%s)
  # modelo-int8.chispa (perfil movil-int8) de MOVIL_S3, con el sha256 del manifiesto de model/bench/movil; si falta o no coincide, no se usa
  # y correr.sh deja las capas en `omitidos`
  mkdir -p "$BASE/lat/artefactos"
  if [ -n "$MOVIL_S3" ] && aws s3 cp --only-show-errors "$MOVIL_S3/modelo-int8.chispa" "$BASE/modelo-int8.chispa"; then
    if echo "$MOVIL_SHA  $BASE/modelo-int8.chispa" | sha256sum -c --quiet; then mv "$BASE/modelo-int8.chispa" "$BASE/lat/artefactos/"; estado latencia "modelo int8 OK sha256 ${MOVIL_SHA:0:12}"
    else mv "$BASE/modelo-int8.chispa" "$BASE/modelo-int8.chispa.malo"; estado latencia "modelo int8 con otro sha256: no se usa"; fi
  else estado latencia "sin modelo int8"; fi
  ( cd "$BASE/lat" && CHISPA_IMAGEN=python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b \
      CHISPA_E5_REV=$E5_REV CHISPA_POTION_REV=73908c3438cf03b6a01bcb9611d62b23d0726f08 CHISPA_CPUSET=7 \
      bash correr.sh -d "$BASE/lat/artefactos" -o "$OUT/latencia" -t "$BASE/bat/tuits-es.jsonl" -n 400 -r 5 -m 128 ) && estado latencia "OK en $(( $(date +%s) - T1 )) s" || { fallo "latencia"; estado latencia "FALLO"; }
fi
cierre
