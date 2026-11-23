#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# Lote de experimentos de L2 en una g5.xlarge (OK de Juan 2026-11-04: «Usa aws para agilizar el entrenamiento»).
# user-data (≤ 16 KB; AMI DLAMI GPU PyTorch AL2023, root, rol con s3:Get/PutObject sin ListBucket), lanzada con
# --instance-initiated-shutdown-behavior terminate: el `shutdown -h` termina la instancia. El orquestador sustituye __CHISPA_S3__ y sube
# $CHISPA_S3/codigo.tgz (`git archive HEAD model sdk/python`: `model.l2.aumento` importa `chispa`). M1, tsjev (train y dev) y e5 se bajan aquí; JevBench se clona.
# Orden: M1 sin tocar (k=0) · E1 (largos, solo inferencia) · knn_e5 · variantes por prioridad (puntos 2 y 3, luego 5 y E3) × semillas,
# CHISPA_PASOS cada una. Tras cada ejecución sube su carpeta a $CHISPA_S3/partes/ (lo que no llegue queda sin medir) y antes de cada
# una comprueba que queda tiempo para otra como la más lenta medida. Si falla la primera ejecución entrenada, se cierra.
# CHISPA_LOCAL=1: sin pip, descargas ni shutdown; datos en CHISPA_DATOS (m1.pt, tsjev/, jevbench/), CPU y 2 pasos (prueba de arranque),
# o el lote entero en CPU con CHISPA_PASOS_LOCAL y CHISPA_SEMILLAS_LOCAL (con la caché de e5, ≈ 0,6 s/paso tras llenarla).
LOCAL=${CHISPA_LOCAL:-0}
PASOS=${CHISPA_PASOS:-300}
SEMILLAS=${CHISPA_SEMILLAS:-"0 1 2"}
APAGADO_MIN=${CHISPA_APAGADO_MIN:-150}
VARIANTES=${CHISPA_VARIANTES:-"control coral_vec acumulados grueso_fino noul_dif noul_softmax k1 coseno sin_nce sin_brier ruido"}
M1_S3=${CHISPA_M1_S3:-s3://chispa-pruebas-cuenta-aws/ajuste-4a2e745/ckpt/mejor.pt}
TSJEV_S3=${CHISPA_TSJEV_S3:-s3://chispa-pruebas-cuenta-aws/meta-20261027/meta/tsjev}
[ "$LOCAL" = 1 ] || shutdown -h +"$APAGADO_MIN"
set -uo pipefail
export HOME="${HOME:-/root}"
T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-lote}
mkdir -p "$BASE"
exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque lote $(date -u +%FT%TZ) local=$LOCAL pasos=$PASOS apagado=$APAGADO_MIN min"
REPO=$BASE/repo OUT=$BASE/resultados
mkdir -p "$REPO" "$OUT"
FALLO=""
quedan_min() { echo $(( APAGADO_MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
subir() {  # $1 = carpeta bajo $OUT
  ( cd "$OUT" && tar -czf "$BASE/$1.tgz" "$1" "$1.log" 2>/dev/null ) && s3cp "$BASE/$1.tgz" "$CHISPA_S3/partes/$1.tgz" || echo "no se pudo subir $1"
}
cierre() {
  trap - EXIT
  [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"
  echo "== listo $(date -u +%FT%TZ) fallo=${FALLO:-no} ($(( $(date +%s) - T0 )) s)"
  cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  ( cd "$BASE" && tar -czf resultados.tgz --exclude='*.pt' resultados ) && s3cp "$BASE/resultados.tgz" "$CHISPA_S3/resultados.tgz"
  s3cp "$BASE/log.txt" "$CHISPA_S3/log.txt"
  echo listo > "$BASE/listo"; s3cp "$BASE/listo" "$CHISPA_S3/listo"
  [ "$LOCAL" = 1 ] || shutdown -h now
  exit 0
}
trap cierre EXIT
trap 'exit 143' TERM INT HUP

# --- 1. entorno, código y datos
if [ "$LOCAL" = 1 ]; then
  case $CHISPA_S3 in __*) CHISPA_S3=$BASE/s3;; esac
  PY=${CHISPA_PY:-python3}; DEV=cpu; DATOS=${CHISPA_DATOS:?falta CHISPA_DATOS}
  PASOS=${CHISPA_PASOS_LOCAL:-2}; SEMILLAS=${CHISPA_SEMILLAS_LOCAL:-0}   # por defecto, prueba de arranque; con ellos, el lote en CPU
  AQUI=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$AQUI/../../../.." && pwd)
else
  PY=""
  for c in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /usr/bin/python3; do
    [ -x "$c" ] && "$c" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && { PY=$c; break; }
  done
  [ -n "$PY" ] || { fallo "sin Python con torch y GPU"; estado entorno FALLO; exit 1; }
  "$PY" -m pip install -q --no-cache-dir -U transformers huggingface_hub pyyaml numpy scipy || { fallo "pip"; estado entorno FALLO; exit 1; }
  DEV=cuda; DATOS=$BASE/datos; mkdir -p "$DATOS/tsjev"
  s3cp "$CHISPA_S3/codigo.tgz" "$BASE/codigo.tgz" && tar -xzf "$BASE/codigo.tgz" -C "$REPO" || { fallo "codigo.tgz"; estado codigo FALLO; exit 1; }
  s3cp "$M1_S3" "$DATOS/m1.pt" || { fallo "M1"; estado datos "FALLO M1"; exit 1; }
  for f in dev-00000 $(seq -f 'train-%05g' 0 11); do   # por nombre: el rol no tiene ListBucket y --recursive lo necesita
    s3cp "$TSJEV_S3/$f.jsonl.gz" "$DATOS/tsjev/$f.jsonl.gz" || { fallo "tsjev $f"; estado datos "FALLO tsjev"; exit 1; }
  done
  git clone -q https://github.com/fstandhartinger/jevbench "$DATOS/jevbench" || { fallo "JevBench"; estado datos "FALLO JevBench"; exit 1; }
  HF_HOME=$HOME/hf "$PY" -c 'from huggingface_hub import snapshot_download as s; s("intfloat/multilingual-e5-small", allow_patterns=["*.json","*.safetensors","*.txt","*.model","tokenizer*"])' \
    || { fallo "e5"; estado datos "FALLO e5"; exit 1; }
fi
export HF_HOME=${HF_HOME:-$HOME/hf} PYTHONPATH="$REPO:$REPO/sdk/python${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO" || exit 1
estado entorno "OK $DEV, $(ls "$DATOS/tsjev" | wc -l) ficheros de tsjev, $(cat "$DATOS"/jevbench/datasets/public/*.jsonl | wc -l) casos de JevBench"
J=$DATOS/jevbench CACHE=$BASE/cache

# --- 2. M1 sin tocar, E1 (solo inferencia) y la línea base kNN-e5; si M1 no se evalúa, nada de lo demás se puede comparar
"$PY" -m model.evaluacion.jevbench_l2 --ckpt "$DATOS/m1.pt" --jevbench "$J" --salida "$OUT/m1" --k 0 --device "$DEV" > "$OUT/m1.log" 2>&1 \
  && estado m1 OK || { tail -5 "$OUT/m1.log"; fallo "M1 en JevBench"; estado m1 FALLO; exit 1; }
subir m1
"$PY" -m model.l2.largos --ckpt "$DATOS/m1.pt" --jevbench "$J" --salida "$OUT/e1" --device "$DEV" --con-cabeza128-cola512 > "$OUT/e1.log" 2>&1 \
  && estado e1 OK || { tail -3 "$OUT/e1.log"; estado e1 "FALLO (sigue el lote)"; }
subir e1
"$PY" -m model.l2.experimentos.correr --variante knn_e5 --jevbench "$J" --salida "$OUT/knn_e5" --device "$DEV" > "$OUT/knn_e5.log" 2>&1 \
  && estado knn_e5 OK || { tail -3 "$OUT/knn_e5.log"; estado knn_e5 "FALLO (sigue el lote)"; }
subir knn_e5

# --- 3. variantes × semillas por prioridad (todas con los mismos pasos que el control)
LENTA=0; N=0
for v in $VARIANTES; do
  for s in $SEMILLAS; do
    d="$v-s$s"
    if [ "$LOCAL" != 1 ] && [ "$(quedan_min)" -lt $(( LENTA / 60 + 8 )) ]; then estado "$d" "OMITIDO sin tiempo (quedan $(quedan_min) min)"; continue; fi
    t=$(date +%s)
    "$PY" -m model.l2.experimentos.correr --ckpt "$DATOS/m1.pt" --tsjev "$DATOS/tsjev" --jevbench "$J" --salida "$OUT/$d" \
      --pasos "$PASOS" --semilla "$s" --variante "$v" --device "$DEV" --cache-disco "$CACHE" > "$OUT/$d.log" 2>&1
    rc=$?; dt=$(( $(date +%s) - t )); [ "$dt" -gt "$LENTA" ] && LENTA=$dt
    if [ "$rc" = 0 ]; then estado "$d" "OK $dt s"; N=$((N + 1))
    else tail -5 "$OUT/$d.log"; estado "$d" "FALLO rc=$rc"; [ "$N" = 0 ] && { fallo "primera ejecución $d"; exit 1; }; fi
    subir "$d"
  done
done
estado lote "OK $N ejecuciones"
