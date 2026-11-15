#!/bin/bash
CHISPA_S3=s3://chispa-pruebas-cuenta-aws/knn-r8b-cc82027
# R8b: la simulación prequential con todo el soporte por logprobs y la regla «umbral» congelada (CRIT_R8B) y la cascada L2→L3 con la
# regla de escalado congelada en el i7 (margen, f = 0,30), en la batería y en las tareas nuevas. c7i.2xlarge BAJO DEMANDA (solo CPU,
# solo numpy; tope duro CHISPA_MIN=25 min ≈ 0,15 USD). user-data (DLAMI, rol rol-ec2-pruebas sin ListBucket), terminate al apagar.
# Usa LOS MISMOS vectores con los que se fijó la regla (i7, transformers 4.46.3, `vec-i7.tgz`): no se recalcula nada.
# Lee:     $CHISPA_S3/{codigo.tgz,bateria.tgz,nuevas.tgz,vec-i7.tgz,fondo.jsonl,fondo2.jsonl,l3_fondo.jsonl,l3_fondo2.jsonl,lp_todo.jsonl,
#                      lp_r8.jsonl,l3_bateria.jsonl,l3_nuevas.jsonl,lp_muestra.jsonl,lp_confirma.jsonl}
# Escribe: $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado (ESPERA_MIN) antes de apagar.
MIN=${CHISPA_MIN:-25}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-8}
E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3; TRANSFORMERS=5.22.2
CRIT8E_SHA=81c88b3141c4d8a5a35858ab66b03f5ea8e59b5e7e9617586a37faa773416ae0
CRIT8B_SHA=2d97fcc6f8e210d7a1be814ebae3eb740bfff91e6d897132b1dd40f4fbcf9a58
N_FONDO=8137; N_FONDO2=18000
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
( while sleep 20; do [ "$(quedan_min)" -le $((4 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f "model.alt."; break; }; done ) &
sin_tiempo() { [ "$(quedan_min)" -le $((4 + ESPERA_MIN + $1)) ]; }

# --- 1. entorno (solo numpy)
echo "== 1. entorno $(date -u +%T)"
if [ "$LOCAL" = 1 ]; then PY=${CHISPA_PY:-python3}
else
  PY=""; for c in /opt/pytorch/bin/python /usr/bin/python3; do [ -x "$c" ] && "$c" -c "import numpy" 2>/dev/null && { PY=$c; break; }; done
  [ -n "$PY" ] || { fallo "sin python con numpy"; estado entorno "FALLO"; cierre; }
fi
estado entorno "OK $("$PY" -c "import numpy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"); $(nproc) vCPU"

# --- 2. código y entradas; criterios ANTES de calcular nada
echo "== 2. entradas $(date -u +%T)"
mkdir -p "$REPO" "$BASE/bat" "$BASE/nuevas" "$BASE/vec"
for f in codigo.tgz bateria.tgz nuevas.tgz vec-i7.tgz fondo.jsonl fondo2.jsonl l3_fondo.jsonl l3_fondo2.jsonl lp_todo.jsonl lp_r8.jsonl \
         l3_bateria.jsonl l3_nuevas.jsonl lp_muestra.jsonl lp_confirma.jsonl; do
  s3cp "$CHISPA_S3/$f" "$BASE/$f" || { fallo "no se pudo bajar $f"; estado entradas "FALLO $f"; cierre; }; done
tar -xzf "$BASE/codigo.tgz" -C "$REPO" && tar -xzf "$BASE/bateria.tgz" -C "$BASE/bat" && tar -xzf "$BASE/nuevas.tgz" -C "$BASE/nuevas" \
  && tar -xzf "$BASE/vec-i7.tgz" -C "$BASE/vec" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
rm -f "$BASE"/bat/._* "$BASE"/nuevas/._* "$BASE"/vec/._*
[ "$(lineas "$BASE/fondo.jsonl")" = "$N_FONDO" ] && [ "$(lineas "$BASE/fondo2.jsonl")" = "$N_FONDO2" ] || { fallo "recuentos del fondo"; estado entradas "FALLO recuentos"; cierre; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
SE=$(cd "$REPO" && "$PY" -m model.alt.escalado criterio | cut -d' ' -f1); S8=$(cd "$REPO" && "$PY" -m model.alt.prequential criterio --cual r8b | cut -d' ' -f1)
[ "$SE" = "$CRIT8E_SHA" ] && [ "$S8" = "$CRIT8B_SHA" ] || { fallo "criterios distintos de los congelados ($SE / $S8)"; estado entradas "FALLO criterio"; cierre; }
TV=$("$PY" -c "import numpy as np; print(np.load('$BASE/vec/fondo.npz')['transformers'])")
[ "$TV" = 4.46.3 ] || { fallo "los vectores no son los de transformers 4.46.3 ($TV)"; estado entradas "FALLO vectores"; cierre; }
estado entradas "OK; criterios ${SE:0:12} ${S8:0:12}; vectores transformers $TV"
VEC=$BASE/vec; FONDOS="$BASE/fondo.jsonl,$BASE/fondo2.jsonl"; LPF="$BASE/l3_fondo.jsonl,$BASE/l3_fondo2.jsonl,$BASE/lp_todo.jsonl,$BASE/lp_r8.jsonl"

# --- 3. R8b: prequential con soporte por logprobs y regla umbral (CRIT_R8B)
echo "== 3. r8b $(date -u +%T) quedan $(quedan_min) min"
T1=$(date +%s)
( cd "$REPO" && "$PY" -m model.alt.prequential medir --bateria "$BASE/bat,$BASE/nuevas" --vec "$VEC" \
    --l3 "$BASE/l3_bateria.jsonl,$BASE/l3_nuevas.jsonl,$BASE/lp_muestra.jsonl,$BASE/lp_confirma.jsonl,$BASE/lp_todo.jsonl" \
    --fondo "$FONDOS" --l3-fondo "$LPF" --vec-fondo "$VEC/fondo.npz" --salida "$OUT/prequential-r8b.json" --semillas 10 --criterio r8b ) \
  && estado r8b "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json,sys; v=json.load(open(sys.argv[1]))['veredicto_r8b']; print(v['estado'], v['dif_nuevas'])" "$OUT/prequential-r8b.json")" \
  || { fallo "r8b prequential"; estado r8b "FALLO"; }

# --- 4. cascada L2→L3 con la regla congelada (margen, f = 0,30), batería y tareas nuevas
echo "== 4. cascada $(date -u +%T) quedan $(quedan_min) min"
T1=$(date +%s)
( cd "$REPO" && "$PY" -m model.alt.escalado medir --bateria "$BASE/bat,$BASE/nuevas" --vec "$VEC" --fondo "$FONDOS" --l3-fondo "$LPF" \
    --l3 "$BASE/lp_muestra.jsonl,$BASE/lp_confirma.jsonl,$BASE/lp_todo.jsonl" --salida "$OUT/cascada-r8b.json" --semillas 10 --regla margen:0.30 ) \
  && estado cascada "OK en $(( $(date +%s) - T1 )) s: $("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['aplicada']['resumen'])" "$OUT/cascada-r8b.json" | cut -c1-200)" \
  || { fallo "cascada"; estado cascada "FALLO"; }
cierre
