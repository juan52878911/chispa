#!/bin/bash
CHISPA_S3=${CHISPA_S3_LOCAL:-s3://chispa-pruebas-879381269751/w1-fs8b-ad2b70e}   # CHISPA_S3_LOCAL (con CHISPA_LOCAL=1): directorio en vez de S3, solo para pruebas
S3A=${CHISPA_S3A_LOCAL:-s3://chispa-pruebas-879381269751/w1-fs8a-ad2b70e}      # prefijo donde la fase A deja faseA.tar y faseA.json (solo se leen)
# W1c fase B: e2e fs8 POR EL SERVIDOR Go (modo gpu: todo a L3, vLLM real con Qwen3-4B-Instruct-2507, banco de Qwen3-Embedding con el pool <banco>.ejemplos.jsonl: las 8 demostraciones MÁS CERCANAS por consulta) frente al fs8-kNN
# de Python (llm_logprobs.py --ejemplos --demos, mismas filas y mismas demostraciones que registra el servidor) en las 6 tareas de DESARROLLO. Criterio: model/alt/crit-w1-fs8-e2e.json (sha abajo, congelado).
# El ONNX, los bancos y los vectores de la muestra los trae la fase A (w1_fs8a.sh): por nombre exacto $S3A/{faseA.tar,faseA.json}; se comprueba el sha256 del tar, el de cada fichero y que A usó este mismo codigo.tgz/dev.tgz/criterio.
# UNA g5.xlarge bajo demanda (A10G, 4 vCPU, 16 GB), DLAMI GPU PyTorch AL2023, volumen raíz >= 100 GB, rol chispa-pruebas-ec2 (sin ListBucket). TOPE DURO CHISPA_MIN=75 min:
# 75/60 x 1,006 USD/h = 1,26 USD de cómputo (+ ≈ 0,02 de EBS); con la fase A (60 min x 0,1785 = 0,18 + 0,01) el total es <= 1,47 USD <= 1,6.
# Se llega a 75 y no a 55 porque no se puede justificar sin red que el vLLM fijado reutilice el torch de la DLAMI: el arranque (uv, torch del venv de la DLAMI si la versión encaja) se mide en estado.tsv.
# Calculo del tiempo: entradas+faseA 4 min, vLLM (uv + 4B + arranque, en paralelo con Go y la comprobación) <= 15, autoprueba 2, e2e 600 filas (codifica en la CPU de la g5: 600 x ≈ 0,5 s ≈ 5 min, más L3) ≈ 20, fs8 de Python ≈ 15.
# Si el e2e no cabe: no se empieza otra tarea cuando quedan <= RESERVA_PY (20) min y el fs8 de Python corre sobre las tareas ya servidas (cifras-parcial.json).
# Lee por nombre exacto $CHISPA_S3/{codigo.tgz,dev.tgz} (sha256 abajo); escribe $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado. Fases: w1_fs8b_fases.sh (en codigo.tgz).
MIN=${CHISPA_MIN:-75}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-4}; RESERVA_PY=${CHISPA_RESERVA_PY:-20}; VLLM=${CHISPA_VLLM:-0.33.0}
REV4=cdbee75f17c01a7cc42f958dc650907174af0554
CRIT_SHA=343365bd36b7d31e363f53d02c85222595e2a24e21addcc7b2d64e694ba498e2; COD_SHA256=2e928a9fd64eaba65c543d72cdb55d2fdbe218aa898109fa24842e9264ce3db2; DEV_SHA256=2962ede496cc005e234b7332d3b5bffa725f658ea1d396437ac9d6d2588ee7c7; FASEA_SHA256=-; N_TEST=2930; N_SOP=728; N_POR_TAREA=100
GO_VERSION=1.27.1; GO_SHA256=63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445
ORT_VERSION=1.31.1; ORT_NUPKG_SHA256=6871ccbc97f3d5bdb4095a1dcec33aa78d56f6ed8de5980ef5fa1d9f2e7c679a; ORT_SO_SHA256=a96f132e5eb5ca04e23a7d2fc1078fb95aa591e8d71d0f237702ece3a1e0f4ea
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-w1-s3b; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-w1b}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque w1-fs8 fase B $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
OUT=$BASE/res; FALLO=""; SRVPID=""; VP=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\t+%ss\n' "$1" "$2" "$(( $(date +%s) - T0 ))" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
hay_tiempo() { [ "$(quedan_min)" -gt $((ESPERA_MIN + $1)) ]; }
parar_servidor() { [ -n "$SRVPID" ] && { kill "$SRVPID" 2>/dev/null; for _ in $(seq 40); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 0.5; done; kill -KILL "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; }; SRVPID=""; return 0; }
medir() { ( cd "$BASE/codigo" && PYTHONPATH=$BASE/codigo "${PY:-python3}" -m model.alt.w1_medir --filas "$BASE/faseA/ent/consultas.jsonl" --go "$OUT/e2e-w1-filas.jsonl" --py "$OUT/py" \
      --crit "$BASE/codigo/model/alt/crit-w1-fs8-e2e.json" --seleccion "$OUT/seleccion.json" "$@" ); }
subir_y_apagar() {
  trap - EXIT; parar_servidor; [ -n "${WPID:-}" ] && kill $WPID 2>/dev/null; [ -n "$VP" ] && kill "$VP" 2>/dev/null; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"
  if [ -f "$BASE/codigo/model/alt/w1_medir.py" ] && [ -f "$OUT/e2e-w1-filas.jsonl" ] && [ -d "$OUT/py" ]; then
    { medir --salida "$OUT/cifras.json"; medir --parcial --salida "$OUT/cifras-parcial.json"; } > "$OUT/medir.txt" 2>&1; cat "$OUT/medir.txt"
  fi
  cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  for l in "$BASE"/servidor-*.log "$BASE"/vllm-*.log; do [ -f "$l" ] && tail -n 100 "$l" > "$OUT/$(basename "$l" .log)-cola.log"; done
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
( while sleep 20; do [ "$(quedan_min)" -le $((ESPERA_MIN + 3)) ] && { echo "== tope: se cierra"; pkill -TERM -f r10_t5_cliente.py; pkill -TERM -f llm_logprobs; pkill -TERM -f chispa-server; pkill -TERM -f vllm.entrypoints; break; }; done ) & WPID=$!
sha_ok() { if command -v sha256sum >/dev/null; then echo "$1  $2" | sha256sum -c --quiet -; else [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" = "$1" ]; fi; }

echo "== 1. entradas $(date -u +%T)"
mkdir -p "$BASE/in" "$BASE/codigo" "$BASE/dev"
for f in codigo dev; do
  sha=$COD_SHA256; [ "$f" = dev ] && sha=$DEV_SHA256
  s3cp "$CHISPA_S3/$f.tgz" "$BASE/in/$f.tgz" && sha_ok "$sha" "$BASE/in/$f.tgz" || { fallo "$f.tgz: descarga o sha256"; estado entradas "FALLO $f.tgz"; cierre; }
done
tar -xzf "$BASE/in/codigo.tgz" -C "$BASE/codigo" && tar -xzf "$BASE/in/dev.tgz" -C "$BASE/dev" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
find "$BASE/dev" "$BASE/codigo" -name '._*' -delete
[ -f "$BASE/codigo/model/alt/aws/w1_fs8b_fases.sh" ] || { fallo "codigo.tgz sin las fases"; estado entradas "FALLO fases"; cierre; }
[ "$(ls "$BASE"/dev/*.jsonl | wc -l | tr -d ' ')" = 6 ] && [ "$(ls "$BASE/dev" | wc -l | tr -d ' ')" = 6 ] || { fallo "dev.tgz no trae solo 6 jsonl"; estado entradas "FALLO dev"; cierre; }
sha_ok "$CRIT_SHA" "$BASE/codigo/model/alt/crit-w1-fs8-e2e.json" || { fallo "criterio distinto del congelado"; estado entradas "FALLO criterio"; cierre; }
estado entradas "OK codigo y dev (sha256 comprobados), criterio ${CRIT_SHA:0:12}"
. "$BASE/codigo/model/alt/aws/w1_fs8b_fases.sh" || { fallo "fases"; cierre; }
fases_main
