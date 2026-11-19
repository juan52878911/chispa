#!/bin/bash
CHISPA_S3=${CHISPA_S3_LOCAL:-s3://chispa-pruebas-879381269751/l2-qwen3emb-5d9e86c}   # CHISPA_S3_LOCAL (con CHISPA_LOCAL=1): directorio en vez de S3, solo para pruebas
# L2-qwen3emb: medida del codificador Qwen3-Embedding-0.6B del servidor Go en una c7i.xlarge (CPU, 4 hilos, 8 GB), NO en GPU. Criterio congelado: model/alt/crit-l2-qwen3emb.json (sha abajo).
# Lanzar con un volumen raíz de >= 40 GB (modelo, ONNX fp32 e int8, venv de torch, Go; el ud lo comprueba) y rol chispa-pruebas-ec2 (sin ListBucket). Tope duro CHISPA_MIN (~0,5 USD).
# Lee por nombre exacto $CHISPA_S3/{codigo.tgz,dev.tgz} (sha256 abajo); escribe $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado. Las fases viven en codigo.tgz
# (model/alt/aws/l2qwen3emb_fases.sh): modelo en la revision fijada -> ONNX fp32/int8 -> referencia de transformers 5.22.2 sobre el fixture -> paridad Go (tokens y coseno) ->
# bancos con la herramienta Go -> servidor con L3 apagado y sin escalado -> e2e HTTP en v32 y v8 x fp32 e int8 -> latencias, RSS y tamanos -> cifras.json.
MIN=${CHISPA_MIN:-150}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-5}
REV=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3; TRANSFORMERS=5.22.2; TORCH=2.15.0; ONNX=1.23.2; ORT_PY=1.31.1
CRIT_SHA=58a62d618a4f917c1c4c232934a31394874a59f4cb5b589e945efd91511ade0b; COD_SHA256=2014f6cf3f8bf90f9e40c51d83be75909d3566e574d5a99801e09cc60710ccc7; DEV_SHA256=3938608459b211ffc2f2c15e792334bd1a1448e28f2d9437a52ce5fa531b21ca; N_TEST=2930; N_SOP=728
GO_VERSION=1.27.1; GO_SHA256=63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445
ORT_VERSION=1.31.1; ORT_NUPKG_SHA256=6871ccbc97f3d5bdb4095a1dcec33aa78d56f6ed8de5980ef5fa1d9f2e7c679a; ORT_SO_SHA256=a96f132e5eb5ca04e23a7d2fc1078fb95aa591e8d71d0f237702ece3a1e0f4ea
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-l2q-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-l2q}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque l2-qwen3emb $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
OUT=$BASE/res; FALLO=""; SRVPID=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
hay_tiempo() { [ "$(quedan_min)" -gt $((ESPERA_MIN + $1)) ]; }
parar_servidor() { [ -n "$SRVPID" ] && { kill "$SRVPID" 2>/dev/null; for _ in $(seq 40); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 0.5; done; kill -KILL "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; }; SRVPID=""; return 0; }
subir_y_apagar() {
  trap - EXIT; parar_servidor; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"
  [ -f "$BASE/codigo/model/alt/qwen3emb_medir.py" ] && [ -d "$BASE/dev" ] && ( cd "$BASE/codigo" && PYTHONPATH=$BASE/codigo "${PY:-python3}" -m model.alt.qwen3emb_medir cifras --res "$OUT" --dev "$BASE/dev" \
      --crit "$BASE/codigo/model/alt/crit-l2-qwen3emb.json" --salida "$OUT/cifras.json" ) > "$OUT/medir.txt" 2>&1; cat "$OUT/medir.txt" 2>/dev/null
  cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  for l in "$BASE"/servidor-*.log; do [ -f "$l" ] && tail -n 100 "$l" > "$OUT/$(basename "$l" .log)-cola.log"; done
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
( while sleep 20; do [ "$(quedan_min)" -le $((ESPERA_MIN + 3)) ] && { echo "== tope: se cierra"; pkill -TERM -f r10_t5_cliente.py; pkill -TERM -f chispa-server; pkill -TERM -f exportar_qwen3emb; pkill -TERM -f qwen3emb_referencia; pkill -TERM -f chispa-banco-qwen; break; }; done ) &
sha_ok() { if command -v sha256sum >/dev/null; then echo "$1  $2" | sha256sum -c --quiet -; else [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" = "$1" ]; fi; }

echo "== 1. entradas $(date -u +%T)"
mkdir -p "$BASE/in" "$BASE/codigo" "$BASE/dev"
for f in codigo dev; do
  sha=$COD_SHA256; [ "$f" = dev ] && sha=$DEV_SHA256
  s3cp "$CHISPA_S3/$f.tgz" "$BASE/in/$f.tgz" && { [ "$LOCAL" = 1 ] || sha_ok "$sha" "$BASE/in/$f.tgz"; } || { fallo "$f.tgz: descarga o sha256"; estado entradas "FALLO $f.tgz"; cierre; }
done
tar -xzf "$BASE/in/codigo.tgz" -C "$BASE/codigo" && tar -xzf "$BASE/in/dev.tgz" -C "$BASE/dev" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
find "$BASE/dev" "$BASE/codigo" -name '._*' -delete
[ -f "$BASE/codigo/model/alt/aws/l2qwen3emb_fases.sh" ] || { fallo "codigo.tgz sin las fases"; estado entradas "FALLO fases"; cierre; }
[ "$(ls "$BASE"/dev/*.jsonl | wc -l | tr -d ' ')" = 6 ] && [ "$(ls "$BASE/dev" | wc -l | tr -d ' ')" = 6 ] || { fallo "dev.tgz no trae solo 6 jsonl"; estado entradas "FALLO dev"; cierre; }
sha_ok "$CRIT_SHA" "$BASE/codigo/model/alt/crit-l2-qwen3emb.json" || { fallo "criterio distinto del congelado"; estado entradas "FALLO criterio"; cierre; }
estado entradas "OK codigo y dev (sha256 comprobados), criterio ${CRIT_SHA:0:12}"
. "$BASE/codigo/model/alt/aws/l2qwen3emb_fases.sh" || { fallo "fases"; cierre; }
fases_main
