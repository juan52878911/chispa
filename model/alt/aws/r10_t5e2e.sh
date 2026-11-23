#!/bin/bash
CHISPA_S3=__CHISPA_S3__
# R10-t5e2e (g5, tope 30 min). Baja y comprueba entradas por nombre exacto y carga las fases de codigo.tgz (model/alt/aws/r10_t5e2e_fases.sh).
MIN=${CHISPA_MIN:-30}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-3}; VLLM=${CHISPA_VLLM:-0.33.0}
REV=cdbee75f17c01a7cc42f958dc650907174af0554
N_FILAS=__N_FILAS__
E5_SHA256=__E5_SHA256__; COD_SHA256=__COD_SHA256__; BAN_SHA256=__BAN_SHA256__; CON_SHA256=__CON_SHA256__
GO_VERSION=1.27.1
GO_SHA256=__GO_SHA256__
ORT_VERSION=1.31.1
ORT_SHA256=c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9
MOVIL_S3=${CHISPA_MOVIL_S3:-s3://chispa-pruebas-cuenta-aws/movil-f6acb278/movil}
MOVIL_SHA=995dd6a7276a7026ce12ad3e04a5324714f62dc05a1c20b11141a580c30db6f4; MOVIL_BYTES=249716640
TOK_SHA=356d7ef09425121e93a0a595ca5a4591fbebeb8785b0fc5f405a374459cd1ed6; TOK_BYTES=12111120
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-r10t5-s3; mkdir -p "$CHISPA_S3";; esac; MOVIL_S3=${CHISPA_MOVIL_S3:-$CHISPA_S3/movil}
else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-r10t5}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque r10-t5e2e $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
OUT=$BASE/res; FALLO=""; SRVPID=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
hay_tiempo() { [ "$(quedan_min)" -gt $((ESPERA_MIN + $1)) ]; }
parar_servidor() { [ -n "$SRVPID" ] && { kill "$SRVPID" 2>/dev/null; for _ in $(seq 20); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 0.5; done; kill -KILL "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; }; SRVPID=""; return 0; }
subir_y_apagar() {
  trap - EXIT; parar_servidor; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"; cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  for l in "$BASE"/vllm-*.log "$BASE"/servidor-*.log; do [ -f "$l" ] && tail -n 200 "$l" > "$OUT/$(basename "$l" .log)-cola.log"; done
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
( while sleep 20; do [ "$(quedan_min)" -le $((ESPERA_MIN + 2)) ] && { echo "== tope: se cierra"; pkill -TERM -f r10_t5_cliente.py; pkill -TERM -f chispa-server; pkill -TERM -f vllm.entrypoints; break; }; done ) &
sha_ok() { if command -v sha256sum >/dev/null; then echo "$1  $2" | sha256sum -c --quiet -; else [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" = "$1" ]; fi; }

echo "== 1. entradas $(date -u +%T)"
mkdir -p "$BASE/in" "$BASE/codigo" "$BASE/bancos"
bajar() { s3cp "$CHISPA_S3/$1" "$BASE/in/$1" && { [ "$LOCAL" = 1 ] || sha_ok "$2" "$BASE/in/$1"; } || { fallo "$1: descarga o sha256"; estado entradas "FALLO $1"; cierre; }; }
bajar codigo.tgz "$COD_SHA256"; bajar bancos.tgz "$BAN_SHA256"; bajar consultas.jsonl "$CON_SHA256"; bajar e5.onnx "$E5_SHA256"
tar -xzf "$BASE/in/codigo.tgz" -C "$BASE/codigo" && tar -xzf "$BASE/in/bancos.tgz" -C "$BASE/bancos" || { fallo "tgz"; cierre; }
NF=$(wc -l < "$BASE/in/consultas.jsonl" | tr -d ' ')
[ "$NF" = "$N_FILAS" ] && [ -f "$BASE/codigo/model/alt/aws/r10_t5e2e_fases.sh" ] && [ -f "$BASE/bancos/referencia.json" ] || { fallo "entradas: $NF filas (esperadas $N_FILAS) o faltan fases/referencia"; estado entradas "FALLO"; cierre; }
estado entradas "OK $NF consultas, e5.onnx sha256 ${E5_SHA256:0:12}"
. "$BASE/codigo/model/alt/aws/r10_t5e2e_fases.sh" || { fallo "fases"; cierre; }
fases_main
