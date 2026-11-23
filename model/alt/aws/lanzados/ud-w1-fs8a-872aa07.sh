#!/bin/bash
CHISPA_S3=${CHISPA_S3_LOCAL:-s3://chispa-pruebas-cuenta-aws/w1-fs8a-872aa07}   # CHISPA_S3_LOCAL (con CHISPA_LOCAL=1): directorio en vez de S3, solo para pruebas
# W1c fase A (CPU, no cuenta en la cuota G): prepara lo que la fase B (g5.xlarge, w1_fs8b.sh) no debe calcular con la GPU alquilada: ONNX fp32 de Qwen3-Embedding (rev. fijada), bancos con chispa-banco-qwen
# (+ <banco>.ejemplos.jsonl del pool) y los vectores de las 60 consultas de la muestra de selección de demostraciones. NO hay vectores de las 600 consultas: el servidor no admite vectores precalculados por HTTP (se inyectan
# en proceso con decide.ConVectorConsulta) y no se toca su camino de producción: en B el servidor codifica las consultas en la CPU de la g5.
# UNA c7i.xlarge bajo demanda (4 vCPU, 8 GB), AL2023 x86_64, volumen raíz >= 60 GB, rol rol-ec2-pruebas (sin ListBucket). TOPE DURO CHISPA_MIN=60 min:
# 60/60 x 0,1785 USD/h = 0,18 USD de cómputo (+ ≈ 0,01 de EBS). Tiempo esperado ≈ 40 min: dnf+pip 4, ONNX 5, bancos ≈ 25 (1524 s medidos en 4 vCPU de la g5), muestra 1, subida 3.
# Lee por nombre exacto $CHISPA_S3/{codigo.tgz,dev.tgz} (sha256 abajo); escribe $CHISPA_S3/{faseA.tar,faseA.json,log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado.
# faseA.json trae el sha256 de faseA.tar, de cada fichero que lleva y de codigo.tgz y dev.tgz (B comprueba que son los suyos). Fases: w1_fs8a_fases.sh (en codigo.tgz).
MIN=${CHISPA_MIN:-60}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-4}
REVE=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3; TRANSFORMERS=5.22.2; TORCH=2.15.0; ONNX=1.23.2; ORT_PY=1.31.1
CRIT_SHA=343365bd36b7d31e363f53d02c85222595e2a24e21addcc7b2d64e694ba498e2; COD_SHA256=2e928a9fd64eaba65c543d72cdb55d2fdbe218aa898109fa24842e9264ce3db2; DEV_SHA256=2962ede496cc005e234b7332d3b5bffa725f658ea1d396437ac9d6d2588ee7c7; N_TEST=2930; N_SOP=728; N_POR_TAREA=100
GO_VERSION=1.27.1; GO_SHA256=63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445
ORT_VERSION=1.31.1; ORT_NUPKG_SHA256=6871ccbc97f3d5bdb4095a1dcec33aa78d56f6ed8de5980ef5fa1d9f2e7c679a; ORT_SO_SHA256=a96f132e5eb5ca04e23a7d2fc1078fb95aa591e8d71d0f237702ece3a1e0f4ea
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-w1-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-w1a}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque w1-fs8 fase A $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
OUT=$BASE/res; FALLO=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\t+%ss\n' "$1" "$2" "$(( $(date +%s) - T0 ))" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
hay_tiempo() { [ "$(quedan_min)" -gt $((ESPERA_MIN + $1)) ]; }
subir_y_apagar() {
  trap - EXIT; [ -n "${WPID:-}" ] && kill $WPID 2>/dev/null; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"
  cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  for l in "$BASE"/*.err "$BASE"/*.log; do [ -f "$l" ] && tail -n 60 "$l" > "$OUT/$(basename "$l")-cola.txt"; done
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
( while sleep 20; do [ "$(quedan_min)" -le $((ESPERA_MIN + 3)) ] && { echo "== tope: se cierra"; pkill -TERM -f chispa-banco-qwen; pkill -TERM -f exportar_qwen3emb; break; }; done ) & WPID=$!
sha_ok() { if command -v sha256sum >/dev/null; then echo "$1  $2" | sha256sum -c --quiet -; else [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" = "$1" ]; fi; }

echo "== 1. entradas $(date -u +%T)"
mkdir -p "$BASE/in" "$BASE/codigo" "$BASE/dev"
for f in codigo dev; do
  sha=$COD_SHA256; [ "$f" = dev ] && sha=$DEV_SHA256
  s3cp "$CHISPA_S3/$f.tgz" "$BASE/in/$f.tgz" && sha_ok "$sha" "$BASE/in/$f.tgz" || { fallo "$f.tgz: descarga o sha256"; estado entradas "FALLO $f.tgz"; cierre; }
done
tar -xzf "$BASE/in/codigo.tgz" -C "$BASE/codigo" && tar -xzf "$BASE/in/dev.tgz" -C "$BASE/dev" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
find "$BASE/dev" "$BASE/codigo" -name '._*' -delete
[ -f "$BASE/codigo/model/alt/aws/w1_fs8a_fases.sh" ] || { fallo "codigo.tgz sin las fases"; estado entradas "FALLO fases"; cierre; }
[ "$(ls "$BASE"/dev/*.jsonl | wc -l | tr -d ' ')" = 6 ] && [ "$(ls "$BASE/dev" | wc -l | tr -d ' ')" = 6 ] || { fallo "dev.tgz no trae solo 6 jsonl"; estado entradas "FALLO dev"; cierre; }
sha_ok "$CRIT_SHA" "$BASE/codigo/model/alt/crit-w1-fs8-e2e.json" || { fallo "criterio distinto del congelado"; estado entradas "FALLO criterio"; cierre; }
estado entradas "OK codigo y dev (sha256 comprobados), criterio ${CRIT_SHA:0:12}"
. "$BASE/codigo/model/alt/aws/w1_fs8a_fases.sh" || { fallo "fases"; cierre; }
fases_main
