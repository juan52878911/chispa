#!/bin/bash
CHISPA_S3=${CHISPA_S3_LOCAL:-s3://chispa-pruebas-cuenta-aws/w1-fs8b2-182d52ca75079c224340387aba77b6ec0b0ff41e}   # CHISPA_S3_LOCAL (con CHISPA_LOCAL=1): directorio en vez de S3, solo para pruebas
S3A=${CHISPA_S3A_LOCAL:-s3://chispa-pruebas-cuenta-aws/w1-fs8a-872aa07}      # donde la fase A dejó faseA.tar y faseA.json (solo se leen)
S3B=${CHISPA_S3B_LOCAL:-s3://chispa-pruebas-cuenta-aws/w1-fs8b-872aa07}      # donde la fase B dejó resultados.tgz (solo se lee)
# W1d fase B2: termina la medida fs8 de la fase B (W1c) que cortó el reloj. NO levanta el servidor Go ni re-mide Go (600/600 filas con 200 y L3 en B, selección de demos 60/60 igual): solo el fs8-kNN de PYTHON
# sobre las filas que B dejó sin respuesta válida (horeca, jql, usajobs y las ids de finanzas que faltan), con las MISMAS ids y las MISMAS demostraciones que registró el servidor (demos-log.jsonl, demos/, test-demos/),
# el mismo modelo y revisión y la misma configuración que B (llm_logprobs.py idéntico: lo comprueba el empaquetador). Junta las filas nuevas con las válidas de B (una fila por id; la nueva solo sustituye a una no
# válida o ausente) y corre w1_medir sobre las 600 de Go y las 600 de Python con la selección de B. Criterio congelado: model/alt/crit-w1-fs8-e2e.json (no cambia). No es una iteración nueva.
# UNA g5.xlarge bajo demanda (A10G), DLAMI GPU PyTorch AL2023, volumen raíz >= 100 GB, rol rol-ec2-pruebas. TOPE DURO CHISPA_MIN=45 min: 45/60 x 1,006 USD/h = 0,75 USD (+ ≈ 0,01 de EBS).
# Tiempo: entradas 2 min, vLLM (uv + 4B + arranque, en paralelo con el entorno y el reparto) <= 15, finanzas 12 filas (B: 14 s/fila con 4 hilos) ≈ 3, score 300 filas en lo que quede (≈ 20 min). Orden: finanzas y luego
# las demás; no se empieza tarea con <= ESPERA+2 min; si se corta deja cifras-parcial.json (con las filas hechas) y las mezcla al cerrar. Escribe $CHISPA_S3/{log.txt,listo,resultados.tgz}; espera $CHISPA_S3/verificado.
# Lee por nombre exacto $CHISPA_S3/codigo.tgz, $S3B/resultados.tgz y $S3A/{faseA.tar,faseA.json} (sha256 abajo). Fases: w1_fs8b2_fases.sh (en codigo.tgz).
MIN=${CHISPA_MIN:-45}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-4}; VLLM=${CHISPA_VLLM:-0.33.0}
REV4=cdbee75f17c01a7cc42f958dc650907174af0554
CRIT_SHA=343365bd36b7d31e363f53d02c85222595e2a24e21addcc7b2d64e694ba498e2; COD_SHA256=9356267bcb445b8a0de6eb45165307da9a10af5126719e8bcfa23717d907a1ba; RESB_SHA256=b95e87a310e733029feb4925c61377292fa73509247bbe302cdce759f666a6d9; FASEA_SHA256=c0e7d2275bf9cb9841cb0d95ef85bcb81dd188e817601fddb96ad9ee7d5c3ad6
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-w1-s3b2; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-w1b2}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque w1-fs8 fase B2 $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
OUT=$BASE/res; FALLO=""; VP=""; PY=${CHISPA_PY:-python3}
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\t+%ss\n' "$1" "$2" "$(( $(date +%s) - T0 ))" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
hay_tiempo() { [ "$(quedan_min)" -gt $((ESPERA_MIN + $1)) ]; }
juntar() { ( cd "$BASE/codigo" && PYTHONPATH=$BASE/codigo "$PY" -m model.alt.w1_juntar juntar --res "$BASE/B/res" --nuevo "$OUT/py-nuevo" --salida "$OUT/py" ); }
medir() { ( cd "$BASE/codigo" && PYTHONPATH=$BASE/codigo "$PY" -m model.alt.w1_medir --filas "$BASE/ent/consultas.jsonl" --go "$BASE/B/res/e2e-w1-filas.jsonl" --py "$OUT/py" \
      --crit "$BASE/codigo/model/alt/crit-w1-fs8-e2e.json" --seleccion "$BASE/B/res/seleccion.json" "$@" ); }
subir_y_apagar() {
  trap - EXIT; [ -n "${WPID:-}" ] && kill $WPID 2>/dev/null; [ -n "$VP" ] && kill "$VP" 2>/dev/null; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"
  if [ -f "$BASE/B/res/e2e-w1-filas.jsonl" ] && [ -f "$BASE/codigo/model/alt/w1_medir.py" ]; then
    { mkdir -p "$OUT/py-nuevo" "$OUT/py"; juntar && cp "$BASE/B/res/seleccion.json" "$BASE/B/res/e2e-w1-filas.jsonl" "$OUT/" && medir --salida "$OUT/cifras.json" && medir --parcial --salida "$OUT/cifras-parcial.json"; } > "$OUT/medir.txt" 2>&1; cat "$OUT/medir.txt"
  fi
  cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  for l in "$BASE"/vllm-*.log; do [ -f "$l" ] && tail -n 100 "$l" > "$OUT/$(basename "$l" .log)-cola.log"; done
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
( while sleep 20; do [ "$(quedan_min)" -le $((ESPERA_MIN + 3)) ] && { echo "== tope: se cierra"; pkill -TERM -f llm_logprobs; pkill -TERM -f vllm.entrypoints; break; }; done ) & WPID=$!
sha_ok() { if command -v sha256sum >/dev/null; then echo "$1  $2" | sha256sum -c --quiet -; else [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" = "$1" ]; fi; }

echo "== 1. entradas $(date -u +%T)"
mkdir -p "$BASE/in" "$BASE/codigo" "$BASE/B" "$OUT/py-nuevo"
for par in "codigo.tgz $CHISPA_S3 $COD_SHA256" "resultados.tgz $S3B $RESB_SHA256" "faseA.tar $S3A $FASEA_SHA256"; do
  set -- $par
  s3cp "$2/$1" "$BASE/in/$1" && sha_ok "$3" "$BASE/in/$1" || { fallo "$1: descarga o sha256"; estado entradas "FALLO $1"; cierre; }
done
s3cp "$S3A/faseA.json" "$BASE/in/faseA.json" || { fallo "faseA.json: descarga"; estado entradas "FALLO faseA.json"; cierre; }
tar -xzf "$BASE/in/codigo.tgz" -C "$BASE/codigo" || { fallo "codigo.tgz"; estado entradas "FALLO tgz"; cierre; }
find "$BASE/codigo" -name '._*' -delete
[ -f "$BASE/codigo/model/alt/aws/w1_fs8b2_fases.sh" ] || { fallo "codigo.tgz sin las fases"; estado entradas "FALLO fases"; cierre; }
sha_ok "$CRIT_SHA" "$BASE/codigo/model/alt/crit-w1-fs8-e2e.json" || { fallo "criterio distinto del congelado"; estado entradas "FALLO criterio"; cierre; }
estado entradas "OK codigo, resultados.tgz de B y faseA.tar (sha256 comprobados), criterio ${CRIT_SHA:0:12}"
. "$BASE/codigo/model/alt/aws/w1_fs8b2_fases.sh" || { fallo "fases"; cierre; }
fases_main
