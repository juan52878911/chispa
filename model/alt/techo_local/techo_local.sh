#!/bin/bash
CHISPA_S3=${CHISPA_S3_LOCAL:-__CHISPA_S3__}   # CHISPA_S3_LOCAL (con CHISPA_LOCAL=1): directorio en vez de S3, solo para pruebas
# Techo local (W4): ¿un LLM abierto más grande servido en local (vLLM) cambia el L3? Una sola g6e.xlarge (L40S 48 GB) BAJO DEMANDA, volumen raíz >= 160 GB, rol rol-ec2-pruebas (sin ListBucket).
# Tope duro CHISPA_MIN=110 min (g6e.xlarge ~1,861 USD/h: ~3,41 USD de cómputo + ~0,04 de EBS; tope de la corrida 4 USD): `shutdown -h +MIN` desde el arranque y terminate al apagar.
# Modelos, en orden de prioridad (lo último en cortarse es lo más prometedor): qwen3-4b (referencia pareada, misma GPU y código), qwen3-30b-a3b (FP8), gpt-oss-20b (MXFP4, cliente harmony).
# Criterio congelado: model/alt/crit-techo-local.json (sha256 abajo; trae las revisiones). Licencia (Apache-2.0 o MIT) comprobada en la tarjeta DENTRO del ud, antes de bajar los pesos.
# Lee por nombre exacto $CHISPA_S3/{codigo.tgz,dev.tgz} (sha256 abajo); escribe $CHISPA_S3/{log.txt,listo,resultados.tgz} (SOLO cifras, predicciones y licencias: nunca textos de dev);
# espera $CHISPA_S3/verificado. Las fases viven en codigo.tgz (model/alt/techo_local/techo_local_fases.sh); el user-data de EC2 admite 16 384 bytes.
MIN=${CHISPA_MIN:-110}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-6}; HILOS=${CHISPA_HILOS:-32}; VLLM=${CHISPA_VLLM:-0.33.0}
D4=${CHISPA_D4:-22}; D30=${CHISPA_D30:-38}; DOSS=${CHISPA_DOSS:-28}   # minutos máximos por modelo (sin la descarga); lo hecho queda aunque se corte
MODELOS=${CHISPA_MODELOS:-"qwen3-4b qwen3-30b-a3b gpt-oss-20b"}
REV4=cdbee75f17c01a7cc42f958dc650907174af0554; REV30=__REV_Q30__; REVOSS=__REV_OSS__; E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3; TRANSFORMERS=5.22.2
CRIT_SHA=__CRIT_SHA__; COD_SHA256=__COD_SHA256__; DEV_SHA256=__DEV_SHA256__; N_TEST=__N_TEST__; N_SOP=__N_SOP__
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-tl-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-tl}
mkdir -p "$BASE/res/pred"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque techo-local $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min modelos=$MODELOS"
OUT=$BASE/res; PRED=$OUT/pred; FALLO=""; VP=""; WATCH=0
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
sha_ok() { if command -v sha256sum >/dev/null; then echo "$1  $2" | sha256sum -c --quiet -; else [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" = "$1" ]; fi; }
parar() { pkill -TERM -f "$1" 2>/dev/null; for _ in $(seq 30); do pgrep -f "$1" > /dev/null || return 0; sleep 2; done; pkill -KILL -f "$1" 2>/dev/null; sleep 3; }
subir_y_apagar() {
  trap - EXIT; kill "${WATCH:-0}" 2>/dev/null; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"
  [ -d "$BASE/codigo/model/alt" ] && [ -d "$BASE/dev" ] && [ -f "$BASE/t/rubrica.json" ] && ( cd "$BASE/codigo" && PYTHONPATH=$BASE/codigo timeout 300 "${PY:-python3}" -m model.alt.techo_local medir --dev "$BASE/dev" \
      --pred "$PRED" --rubrica "$BASE/t/rubrica.json" --licencias "$OUT/licencias" --salida "$OUT/cifras.json" ) > "$OUT/medir.txt" 2>&1; cat "$OUT/medir.txt" 2>/dev/null
  cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  for l in "$BASE"/vllm-*.log; do [ -f "$l" ] && tail -n 200 "$l" > "$OUT/$(basename "$l" .log)-cola.log"; done
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
( while sleep 30; do [ "$(quedan_min)" -le $((5 + ESPERA_MIN)) ] && { echo "== tope: se cierra"; pkill -TERM -f techo_local_cliente; pkill -TERM -f llm_logprobs; pkill -TERM -f model.alt.techo_local; pkill -TERM -f vllm.entrypoints; break; }; done ) & WATCH=$!

echo "== 1. entradas $(date -u +%T)"
mkdir -p "$BASE/in" "$BASE/codigo" "$BASE/dev"
for f in codigo dev; do
  sha=$COD_SHA256; [ "$f" = dev ] && sha=$DEV_SHA256
  s3cp "$CHISPA_S3/$f.tgz" "$BASE/in/$f.tgz" && sha_ok "$sha" "$BASE/in/$f.tgz" || { fallo "$f.tgz: descarga o sha256"; estado entradas "FALLO $f.tgz"; cierre; }
done
tar -xzf "$BASE/in/codigo.tgz" -C "$BASE/codigo" && tar -xzf "$BASE/in/dev.tgz" -C "$BASE/dev" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
find "$BASE/dev" "$BASE/codigo" -name '._*' -delete
[ -f "$BASE/codigo/model/alt/techo_local/techo_local_fases.sh" ] || { fallo "codigo.tgz sin las fases"; estado entradas "FALLO fases"; cierre; }
[ "$(ls "$BASE"/dev/*.jsonl | wc -l | tr -d ' ')" = 6 ] && [ "$(ls "$BASE/dev" | wc -l | tr -d ' ')" = 6 ] || { fallo "dev.tgz no trae solo 6 jsonl"; estado entradas "FALLO dev"; cierre; }
sha_ok "$CRIT_SHA" "$BASE/codigo/model/alt/crit-techo-local.json" || { fallo "criterio distinto del congelado"; estado entradas "FALLO criterio"; cierre; }
estado entradas "OK codigo y dev (sha256 comprobados), criterio ${CRIT_SHA:0:12}"
. "$BASE/codigo/model/alt/techo_local/techo_local_fases.sh" || { fallo "fases"; cierre; }
fases_main
cierre
