#!/bin/bash
CHISPA_S3=${CHISPA_S3_LOCAL:-__CHISPA_S3__}   # CHISPA_S3_LOCAL (con CHISPA_LOCAL=1): directorio en vez de S3, solo para pruebas en seco
# Batería v3, FASE 2: hace en la propia instancia las huellas de v2 (dev y selladas) y del entrenamiento (fuentes públicas), baja las candidatas por su URL y revisión, lee y guarda cada licencia,
#   comprueba la lista admitida, mide solapes, parte, sella y sube. Los textos de v2 y del entrenamiento NO salen de la instancia: solo hashes y sketches (y el log se audita contra ellos).
# Máquina: c7i.xlarge (4 vCPU, 8 GB), volumen raíz de 80 GB (tsjev se baja fichero a fichero: el mayor pesa hasta 40 GB), rol rol-ec2-pruebas (sin ListBucket), CON red saliente (Hugging Face, NOAA, GitHub...). NO lanzar sin que Juan lo decida.
# COSTE: tope duro CHISPA_MIN = 90 min con autoapagado (shutdown -h +90) -> 90/60 h x 0,1785 USD/h (el que usa LEEME §19.3 para c7i.xlarge en us-east-1; COMPROBAR en la consola: otra región o un precio
#   vigente distinto cambian la cuenta: a 0,20 USD/h serían 0,30) = 0,268 USD, más ~0,013 USD de volumen de 80 GB (gp3: 80 × 0,08 USD/GB-mes ÷ 730 h × 1,5 h) = ~0,281 USD < 0,3 USD. Sin GPU, sin llamadas de pago a ningún LLM.
# Lee por nombre exacto $CHISPA_S3/codigo.tgz y los dos .tgz de v2 de abajo (el rol no tiene ListBucket; sha256 comprobados antes de abrirlos); escribe $CHISPA_S3/{manifiesto.json,licencias.tgz,datos-sellados.tgz,resumen.json,reservas-v2.json,log.txt,listo}
# y espera $CHISPA_S3/verificado. NUNCA sube texto de test fuera de datos-sellados.tgz ni lo imprime: el log se audita antes de subirlo (si trae texto de test no se sube).
MIN=${CHISPA_MIN:-90}; ESPERA_MIN=${CHISPA_ESPERA_MIN:-5}
COD_SHA256=__COD_SHA256__; MAN_SHA256=__MAN_SHA256__; N_TAREAS=__N_TAREAS__
DEV_URI=${CHISPA_V3_DEV:-s3://chispa-pruebas-cuenta-aws/w1-fs8-27fdfc9/dev.tgz}; DEV_SHA256=ce0423dc05a5b393067b6aeb2c1da4aa4d6192d249d1a52061c5f579e8dd3ca9
SEL_URI=${CHISPA_V3_SELLADAS:-s3://chispa-pruebas-cuenta-aws/r10-apertura-3d86859-a/selladas.tgz}; SEL_SHA256=a2bb334e5a1e0961a3d79f011d9525dfb872112d5e305a14303dbf16b5f487a0
HUB=2.4.0; PYARROW=26.0.0; PYYAML=6.0.3
LOCAL=${CHISPA_LOCAL:-0}
if [ "$LOCAL" = 1 ]; then case $CHISPA_S3 in __*) CHISPA_S3=/tmp/chispa-b3-s3; mkdir -p "$CHISPA_S3";; esac; else shutdown -h +$MIN; fi
set -uo pipefail
export HOME="${HOME:-/root}"; T0=$(date +%s)
BASE=${CHISPA_BASE:-/opt/chispa}; [ "$LOCAL" = 1 ] && BASE=${CHISPA_BASE:-/tmp/chispa-b3}
mkdir -p "$BASE/res"; exec > >(tee -a "$BASE/log.txt") 2>&1
echo "== arranque bateria-v3 fase 2 $(date -u +%FT%TZ) local=$LOCAL tope=${MIN} min"
OUT=$BASE/res; FALLO=""
quedan_min() { echo $(( MIN - ($(date +%s) - T0) / 60 )); }
s3cp() { if [ "$LOCAL" = 1 ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; else aws s3 cp --only-show-errors "$1" "$2"; fi; }
estado() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/estado.tsv"; echo "[$1] $2"; }
fallo() { echo "FALLO: $*"; FALLO="${FALLO:+$FALLO; }$*"; }
sha_ok() { if command -v sha256sum >/dev/null; then echo "$1  $2" | sha256sum -c --quiet -; else [ "$(shasum -a 256 "$2" | cut -d' ' -f1)" = "$1" ]; fi; }
subir_y_apagar() {
  trap - EXIT; [ -n "$FALLO" ] && echo "$FALLO" > "$OUT/FALLO"
  cp "$BASE/log.txt" "$OUT/log.txt" 2>/dev/null
  # el log solo sube si no filtra texto de test, ni de v2 ni del entrenamiento (auditar-log abre las selladas con abrir_sellada y solo cuenta coincidencias)
  if [ -d "$OUT/datos/sellada" ] && [ -f "$BASE/codigo/model/datos/bateria_v3.py" ]; then
    EXTRA=""; for x in "$BASE/v2txt/v2-dev" "$BASE/v2txt/v2-selladas" "$BASE/muestra"; do [ -d "$x" ] && EXTRA="$EXTRA $x"; done
    ( cd "$BASE/codigo" && PYTHONPATH=$BASE/codigo "${PY:-python3}" -m model.datos.bateria_v3 auditar-log --datos "$OUT/datos" --log "$BASE/log.txt" ${EXTRA:+--textos-extra $EXTRA} ) > "$OUT/auditoria.txt" 2>&1
    [ "$?" = 0 ] || { echo "AUDITORIA: el log trae texto de test; no se sube log.txt"; rm -f "$OUT/log.txt"; FALLO="${FALLO:+$FALLO; }log con texto de test"; }
    cat "$OUT/auditoria.txt"
  fi
  rm -rf "$BASE/v2txt" "$BASE/muestra" "$BASE/entr"      # los textos de v2 y del entrenamiento no se guardan ni suben
  D=$OUT/datos
  [ -f "$D/manifiesto.json" ] && s3cp "$D/manifiesto.json" "$CHISPA_S3/manifiesto.json"
  [ -f "$D/resumen.json" ] && s3cp "$D/resumen.json" "$CHISPA_S3/resumen.json"
  [ -f "$D/reservas-v2.json" ] && s3cp "$D/reservas-v2.json" "$CHISPA_S3/reservas-v2.json"
  LIC=""; for x in licencias reservas-v2; do [ -d "$D/$x" ] && LIC="$LIC $x"; done
  [ -n "$LIC" ] && tar -czf "$OUT/licencias.tgz" -C "$D" $LIC && s3cp "$OUT/licencias.tgz" "$CHISPA_S3/licencias.tgz"
  [ -d "$D/sellada" ] && tar -czf "$OUT/datos-sellados.tgz" -C "$OUT" datos && s3cp "$OUT/datos-sellados.tgz" "$CHISPA_S3/datos-sellados.tgz"
  [ -f "$OUT/log.txt" ] && s3cp "$OUT/log.txt" "$CHISPA_S3/log.txt"
  if [ "$LOCAL" != 1 ]; then
    echo listo > "$BASE/listo"; s3cp "$BASE/listo" "$CHISPA_S3/listo" || true
    for _ in $(seq $((ESPERA_MIN * 2))); do aws s3 cp --only-show-errors "$CHISPA_S3/verificado" "$BASE/verificado" 2>/dev/null && break; sleep 30; done
    shutdown -h now
  fi
}
trap subir_y_apagar EXIT; trap 'exit 143' TERM INT HUP
cierre() { echo "== listo $(date -u +%FT%TZ) fallo=${FALLO:-no} ($(( $(date +%s) - T0 )) s)"; exit 0; }

echo "== 1. entradas $(date -u +%T)"
mkdir -p "$BASE/in" "$BASE/codigo" "$BASE/huellas"
s3cp "$CHISPA_S3/codigo.tgz" "$BASE/in/codigo.tgz" && sha_ok "$COD_SHA256" "$BASE/in/codigo.tgz" || { fallo "codigo.tgz: descarga o sha256"; estado entradas "FALLO codigo.tgz"; cierre; }
s3cp "$DEV_URI" "$BASE/in/dev.tgz" && sha_ok "$DEV_SHA256" "$BASE/in/dev.tgz" || { fallo "dev.tgz de v2: descarga o sha256"; estado entradas "FALLO dev.tgz"; cierre; }
s3cp "$SEL_URI" "$BASE/in/selladas.tgz" && sha_ok "$SEL_SHA256" "$BASE/in/selladas.tgz" || { fallo "selladas.tgz de v2: descarga o sha256"; estado entradas "FALLO selladas.tgz"; cierre; }
tar -xzf "$BASE/in/codigo.tgz" -C "$BASE/codigo" || { fallo "tgz"; estado entradas "FALLO tgz"; cierre; }
find "$BASE/codigo" -name '._*' -delete
[ -f "$BASE/codigo/model/datos/aws/bateria_v3_fases.sh" ] || { fallo "codigo.tgz sin las fases"; estado entradas "FALLO fases"; cierre; }
MANIF=${CHISPA_V3_MANIFIESTO:-$BASE/codigo/model/datos/bateria-v3-manifiesto.json}
sha_ok "$MAN_SHA256" "$MANIF" || { fallo "manifiesto distinto del empaquetado"; estado entradas "FALLO manifiesto"; cierre; }
[ -f "$BASE/codigo/model/alt/manifiesto-entrenamiento.json" ] || { fallo "codigo.tgz sin el manifiesto de entrenamiento"; estado entradas "FALLO manifiesto de entrenamiento"; cierre; }
estado entradas "OK codigo, dev y selladas de v2 y manifiesto (sha256 comprobados)"
. "$BASE/codigo/model/datos/aws/bateria_v3_fases.sh" || { fallo "fases"; cierre; }
fases_main
