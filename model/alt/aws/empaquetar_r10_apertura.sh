#!/usr/bin/env bash
# R10-apertura: empaqueta lo que lee r10_apertura.sh y escribe sus DOS user-data (partes a y b). No sube nada.
#   DIR_SALIDA/codigo.tgz     el código de SHA (lo commiteado; incluye crit-r10-apertura.json), sin ._*
#   DIR_SALIDA/selladas.tgz   las 8 selladas (o, con --dev-como-selladas, los jsonl de DESARROLLO como ensayo)
#   DIR_SALIDA/aperturas-nuevas.jsonl   (solo real) las líneas de registro de ESTA apertura: añadirlas a model/alt/bateria-v2-aperturas.jsonl y commitear
#   model/alt/aws/lanzados/ud-r10-apertura-SHA-{a,b}.sh   r10_apertura.sh de SHA con __CHISPA_S3__=s3://chispa-pruebas-879381269751/r10-apertura-SHA-{a,b}, el sha256 del
#                             criterio de SHA, los recuentos y la parte; cada parte necesita codigo.tgz y selladas.tgz en SU prefijo.
# MODO REAL (lo corre la sesión principal, en el i7: ssh px-01-i7, datos en /mnt/pve/nfs-ascci/chispa-audio/bateria-v2/selladas, dentro de un checkout que tenga SHA):
#   la ÚNICA lectura de las selladas es apertura_v2.py sellar → bateria_v2.abrir_sellada(ruta, motivo, manifiesto=...), que anota la apertura en el registro ANTES de leer.
#   Se niega si el criterio aún trae marcadores __...__ (sin congelar) o si el registro de SHA ya trae una apertura de cualquiera de esos ficheros (apertura única).
# MODO ENSAYO (--dev-como-selladas): empaqueta dev (CHISPA_SOLO=tarea1,tarea2 para un subconjunto) como si fuera sellada; no abre nada, no registra nada.
# Uso: bash model/alt/aws/empaquetar_r10_apertura.sh [--dev-como-selladas] SHA DIR_DATOS DIR_SALIDA     (DIR_DATOS = selladas/, o dev/ con --dev-como-selladas)
set -euo pipefail
ensayo=0; [ "${1:-}" = "--dev-como-selladas" ] && { ensayo=1; shift; }
# Sin git (el i7 no lo trae): CHISPA_ARBOL=DIR usa un árbol ya extraído del commit (git archive SHA | tar -x -C DIR) y SHA es solo la etiqueta.
if [ -n "${CHISPA_ARBOL:-}" ]; then cd "$CHISPA_ARBOL"; fuente() { cat "$1"; }; existe() { [ -e "$1" ]; }
  archivar() { tar --exclude='model/alt/resultados*' --exclude='._*' --exclude='*/__pycache__' -czf - "$@"; }; LAN=${CHISPA_LANZADOS:-$PWD/lanzados}
else cd "$(git rev-parse --show-toplevel)"; fuente() { git show "$SHA:$1"; }; existe() { git cat-file -e "$SHA:$1"; }
  archivar() { git archive --format=tar.gz "$SHA" -- "$@" ':(exclude)model/alt/resultados*'; }; LAN=model/alt/aws/lanzados; fi
sha=${1:?SHA}; SHA=$sha; datos=${2:?DIR_DATOS}; sal=${3:?DIR_SALIDA}
datos=$(cd "$datos" && pwd); mkdir -p "$sal"; sal=$(cd "$sal" && pwd)
M=model/alt
for f in $M/aws/r10_apertura.sh $M/crit-r10-apertura.json $M/apertura_v2.py $M/bateria-v2-manifiesto.json $M/bateria-v2-aperturas.jsonl; do existe "$f" || { echo "$sha no trae $f" >&2; exit 1; }; done
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
if [ -n "${CHISPA_ARBOL:-}" ]; then tar -c --exclude=lanzados . | tar -x -C "$tmp"; else git archive --format=tar "$sha" | tar -x -C "$tmp"; fi   # el código de SHA, con el que se corre `sellar`/`contar`
export COPYFILE_DISABLE=1 PYTHONPATH="$tmp" PYTHONDONTWRITEBYTECODE=1
PY=${CHISPA_PYTHON:-python3}
rm -f "$sal/selladas.tgz" "$sal/conteo.txt"; mkdir -p "$tmp/datos"
if [ $ensayo = 1 ]; then
  modo=ensayo
  if [ -n "${CHISPA_SOLO:-}" ]; then want=$(for t in ${CHISPA_SOLO//,/ }; do printf '%s.jsonl ' "$t"; done)
  else want="cfpb-quejas.jsonl congreso-eeuu-partido.jsonl finanzas-documentos.jsonl horeca-resenas.jsonl jql-valor-educativo.jsonl usajobs-nivel.jsonl "; fi
  hay=$(cd "$datos" && ls | grep -v '^\._' | tr '\n' ' ')
  [ -n "${CHISPA_SOLO:-}" ] || [ "$hay" = "$want" ] || { echo "DIR_DATOS (dev) debe traer exactamente los 6 jsonl de dev; trae: $hay" >&2; exit 1; }
  for f in $want; do [ -f "$datos/$f" ] || { echo "falta $f en $datos" >&2; exit 1; }; cp "$datos/$f" "$tmp/datos/$f"; done
  ( cd "$tmp" && $PY -m model.alt.apertura_v2 contar --datos datos ) > "$tmp/cuenta.txt"
else
  modo=real
  if fuente "$M/crit-r10-apertura.json" | grep -Eq '__[A-Z0-9_]{4,}__'; then echo "el criterio de $sha aún trae marcadores __...__: sin congelar, no se abre nada" >&2; exit 1; fi
  [ ! -s "$sal/aperturas-nuevas.jsonl" ] || { echo "$sal/aperturas-nuevas.jsonl ya existe: apertura única, no se repite" >&2; exit 1; }
  fuente "$M/bateria-v2-aperturas.jsonl" > "$tmp/previo.jsonl"
  motivo=$(fuente "$M/crit-r10-apertura.json" | $PY -I -c "import json, sys; print(json.load(sys.stdin)['motivo'])")
  ( cd "$tmp" && $PY -m model.alt.apertura_v2 sellar --manifiesto "$M/bateria-v2-manifiesto.json" --selladas "$datos" --salida "$tmp/datos" --motivo "$motivo" \
      --registro-previo "$tmp/previo.jsonl" --registro-nuevo "$sal/aperturas-nuevas.jsonl" )
  cp "$tmp/datos/conteo.txt" "$tmp/cuenta.txt"; rm -f "$tmp/datos/conteo.txt"
fi
read -r nta nt ns < "$tmp/cuenta.txt"
archivar model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py \
  model/l2/largos.py model/l2/modelo.py model/l2/evaluar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py sdk/python > "$sal/codigo.tgz"
( cd "$tmp/datos" && ls *.jsonl > /dev/null && { tar --no-xattrs -czf "$sal/selladas.tgz" *.jsonl 2>/dev/null || tar -czf "$sal/selladas.tgz" *.jsonl; } )
for x in codigo selladas; do tar -tzf "$sal/$x.tgz" | grep -q '\._' && { echo "$x.tgz trae ._*" >&2; exit 1; }; done
crit=$(fuente "$M/crit-r10-apertura.json" | { sha256sum 2>/dev/null || shasum -a 256; } | cut -d' ' -f1)
mkdir -p "$LAN"
for p in a b; do
  ud=$LAN/ud-r10-apertura-$sha-$p.sh
  fuente "$M/aws/r10_apertura.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-879381269751/r10-apertura-$sha-$p#" -e "s#__CRIT_SHA__#$crit#" -e "s#__N_TAREAS__#$nta#" \
    -e "s#__N_TEST__#$nt#" -e "s#__N_SOP__#$ns#" -e "s#__PARTE__#$p#" -e "s#__MODO__#$modo#" > "$ud"
  bash -n "$ud"
  grep -n '__[A-Z0-9_]*__' "$ud" && { echo "quedan placeholders sin sustituir" >&2; exit 1; }
  [ "$(wc -c < "$ud")" -le 16384 ] || { echo "el ud pasa de 16 KB" >&2; exit 1; }
  wc -c "$ud"
done
(cd "$sal" && { sha256sum codigo.tgz selladas.tgz 2>/dev/null || shasum -a 256 codigo.tgz selladas.tgz; }); echo "modo $modo; criterio sha256 $crit; tareas=$nta test=$nt soporte=$ns"
[ $ensayo = 1 ] || { echo "registro de esta apertura (añadir a $M/bateria-v2-aperturas.jsonl):"; cat "$sal/aperturas-nuevas.jsonl"; }
for p in a b; do echo "sube a s3://chispa-pruebas-879381269751/r10-apertura-$sha-$p/: codigo.tgz y selladas.tgz; lanza $LAN/ud-r10-apertura-$sha-$p.sh"; done
