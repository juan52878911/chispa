#!/usr/bin/env bash
# Batería v3, fase 2: empaqueta lo que lee bateria_v3.sh y escribe su user-data. NO sube ni lanza nada: se sube a mano, por nombre exacto, a
# s3://chispa-pruebas-cuenta-aws/bateria-v3-SHA/codigo.tgz, y el ud se lanza (si Juan lo decide) en una c7i.xlarge con volumen >= 30 GB.
#   codigo.tgz    de `git archive SHA`: model/__init__.py, model/datos/{__init__,bateria_v3*.py}, model/datos/bateria-v3-manifiesto.json, model/datos/aws/bateria_v3_fases.sh y
#                 model/alt/manifiesto-entrenamiento.json (las fuentes públicas del entrenamiento de las que el ud saca huellas).
#   Ya no hay huellas.tgz: el ud baja dev.tgz y selladas.tgz de v2 (nombre exacto y sha256 fijos en el propio ud) y hace TODAS las huellas dentro de la c7i.
#   model/datos/aws/lanzados/ud-bateria-v3-SHA.sh   bateria_v3.sh de SHA con __CHISPA_S3__ y los sha256 de codigo.tgz y del manifiesto y el nº de tareas (cabe en los 16 384 bytes de user-data).
# CHISPA_UD_DIR cambia el directorio del ud (pruebas en seco: no deja nada en lanzados/).
# Uso: bash model/datos/aws/empaquetar_bateria_v3.sh SHA DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; sal=${2:?DIR_SALIDA}
mkdir -p "$sal"; sal=$(cd "$sal" && pwd)
COD="model/__init__.py model/datos/__init__.py model/datos/bateria_v3.py model/datos/bateria_v3_candidatas.py model/datos/bateria_v3_licencias.py model/datos/bateria_v3_solapes.py model/datos/bateria_v3_huellas.py model/alt/manifiesto-entrenamiento.json
  model/datos/bateria-v3-manifiesto.json model/datos/aws/bateria_v3_fases.sh"
for f in $COD model/datos/aws/bateria_v3.sh; do git cat-file -e "$sha:$f" || { echo "$sha no trae $f" >&2; exit 1; }; done
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
git archive "$sha" $COD | tar -x -C "$t"
python3 -I - "$t/model/datos/bateria-v3-manifiesto.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["version"] == 3 and m["tareas"], "manifiesto vacío"
assert all(not t["sha256"] and not t["licencia_sha256"] for t in m["tareas"]), "el manifiesto empaquetado debe ser el de fase 1 (sin sha256)"
PY
( cd "$t" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$t python3 -c "import model.datos.bateria_v3, model.datos.bateria_v3_solapes, model.datos.bateria_v3_licencias, model.datos.bateria_v3_huellas" ) || { echo "el código de $sha no importa" >&2; exit 1; }
find "$t" -name '._*' -delete
export COPYFILE_DISABLE=1
tarz() { tar --no-xattrs -czf "$1" -C "$2" "${@:3}" 2>/dev/null || tar -czf "$1" -C "$2" "${@:3}"; }
rm -f "$sal/codigo.tgz" "$sal/huellas.tgz"           # huellas.tgz ya no existe: se borra por si quedó de una versión anterior
tarz "$sal/codigo.tgz" "$t" model
tar -tzf "$sal/codigo.tgz" | grep -q '\._' && { echo "codigo.tgz trae ._*" >&2; exit 1; }
tar -tzf "$sal/codigo.tgz" | grep -q '__pycache__' && { echo "codigo.tgz trae __pycache__" >&2; exit 1; }
shf() { shasum -a 256 "$1" | cut -d' ' -f1; }
ntareas=$(python3 -I -c "import json,sys;print(len(json.load(open(sys.argv[1]))['tareas']))" "$t/model/datos/bateria-v3-manifiesto.json")
udd=${CHISPA_UD_DIR:-model/datos/aws/lanzados}; mkdir -p "$udd"
ud=$udd/ud-bateria-v3-$sha.sh
git show "$sha:model/datos/aws/bateria_v3.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/bateria-v3-$sha#" -e "s#__COD_SHA256__#$(shf "$sal/codigo.tgz")#" \
  -e "s#__MAN_SHA256__#$(shf "$t/model/datos/bateria-v3-manifiesto.json")#" -e "s#__N_TAREAS__#$ntareas#" > "$ud"
bash -n "$ud"
grep -n '__[A-Z0-9_]*__' "$ud" | grep -v 'CHISPA_S3_LOCAL' && { echo "quedan placeholders sin sustituir" >&2; exit 1; }
[ "$(wc -c < "$ud")" -lt 15500 ] || { echo "el ud pesa $(wc -c < "$ud") bytes: el user-data de EC2 admite 16384 como máximo (objetivo < 15500)" >&2; exit 1; }
bash -n <(git show "$sha:model/datos/aws/bateria_v3_fases.sh")
(cd "$sal" && ls -l codigo.tgz && shasum -a 256 codigo.tgz)
echo "ud: $ud ($(wc -c < "$ud") bytes, sha256 $(shf "$ud")); tareas=$ntareas"
echo "sube a s3://chispa-pruebas-cuenta-aws/bateria-v3-$sha/: codigo.tgz; lanza $ud en una c7i.xlarge con volumen >= 30 GB (tope 90 min, ~0,28 USD)"
