#!/usr/bin/env bash
# Batería v3, fase 2: empaqueta lo que lee bateria_v3.sh y escribe su user-data. NO sube ni lanza nada: se suben a mano, por nombre exacto, a
# s3://chispa-pruebas-879381269751/bateria-v3-SHA/ los dos tgz de DIR_SALIDA, y el ud se lanza (si Juan lo decide) en una c7i.xlarge con volumen >= 30 GB.
#   codigo.tgz    de `git archive SHA`: model/__init__.py, model/datos/{__init__,bateria_v3*.py}, model/datos/bateria-v3-manifiesto.json y model/datos/aws/bateria_v3_fases.sh.
#   huellas.tgz   SOLO los `huellas-*.jsonl.gz` de DIR_HUELLAS (hashes y sketches de v2 y del entrenamiento, hechos con `python -m model.datos.bateria_v3_solapes huellas`); si una
#                 línea trae algún campo distinto de h, s, f (p. ej. texto en claro) o hay otros ficheros, falla.
#   model/datos/aws/lanzados/ud-bateria-v3-SHA.sh   bateria_v3.sh de SHA con __CHISPA_S3__ y los sha256 de los dos tgz, del manifiesto y el nº de tareas (cabe en los 16 384 bytes de user-data).
# CHISPA_UD_DIR cambia el directorio del ud (pruebas en seco: no deja nada en lanzados/).
# Uso: bash model/datos/aws/empaquetar_bateria_v3.sh SHA DIR_HUELLAS DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; hue=${2:?DIR_HUELLAS}; sal=${3:?DIR_SALIDA}
hue=$(cd "$hue" && pwd); mkdir -p "$sal"; sal=$(cd "$sal" && pwd)
COD="model/__init__.py model/datos/__init__.py model/datos/bateria_v3.py model/datos/bateria_v3_candidatas.py model/datos/bateria_v3_licencias.py model/datos/bateria_v3_solapes.py
  model/datos/bateria-v3-manifiesto.json model/datos/aws/bateria_v3_fases.sh"
for f in $COD model/datos/aws/bateria_v3.sh; do git cat-file -e "$sha:$f" || { echo "$sha no trae $f" >&2; exit 1; }; done
hay=$(cd "$hue" && ls | grep -v '^\._' | tr '\n' ' ')
[ -n "$hay" ] && ! echo "$hay" | tr ' ' '\n' | grep -v '^$' | grep -qv '^huellas-[A-Za-z0-9_.-]*\.jsonl\.gz$' || { echo "DIR_HUELLAS debe traer solo huellas-*.jsonl.gz; trae: $hay" >&2; exit 1; }
python3 -I - "$hue" <<'PY'
import glob, gzip, json, sys
n = 0
for f in sorted(glob.glob(sys.argv[1] + "/huellas-*.jsonl.gz")):
    for l in gzip.open(f, "rt", encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            if set(d) - {"h", "s", "f"} or not isinstance(d.get("h"), str) or len(d["h"]) != 64:
                sys.exit("%s trae una línea que no es una huella (campos %s)" % (f, sorted(d)))
            n += 1
print("huellas válidas:", n)
PY
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
git archive "$sha" $COD | tar -x -C "$t"
python3 -I - "$t/model/datos/bateria-v3-manifiesto.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["version"] == 3 and m["tareas"], "manifiesto vacío"
assert all(not t["sha256"] and not t["licencia_sha256"] for t in m["tareas"]), "el manifiesto empaquetado debe ser el de fase 1 (sin sha256)"
PY
( cd "$t" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$t python3 -c "import model.datos.bateria_v3, model.datos.bateria_v3_solapes, model.datos.bateria_v3_licencias" ) || { echo "el código de $sha no importa" >&2; exit 1; }
find "$t" -name '._*' -delete
export COPYFILE_DISABLE=1
tarz() { tar --no-xattrs -czf "$1" -C "$2" "${@:3}" 2>/dev/null || tar -czf "$1" -C "$2" "${@:3}"; }
rm -f "$sal/codigo.tgz" "$sal/huellas.tgz"
tarz "$sal/codigo.tgz" "$t" model
( cd "$hue" && tar -czf "$sal/huellas.tgz" huellas-*.jsonl.gz )
for x in codigo huellas; do tar -tzf "$sal/$x.tgz" | grep -q '\._' && { echo "$x.tgz trae ._*" >&2; exit 1; }; done
tar -tzf "$sal/codigo.tgz" | grep -q '__pycache__' && { echo "codigo.tgz trae __pycache__" >&2; exit 1; }
shf() { shasum -a 256 "$1" | cut -d' ' -f1; }
ntareas=$(python3 -I -c "import json,sys;print(len(json.load(open(sys.argv[1]))['tareas']))" "$t/model/datos/bateria-v3-manifiesto.json")
udd=${CHISPA_UD_DIR:-model/datos/aws/lanzados}; mkdir -p "$udd"
ud=$udd/ud-bateria-v3-$sha.sh
git show "$sha:model/datos/aws/bateria_v3.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-879381269751/bateria-v3-$sha#" -e "s#__COD_SHA256__#$(shf "$sal/codigo.tgz")#" \
  -e "s#__HUE_SHA256__#$(shf "$sal/huellas.tgz")#" -e "s#__MAN_SHA256__#$(shf "$t/model/datos/bateria-v3-manifiesto.json")#" -e "s#__N_TAREAS__#$ntareas#" > "$ud"
bash -n "$ud"
grep -n '__[A-Z0-9_]*__' "$ud" | grep -v 'CHISPA_S3_LOCAL' && { echo "quedan placeholders sin sustituir" >&2; exit 1; }
[ "$(wc -c < "$ud")" -lt 15500 ] || { echo "el ud pesa $(wc -c < "$ud") bytes: el user-data de EC2 admite 16384 como máximo (objetivo < 15500)" >&2; exit 1; }
bash -n <(git show "$sha:model/datos/aws/bateria_v3_fases.sh")
(cd "$sal" && ls -l codigo.tgz huellas.tgz && shasum -a 256 codigo.tgz huellas.tgz)
echo "ud: $ud ($(wc -c < "$ud") bytes, sha256 $(shf "$ud")); tareas=$ntareas"
echo "sube a s3://chispa-pruebas-879381269751/bateria-v3-$sha/: codigo.tgz y huellas.tgz; lanza $ud en una c7i.xlarge con volumen >= 30 GB (tope 75 min, ~0,23 USD)"
