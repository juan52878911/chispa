#!/usr/bin/env bash
# R10-c, c7i: codigo.tgz de SHA (con crit-r10-c.json), dev.tgz (SOLO los 6 .jsonl de dev), oro32, fondo, fondo-fuente y lp_reetiqueta (resultado de la g5).
# El ud comprueba el sha256 de crit-r10-c.json antes de medir. No sube nada.
# Uso: bash model/alt/aws/empaquetar_r10_c_medir.sh SHA DIR_DEV ORO32 FONDO FONDO_FUENTE LP_REETIQUETA DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$1; dev=$2; oro=$3; fondo=$4; ffu=$5; lp=$6; sal=$7; mkdir -p "$sal"
export COPYFILE_DISABLE=1
git archive --format=tar.gz "$sha" -- model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py \
  model/l2/largos.py model/l2/modelo.py model/l2/evaluar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py sdk/python \
  ':(exclude)model/alt/resultados*' > "$sal/codigo.tgz"
# solo dev: los nombres tienen que ser los de dev del manifiesto, ninguno sellado
python3 - "$dev" "$sha" <<'PY'
import json, subprocess, sys
from pathlib import Path
m = json.loads(subprocess.check_output(["git", "show", sys.argv[2] + ":model/alt/bateria-v2-manifiesto.json"]))
dev = sorted(t["nombre"] + ".jsonl" for t in m["tareas"] if t["particion"] == "dev")
hay = sorted(p.name for p in Path(sys.argv[1]).glob("*.jsonl"))
assert hay == dev, "DIR_DEV tiene %s y dev es %s" % (hay, dev)
PY
tar -czf "$sal/dev.tgz" -C "$dev" $(cd "$dev" && ls ./*.jsonl | sed 's#^\./##')
for x in codigo dev; do tar -tzf "$sal/$x.tgz" | grep -q '\._' && { echo "$x.tgz trae ._*" >&2; exit 1; }; done
tar -tzf "$sal/dev.tgz" | grep -qi sellad && { echo "dev.tgz trae selladas" >&2; exit 1; }
cp "$oro" "$sal/oro32.jsonl"; cp "$fondo" "$sal/fondo.jsonl"; cp "$ffu" "$sal/fondo-fuente.jsonl"; cp "$lp" "$sal/lp_reetiqueta.jsonl"
csha=$(git show "$sha:model/alt/crit-r10-c.json" | shasum -a 256 | cut -d' ' -f1)
f="model/alt/aws/lanzados/ud-r10-c-medir-$sha.sh"
git show "$sha:model/alt/aws/r10_c_medir.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-879381269751/r10-c-medir-$sha#" -e "s#__CRIT_SHA__#$csha#" \
  -e "s#__N_FONDO__#$(wc -l < "$sal/fondo.jsonl" | tr -d ' ')#" > "$f"
[ "$(wc -c < "$f")" -le 16384 ] || { echo "el ud pasa de 16 KB" >&2; exit 1; }
bash -n "$f"
wc -c "$f"; (cd "$sal" && shasum -a 256 ./*)
