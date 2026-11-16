#!/usr/bin/env bash
# R10-T3: empaqueta lo que lee r10_t3.sh y escribe su user-data. No sube nada.
#   DIR_SALIDA/codigo.tgz  el código de SHA (lo commiteado; mismo conjunto que R8b: model/alt, datos, meta, evaluacion, l2 mínimo, sdk/python), sin ._*
#   DIR_SALIDA/dev.tgz     SOLO los 6 jsonl de DESARROLLO de DIR_DEV (bateria-v2/dev); las selladas no se abren ni se copian: si DIR_DEV trae algo más, falla
#   model/alt/aws/lanzados/ud-r10-t3-SHA.sh   r10_t3.sh de SHA con __CHISPA_S3__=s3://chispa-pruebas-cuenta-aws/r10-t3-SHA, el sha256 de
#                          crit-r10-t3.json de SHA y los recuentos (filas de test y de soporte de dev).
# Uso: bash model/alt/aws/empaquetar_r10_t3.sh SHA DIR_DEV DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; dev=${2:?DIR_DEV}; sal=${3:?DIR_SALIDA}
dev=$(cd "$dev" && pwd); mkdir -p "$sal"; sal=$(cd "$sal" && pwd)
git cat-file -e "$sha:model/alt/aws/r10_t3.sh" && git cat-file -e "$sha:model/alt/crit-r10-t3.json" || { echo "$sha no trae r10_t3.sh y crit-r10-t3.json" >&2; exit 1; }
esperadas="cfpb-quejas congreso-eeuu-partido finanzas-documentos horeca-resenas jql-valor-educativo usajobs-nivel"
hay=$(cd "$dev" && ls | grep -v '^\._' | tr '\n' ' ')
want=$(for t in $esperadas; do printf '%s.jsonl ' "$t"; done)
[ "$hay" = "$want" ] || { echo "DIR_DEV debe traer exactamente los 6 jsonl de dev; trae: $hay" >&2; exit 1; }
mkdir -p "$sal"; export COPYFILE_DISABLE=1
git archive --format=tar.gz "$sha" -- model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py \
  model/l2/largos.py model/l2/modelo.py model/l2/evaluar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py sdk/python \
  ':(exclude)model/alt/resultados*' > "$sal/codigo.tgz"
rm -f "$sal/dev.tgz"
( cd "$dev" && { tar --no-xattrs -czf "$sal/dev.tgz" $want 2>/dev/null || tar -czf "$sal/dev.tgz" $want; } )
for x in codigo dev; do tar -tzf "$sal/$x.tgz" | grep -q '\._' && { echo "$x.tgz trae ._*" >&2; exit 1; }; done
tar -tzf "$sal/dev.tgz" | grep -qi 'sellad' && { echo "dev.tgz trae algo sellado" >&2; exit 1; }
crit=$(git show "$sha:model/alt/crit-r10-t3.json" | shasum -a 256 | cut -d' ' -f1)
cuenta=$(python3 -I - "$dev" <<'PY'
import glob, json, sys
t = s = 0
for f in sorted(glob.glob(sys.argv[1] + "/*.jsonl")):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            sp = json.loads(l)["split"]
            t += sp == "test"
            s += sp == "train"
print(t, s)
PY
)
nt=${cuenta% *}; ns=${cuenta#* }
ud=model/alt/aws/lanzados/ud-r10-t3-$sha.sh
mkdir -p model/alt/aws/lanzados
git show "$sha:model/alt/aws/r10_t3.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/r10-t3-$sha#" -e "s#__CRIT_SHA__#$crit#" -e "s#__N_TEST__#$nt#" -e "s#__N_SOP__#$ns#" > "$ud"
bash -n "$ud"
grep -n '__[A-Z0-9_]*__' "$ud" && { echo "quedan placeholders sin sustituir" >&2; exit 1; }
[ "$(wc -c < "$ud")" -le 16384 ] || { echo "el ud pasa de 16 KB" >&2; exit 1; }
wc -c "$ud"; (cd "$sal" && shasum -a 256 codigo.tgz dev.tgz); echo "criterio sha256 $crit; test=$nt soporte=$ns"
echo "sube a s3://chispa-pruebas-cuenta-aws/r10-t3-$sha/: codigo.tgz y dev.tgz; lanza $ud (lee ademas ajuste-80ab8ef/ckpt/mejor.pt)"
