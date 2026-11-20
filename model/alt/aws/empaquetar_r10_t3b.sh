#!/usr/bin/env bash
# R10-T3b: empaqueta lo que lee r10_t3b.sh y escribe su user-data. No sube nada.
#   DIR_SALIDA/codigo.tgz  el código de SHA (lo commiteado; incluye crit-r10-t3b.json y rubricas-r10-t3b.json), sin ._*
#   DIR_SALIDA/dev.tgz     SOLO los 6 jsonl de DESARROLLO de DIR_DEV; las selladas no se abren ni se copian: si DIR_DEV trae algo más, falla
#   DIR_SALIDA/t3.tgz      las predicciones de T3 que usa el criterio (4b-zs/ y 4b-fs8/ de DIR_T3PRED, sin ficheros vacíos); se comprueba el sha256 de cada
#                          una contra crit-r10-t3b.json de SHA antes de empaquetar
#   model/alt/aws/lanzados/ud-r10-t3b-SHA.sh   r10_t3b.sh de SHA con __CHISPA_S3__=s3://chispa-pruebas-879381269751/r10-t3b-SHA, el sha256 de
#                          crit-r10-t3b.json de SHA y los recuentos (filas de test y de soporte de dev).
# Uso: bash model/alt/aws/empaquetar_r10_t3b.sh SHA DIR_DEV DIR_T3PRED DIR_SALIDA     (DIR_T3PRED = la carpeta pred/ de los resultados de T3)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; dev=${2:?DIR_DEV}; t3=${3:?DIR_T3PRED}; sal=${4:?DIR_SALIDA}
dev=$(cd "$dev" && pwd); t3=$(cd "$t3" && pwd); mkdir -p "$sal"; sal=$(cd "$sal" && pwd)
for f in model/alt/aws/r10_t3b.sh model/alt/crit-r10-t3b.json model/alt/rubricas-r10-t3b.json; do git cat-file -e "$sha:$f" || { echo "$sha no trae $f" >&2; exit 1; }; done
esperadas="cfpb-quejas congreso-eeuu-partido finanzas-documentos horeca-resenas jql-valor-educativo usajobs-nivel"
hay=$(cd "$dev" && ls | grep -v '^\._' | tr '\n' ' ')
want=$(for t in $esperadas; do printf '%s.jsonl ' "$t"; done)
[ "$hay" = "$want" ] || { echo "DIR_DEV debe traer exactamente los 6 jsonl de dev; trae: $hay" >&2; exit 1; }
# las predicciones de T3 son las del criterio
git show "$sha:model/alt/crit-r10-t3b.json" | python3 -I -c "
import hashlib, json, sys
c = json.load(sys.stdin)
mal = [f for f, s in c['entradas_t3_sha256'].items() if hashlib.sha256(open(sys.argv[1] + '/' + f, 'rb').read()).hexdigest() != s]
sys.exit('predicciones de T3 distintas de las del criterio: %s' % mal if mal else 0)" "$t3"
export COPYFILE_DISABLE=1
git archive --format=tar.gz "$sha" -- model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py \
  model/l2/largos.py model/l2/modelo.py model/l2/evaluar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py sdk/python \
  ':(exclude)model/alt/resultados*' > "$sal/codigo.tgz"
rm -f "$sal/dev.tgz" "$sal/t3.tgz"
( cd "$dev" && { tar --no-xattrs -czf "$sal/dev.tgz" $want 2>/dev/null || tar -czf "$sal/dev.tgz" $want; } )
t3f=$(cd "$t3" && for s in 4b-zs 4b-fs8; do find "$s" -name '*.jsonl' -size +0 -not -name '._*'; done | sort | tr '\n' ' ')
( cd "$t3" && { tar --no-xattrs -czf "$sal/t3.tgz" $t3f 2>/dev/null || tar -czf "$sal/t3.tgz" $t3f; } )
for x in codigo dev t3; do tar -tzf "$sal/$x.tgz" | grep -q '\._' && { echo "$x.tgz trae ._*" >&2; exit 1; }; done
tar -tzf "$sal/dev.tgz" | grep -qi 'sellad' && { echo "dev.tgz trae algo sellado" >&2; exit 1; }
crit=$(git show "$sha:model/alt/crit-r10-t3b.json" | shasum -a 256 | cut -d' ' -f1)
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
ud=model/alt/aws/lanzados/ud-r10-t3b-$sha.sh
mkdir -p model/alt/aws/lanzados
git show "$sha:model/alt/aws/r10_t3b.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-879381269751/r10-t3b-$sha#" -e "s#__CRIT_SHA__#$crit#" -e "s#__N_TEST__#$nt#" -e "s#__N_SOP__#$ns#" > "$ud"
bash -n "$ud"
grep -n '__[A-Z0-9_]*__' "$ud" && { echo "quedan placeholders sin sustituir" >&2; exit 1; }
[ "$(wc -c < "$ud")" -le 16384 ] || { echo "el ud pasa de 16 KB" >&2; exit 1; }
wc -c "$ud"; (cd "$sal" && shasum -a 256 codigo.tgz dev.tgz t3.tgz); echo "criterio sha256 $crit; test=$nt soporte=$ns"
echo "sube a s3://chispa-pruebas-879381269751/r10-t3b-$sha/: codigo.tgz, dev.tgz y t3.tgz; lanza $ud"
