#!/usr/bin/env bash
# R10 piloto: empaqueta lo que lee r10_piloto.sh y escribe su user-data. No sube nada.
#   DIR_SALIDA/codigo.tgz  `git archive SHA` de model/alt (sin resultados* ni lanzados), model/{datos,meta,evaluacion,maestro,l2/...} y el vocabulario podado; sin ._*.
#   DIR_SALIDA/dev.tgz     las 6 tareas DEV de la batería v2 de DIR_DEV (*.jsonl: nunca las selladas), en la raíz del tarball, sin ._* ni xattrs.
#   model/alt/aws/lanzados/ud-r10-piloto-SHA.sh   r10_piloto.sh de SHA con __CHISPA_S3__=s3://chispa-pruebas-cuenta-aws/r10-piloto-SHA y
#                          __CRIT_SHA__ = sha256 de model/alt/crit-r10-piloto.json en SHA. Se comprueba: bash -n, < 16 KB, sin placeholders.
# Sube codigo.tgz y dev.tgz a s3://chispa-pruebas-cuenta-aws/r10-piloto-SHA/ y lanza el ud en una g5.xlarge bajo demanda (DLAMI GPU PyTorch AL2023),
# con --instance-initiated-shutdown-behavior terminate. Uso: bash model/alt/aws/empaquetar_r10_piloto.sh SHA DIR_DEV DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; dev=${2:?DIR_DEV}; sal=${3:?DIR_SALIDA}
git cat-file -e "$sha:model/alt/aws/r10_piloto.sh" || { echo "$sha no trae model/alt/aws/r10_piloto.sh" >&2; exit 1; }
git cat-file -e "$sha:model/alt/crit-r10-piloto.json" || { echo "$sha no trae el criterio" >&2; exit 1; }
# las 6 DEV y solo ellas: nombres del manifiesto de la batería v2 con particion=dev
esperadas=$(git show "$sha:model/alt/bateria-v2-manifiesto.json" | python3 -c 'import json,sys; print("\n".join(sorted(t["nombre"]+".jsonl" for t in json.load(sys.stdin)["tareas"] if t["particion"]=="dev")))')
hay=$(cd "$dev" && ls -1 *.jsonl 2>/dev/null | LC_ALL=C sort)
[ "$hay" = "$(echo "$esperadas" | LC_ALL=C sort)" ] || { echo "DIR_DEV debe traer exactamente las 6 DEV:"; echo "$esperadas"; echo "y trae:"; echo "$hay"; exit 1; } >&2
case "$dev" in *sellada*) echo "DIR_DEV parece de selladas" >&2; exit 1;; esac
mkdir -p "$sal"
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
git archive "$sha" -- model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py model/l2/largos.py \
  model/l2/modelo.py model/l2/evaluar.py model/l2/exportar.py model/l2/calibrar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py \
  model/bench/vocab-podado-e5-small.json ':(exclude)model/alt/resultados*' ':(exclude)model/alt/aws/lanzados' | tar -x -C "$t"
find "$t" -name '._*' -delete
rm -f "$sal/codigo.tgz" "$sal/dev.tgz"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$sal/codigo.tgz" -C "$t" . 2>/dev/null || COPYFILE_DISABLE=1 tar -czf "$sal/codigo.tgz" -C "$t" .
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$sal/dev.tgz" -C "$dev" $(cd "$dev" && ls -1 *.jsonl) 2>/dev/null || COPYFILE_DISABLE=1 tar -czf "$sal/dev.tgz" -C "$dev" $(cd "$dev" && ls -1 *.jsonl)
for f in codigo dev; do tar -tzf "$sal/$f.tgz" | grep -q '\._' && { echo "$f.tgz trae ._*" >&2; exit 1; }; done
crit=$(git show "$sha:model/alt/crit-r10-piloto.json" | shasum -a 256 | cut -d' ' -f1)
# el criterio del tarball es el mismo que el de SHA
[ "$(shasum -a 256 < "$t/model/alt/crit-r10-piloto.json" | cut -d' ' -f1)" = "$crit" ] || { echo "el criterio del tarball no coincide" >&2; exit 1; }
ud=model/alt/aws/lanzados/ud-r10-piloto-$sha.sh
mkdir -p model/alt/aws/lanzados
git show "$sha:model/alt/aws/r10_piloto.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/r10-piloto-$sha#" -e "s#__CRIT_SHA__#$crit#" > "$ud"
bash -n "$ud"
grep -n '__[A-Z0-9_]*__' "$ud" && { echo "quedan placeholders sin sustituir" >&2; exit 1; }
n=$(wc -c < "$ud" | tr -d ' ')
[ "$n" -le 16384 ] || { echo "el ud pesa $n bytes (> 16 KB)" >&2; exit 1; }
echo "ud: $ud ($n bytes); criterio sha256 $crit"
(cd "$sal" && shasum -a 256 codigo.tgz dev.tgz && ls -l codigo.tgz dev.tgz)
echo "sube codigo.tgz y dev.tgz a s3://chispa-pruebas-cuenta-aws/r10-piloto-$sha/ y lanza $ud (g5.xlarge bajo demanda, terminate al apagar)"
