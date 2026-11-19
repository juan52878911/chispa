#!/usr/bin/env bash
# R11: empaqueta lo que lee r11_alternativas.sh y escribe su user-data. No sube nada.
#   DIR_SALIDA/codigo.tgz  el código de SHA (lo commiteado; el mismo conjunto que T3a), sin ._*
#   DIR_SALIDA/dev.tgz     SOLO los 6 jsonl de DESARROLLO de DIR_DEV (bateria-v2/dev); si DIR_DEV trae algo más, falla: las selladas no se copian
#   DIR_SALIDA/ref.tgz     las referencias ya medidas (solo id y probabilidades, sin textos): 4b-zs/ (T3), zs-topk/finanzas (T3b), l2-zs/ y knn-e5-k8-t3a/ (T3a)
#   model/alt/aws/lanzados/ud-r11-alternativas-SHA.sh  r11_alternativas.sh de SHA con __CHISPA_S3__=s3://chispa-pruebas-cuenta-aws/r11-alternativas-SHA,
#                          el sha256 de crit-r11.json de SHA y los recuentos.
# Uso: bash model/alt/aws/empaquetar_r11_alternativas.sh SHA DIR_DEV PRED_T3 PRED_T3A PRED_T3B DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; dev=${2:?DIR_DEV}; t3=${3:?PRED_T3}; t3a=${4:?PRED_T3A}; t3b=${5:?PRED_T3B}; sal=${6:?DIR_SALIDA}
dev=$(cd "$dev" && pwd); mkdir -p "$sal"; sal=$(cd "$sal" && pwd)
git cat-file -e "$sha:model/alt/aws/r11_alternativas.sh" && git cat-file -e "$sha:model/alt/crit-r11.json" || { echo "$sha no trae r11_alternativas.sh y crit-r11.json" >&2; exit 1; }
esperadas="cfpb-quejas congreso-eeuu-partido finanzas-documentos horeca-resenas jql-valor-educativo usajobs-nivel"
hay=$(cd "$dev" && ls | grep -v '^\._' | tr '\n' ' ')
want=$(for t in $esperadas; do printf '%s.jsonl ' "$t"; done)
[ "$hay" = "$want" ] || { echo "DIR_DEV debe traer exactamente los 6 jsonl de dev; trae: $hay" >&2; exit 1; }
export COPYFILE_DISABLE=1
git archive --format=tar.gz "$sha" -- model/__init__.py model/alt model/datos model/meta model/evaluacion model/l2/__init__.py model/l2/datos.py \
  model/l2/largos.py model/l2/modelo.py model/l2/evaluar.py model/l2/linea_base_knn_e5.py model/maestro/__init__.py model/maestro/blandas.py sdk/python \
  ':(exclude)model/alt/resultados*' > "$sal/codigo.tgz"
rm -f "$sal/dev.tgz" "$sal/ref.tgz"
( cd "$dev" && { tar --no-xattrs -czf "$sal/dev.tgz" $want 2>/dev/null || tar -czf "$sal/dev.tgz" $want; } )
r=$(mktemp -d); mkdir -p "$r/4b-zs" "$r/zs-topk" "$r/l2-zs" "$r/knn-e5-k8-t3a"
for t in $esperadas; do
  cp "$t3/4b-zs/$t.jsonl" "$r/4b-zs/"; cp "$t3a/l2-zs/$t.jsonl" "$r/l2-zs/"; cp "$t3a/knn-e5-k8/$t.jsonl" "$r/knn-e5-k8-t3a/"
done
cp "$t3b/zs-topk/finanzas-documentos.jsonl" "$r/zs-topk/"
python3 -I - "$r" "$dev" <<'PY'
import glob, json, sys
textos = set()
for f in glob.glob(sys.argv[2] + "/*.jsonl"):
    for l in open(f, encoding="utf-8"):
        textos.add(json.loads(l)["texto"][:200])
for f in glob.glob(sys.argv[1] + "/*/*.jsonl"):
    for l in open(f, encoding="utf-8"):
        x = json.loads(l)
        assert set(x) <= {"id", "probs", "valida", "raw", "ms", "error", "intentos"}, (f, sorted(x))
        assert ":test:" in x["id"], (f, x["id"])
        assert not any(s and s in l for s in textos if len(s) > 40), f   # las referencias no traen textos
PY
( cd "$r" && { tar --no-xattrs -czf "$sal/ref.tgz" 4b-zs zs-topk l2-zs knn-e5-k8-t3a 2>/dev/null || tar -czf "$sal/ref.tgz" 4b-zs zs-topk l2-zs knn-e5-k8-t3a; } )
nref=$(cat "$r"/*/*.jsonl | wc -l | tr -d ' '); rm -rf "$r"
for x in codigo dev ref; do tar -tzf "$sal/$x.tgz" | grep -q '\._' && { echo "$x.tgz trae ._*" >&2; exit 1; }; done
tar -tzf "$sal/dev.tgz" | grep -qi 'sellad' && { echo "dev.tgz trae algo sellado" >&2; exit 1; }
crit=$(git show "$sha:model/alt/crit-r11.json" | shasum -a 256 | cut -d' ' -f1)
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
ud=model/alt/aws/lanzados/ud-r11-alternativas-$sha.sh
mkdir -p model/alt/aws/lanzados
git show "$sha:model/alt/aws/r11_alternativas.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/r11-alternativas-$sha#" -e "s#__CRIT_SHA__#$crit#" \
  -e "s#__N_TEST__#$nt#" -e "s#__N_SOP__#$ns#" -e "s#__N_REF__#$nref#" > "$ud"
bash -n "$ud"
grep -n '__[A-Z0-9_]*__' "$ud" && { echo "quedan placeholders sin sustituir" >&2; exit 1; }
[ "$(wc -c < "$ud")" -le 16384 ] || { echo "el ud pasa de 16 KB" >&2; exit 1; }
wc -c "$ud"; shasum -a 256 "$ud"; (cd "$sal" && shasum -a 256 codigo.tgz dev.tgz ref.tgz); echo "criterio sha256 $crit; test=$nt soporte=$ns ref=$nref"
echo "sube a s3://chispa-pruebas-cuenta-aws/r11-alternativas-$sha/: codigo.tgz, dev.tgz y ref.tgz; lanza $ud"
