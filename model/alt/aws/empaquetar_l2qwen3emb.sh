#!/usr/bin/env bash
# L2-qwen3emb: empaqueta lo que lee l2qwen3emb.sh y escribe su user-data. NO sube ni lanza nada: se suben a mano, por nombre exacto, a
# s3://chispa-pruebas-cuenta-aws/l2-qwen3emb-SHA/ los dos tgz de DIR_SALIDA.
#   codigo.tgz   de `git archive SHA`: server/, runtime/ (l1, tok con el fixture y el tokenizador, torre), format/, y de model/alt solo lo que usa el ud (exportador, referencia,
#                medida, filas, cliente, fases y el criterio), más gomod-proxy/ (kindling y onnxruntime_go de la caché de módulos local). Comprueba que el servidor y la herramienta
#                de bancos compilan (-tags ort) solo con gomod-proxy/.
#   dev.tgz      SOLO los 6 jsonl de DESARROLLO de DIR_DEV (bateria-v2/dev); si trae algo más, falla: las selladas no se copian.
#   model/alt/aws/lanzados/ud-l2-qwen3emb-SHA.sh   l2qwen3emb.sh de SHA con __CHISPA_S3__ y los sha256 del criterio, de los dos tgz y de Go, y los recuentos de dev
#                (el ud, que debe caber en los 16 384 bytes de user-data de EC2, solo baja y comprueba; las fases viven en codigo.tgz: l2qwen3emb_fases.sh).
# CHISPA_UD_DIR cambia el directorio del ud (pruebas en seco con un dev inventado: no deja nada en lanzados/).
# Uso: GO_SHA256=<sha256 de go1.27.1.linux-amd64.tar.gz> bash model/alt/aws/empaquetar_l2qwen3emb.sh SHA DIR_DEV DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; dev=${2:?DIR_DEV}; sal=${3:?DIR_SALIDA}
[ -n "${GO_SHA256:-}" ] || { echo "falta GO_SHA256 (sha256 de go1.27.1.linux-amd64.tar.gz): nada se descarga sin sha256" >&2; exit 1; }
[[ $GO_SHA256 =~ ^[0-9a-f]{64}$ ]] || { echo "GO_SHA256 no es un sha256 en hex" >&2; exit 1; }
dev=$(cd "$dev" && pwd); mkdir -p "$sal"; sal=$(cd "$sal" && pwd)
MODELALT="model/alt/__init__.py model/alt/crit-l2-qwen3emb.json model/alt/exportar_qwen3emb_onnx.py model/alt/qwen3emb_referencia.py model/alt/qwen3emb_medir.py
  model/alt/lineas_base_v2.py model/alt/fusion_knn_l2.py model/alt/aws/qwen3emb_filas.py model/alt/aws/r10_t5_cliente.py model/alt/aws/r10_cliente.py
  model/alt/aws/l2qwen3emb.sh model/alt/aws/l2qwen3emb_fases.sh"
for f in $MODELALT model/__init__.py runtime/tok/exportar_bpe.py runtime/tok/testdata/qwen/fixture_paridad.jsonl server/cmd/chispa-banco-qwen/main.go server/decide/qwen_paridad_real_test.go; do
  git cat-file -e "$sha:$f" || { echo "$sha no trae $f" >&2; exit 1; }
done
esperadas="cfpb-quejas congreso-eeuu-partido finanzas-documentos horeca-resenas jql-valor-educativo usajobs-nivel"
hay=$(cd "$dev" && ls | grep -v '^\._' | tr '\n' ' ')
want=$(for t in $esperadas; do printf '%s.jsonl ' "$t"; done)
[ "$hay" = "$want" ] || { echo "DIR_DEV debe traer exactamente los 6 jsonl de dev; trae: $hay" >&2; exit 1; }
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
git archive "$sha" server runtime format model/__init__.py $MODELALT | tar -x -C "$t"
rm -rf "$t/server/deploy/resultados" "$t/server/deploy/aws/resultados"
mod=$(go env GOMODCACHE)/cache/download; pq=$t/gomod-proxy
copiar() { local m=$1 v=$2 d=$pq/$1/@v s=$mod/$1/@v; mkdir -p "$d"; for e in info mod zip ziphash; do cp "$s/$v.$e" "$d/$v.$e"; done; echo "$v" > "$d/list"; }
kv=$(sed -n 's#^require github.com/juan52878911/kindling \(v[^ ]*\)#\1#p; s#^[[:space:]]*github.com/juan52878911/kindling \(v[^ ]*\)#\1#p' "$t/server/go.mod" | head -1)
ov=$(sed -n 's#^require github.com/yalue/onnxruntime_go \(v[^ ]*\)#\1#p; s#^[[:space:]]*github.com/yalue/onnxruntime_go \(v[^ ]*\)#\1#p' "$t/server/go.mod" | head -1)
[ -n "$kv" ] && [ -n "$ov" ] || { echo "no se pudo leer las versiones de server/go.mod" >&2; exit 1; }
copiar github.com/juan52878911/kindling "$kv"; copiar github.com/yalue/onnxruntime_go "$ov"
( cd "$t/server" && GOMODCACHE=$t/.mc GOPATH=$t/.gp GOFLAGS=-mod=readonly GOSUMDB=off GOPROXY=file://$pq GOTOOLCHAIN=local \
    sh -c 'go build -tags ort -o /dev/null ./cmd/chispa-server && go build -tags ort -o /dev/null ./cmd/chispa-banco-qwen && go test -tags ort -c -o /dev/null ./decide' ) \
  || { echo "el servidor, la herramienta de bancos o el test de paridad no compilan (-tags ort) solo con gomod-proxy/" >&2; exit 1; }
chmod -R u+w "$t/.mc" "$t/.gp" 2>/dev/null || true; rm -rf "$t/.mc" "$t/.gp"
find "$t" -name '._*' -delete
export COPYFILE_DISABLE=1
tarz() { tar --no-xattrs -czf "$1" -C "$2" "${@:3}" 2>/dev/null || tar -czf "$1" -C "$2" "${@:3}"; }
rm -f "$sal/codigo.tgz" "$sal/dev.tgz"
tarz "$sal/codigo.tgz" "$t" server runtime format model gomod-proxy
( cd "$dev" && { tar --no-xattrs -czf "$sal/dev.tgz" $want 2>/dev/null || tar -czf "$sal/dev.tgz" $want; } )
for x in codigo dev; do tar -tzf "$sal/$x.tgz" | grep -q '\._' && { echo "$x.tgz trae ._*" >&2; exit 1; }; done
tar -tzf "$sal/dev.tgz" | grep -qi 'sellad' && { echo "dev.tgz trae algo sellado" >&2; exit 1; }
[ "$(tar -tzf "$sal/dev.tgz" | wc -l | tr -d ' ')" = 6 ] || { echo "dev.tgz no trae 6 ficheros" >&2; exit 1; }
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
shf() { shasum -a 256 "$1" | cut -d' ' -f1; }
crit=$(git show "$sha:model/alt/crit-l2-qwen3emb.json" | shasum -a 256 | cut -d' ' -f1)
[ "$crit" = 58a62d618a4f917c1c4c232934a31394874a59f4cb5b589e945efd91511ade0b ] || { echo "el criterio de $sha no es el congelado ($crit)" >&2; exit 1; }
udd=${CHISPA_UD_DIR:-model/alt/aws/lanzados}; mkdir -p "$udd"
ud=$udd/ud-l2-qwen3emb-$sha.sh
git show "$sha:model/alt/aws/l2qwen3emb.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/l2-qwen3emb-$sha#" -e "s#__CRIT_SHA__#$crit#" -e "s#__COD_SHA256__#$(shf "$sal/codigo.tgz")#" \
  -e "s#__DEV_SHA256__#$(shf "$sal/dev.tgz")#" -e "s#__N_TEST__#$nt#" -e "s#__N_SOP__#$ns#" -e "s#__GO_SHA256__#$GO_SHA256#" > "$ud"
bash -n "$ud"
grep -n '__[A-Z0-9_]*__' "$ud" && { echo "quedan placeholders sin sustituir" >&2; exit 1; }
[ "$(wc -c < "$ud")" -lt 15500 ] || { echo "el ud pesa $(wc -c < "$ud") bytes: el user-data de EC2 admite 16384 como máximo (objetivo < 15500)" >&2; exit 1; }
bash -n <(git show "$sha:model/alt/aws/l2qwen3emb_fases.sh")
(cd "$sal" && ls -l codigo.tgz dev.tgz && shasum -a 256 codigo.tgz dev.tgz)
echo "ud: $ud ($(wc -c < "$ud") bytes, sha256 $(shf "$ud")); criterio sha256 $crit; dev: test=$nt soporte=$ns"
echo "sube a s3://chispa-pruebas-cuenta-aws/l2-qwen3emb-$sha/: codigo.tgz y dev.tgz; lanza $ud en una c7i.xlarge con volumen >= 40 GB"
