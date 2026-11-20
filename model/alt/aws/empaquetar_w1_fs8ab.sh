#!/usr/bin/env bash
# W1c: empaqueta las DOS fases (A en c7i.xlarge, B en g5.xlarge) y escribe sus user-data. NO sube ni lanza nada. Un solo codigo.tgz y un solo dev.tgz (los mismos sha256 en los dos ud):
# se suben a mano, por nombre exacto, a s3://chispa-pruebas-879381269751/w1-fs8a-SHA/ Y a w1-fs8b-SHA/ (A lee y escribe en el suyo; B lee lo de A en w1-fs8a-SHA/{faseA.tar,faseA.json} y escribe en el suyo).
#   codigo.tgz   de `git archive SHA`: server/, runtime/, format/ y de model/alt solo lo que usan las fases, más gomod-proxy/ (kindling y onnxruntime_go de la caché de módulos local). Comprueba que el servidor
#                y la herramienta de bancos compilan (-tags ort) solo con gomod-proxy/.
#   dev.tgz      SOLO los 6 jsonl de DESARROLLO de DIR_DEV (bateria-v2/dev); si trae algo más, falla: las selladas no se copian.
#   model/alt/aws/lanzados/ud-w1-fs8a-SHA.sh (c7i.xlarge, tope 60 min) y ud-w1-fs8b-SHA.sh (g5.xlarge, tope 75 min), cada uno < 15 500 bytes (user-data de EC2: 16 384).
# El criterio (model/alt/crit-w1-fs8-e2e.json) lo congela el orquestador: CRIT_SHA256 es su sha256 y debe ser el del fichero en SHA, y el JSON debe decir congelado_antes_de_medir: true.
# FASEA_SHA256 (opcional): sha256 de faseA.tar si ya se conoce (A ya corrió); se fija en el ud de B. Si falta, B se fía del faseA.json de A (que trae el sha256 del tar, el de cada fichero y los de codigo/dev/criterio).
# CHISPA_UD_DIR cambia el directorio de los ud (pruebas en seco con un dev inventado: no deja nada en lanzados/).
# Uso: GO_SHA256=<sha256 de go1.27.1.linux-amd64.tar.gz> CRIT_SHA256=<sha256 del criterio congelado> [FASEA_SHA256=...] bash model/alt/aws/empaquetar_w1_fs8ab.sh SHA DIR_DEV DIR_SALIDA
# (CHISPA_PERMITIR_CRIT_SIN_CONGELAR=1 solo en la prueba en seco: acepta el criterio propuesto sin congelar)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; dev=${2:?DIR_DEV}; sal=${3:?DIR_SALIDA}
[ -n "${GO_SHA256:-}" ] || { echo "falta GO_SHA256 (sha256 de go1.27.1.linux-amd64.tar.gz): nada se descarga sin sha256" >&2; exit 1; }
[[ $GO_SHA256 =~ ^[0-9a-f]{64}$ ]] || { echo "GO_SHA256 no es un sha256 en hex" >&2; exit 1; }
[ -n "${CRIT_SHA256:-}" ] || { echo "falta CRIT_SHA256 (sha256 del criterio congelado por el orquestador)" >&2; exit 1; }
[[ $CRIT_SHA256 =~ ^[0-9a-f]{64}$ ]] || { echo "CRIT_SHA256 no es un sha256 en hex" >&2; exit 1; }
dev=$(cd "$dev" && pwd); mkdir -p "$sal"; sal=$(cd "$sal" && pwd)
MODELALT="model/alt/__init__.py model/alt/crit-w1-fs8-e2e.json model/alt/exportar_qwen3emb_onnx.py model/alt/llm_logprobs.py model/alt/llm_jevbench.py model/alt/lineas_base_v2.py
  model/alt/fusion_knn_l2.py model/alt/w1_filas.py model/alt/w1_medir.py model/alt/w1_demos.py model/alt/lineas_base_v2_pred.py model/alt/w1_banco_falso.py model/alt/aws/r10_t5_cliente.py model/alt/aws/r10_cliente.py model/alt/aws/w1_fs8a.sh model/alt/aws/w1_fs8a_fases.sh model/alt/aws/w1_fs8b.sh model/alt/aws/w1_fs8b_fases.sh"
for f in $MODELALT model/__init__.py runtime/tok/exportar_bpe.py server/cmd/chispa-banco-qwen/main.go server/deploy/config.modo-gpu.json; do
  git cat-file -e "$sha:$f" || { echo "$sha no trae $f" >&2; exit 1; }
done
crit=$(git show "$sha:model/alt/crit-w1-fs8-e2e.json" | shasum -a 256 | cut -d' ' -f1)
[ "$crit" = "$CRIT_SHA256" ] || { echo "el criterio de $sha tiene sha256 $crit y no el congelado ($CRIT_SHA256)" >&2; exit 1; }
if [ "${CHISPA_PERMITIR_CRIT_SIN_CONGELAR:-0}" != 1 ]; then
  git show "$sha:model/alt/crit-w1-fs8-e2e.json" | python3 -I -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("congelado_antes_de_medir") is True else 1)' \
    || { echo "el criterio de $sha no está congelado (congelado_antes_de_medir: true)" >&2; exit 1; }
fi
npt=$(git show "$sha:model/alt/crit-w1-fs8-e2e.json" | python3 -I -c 'import json,sys; print(int(json.load(sys.stdin)["datos"]["n_por_tarea"]))')
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
    sh -c 'go build -tags ort -o /dev/null ./cmd/chispa-server && go build -tags ort -o /dev/null ./cmd/chispa-banco-qwen' ) \
  || { echo "el servidor o la herramienta de bancos no compilan (-tags ort) solo con gomod-proxy/" >&2; exit 1; }
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
udd=${CHISPA_UD_DIR:-model/alt/aws/lanzados}; mkdir -p "$udd"
fas=${FASEA_SHA256:--}
[ "$fas" = - ] || [[ $fas =~ ^[0-9a-f]{64}$ ]] || { echo "FASEA_SHA256 no es un sha256 en hex" >&2; exit 1; }
s3a=s3://chispa-pruebas-879381269751/w1-fs8a-$sha
for f in a b; do
  ud=$udd/ud-w1-fs8$f-$sha.sh
  git show "$sha:model/alt/aws/w1_fs8$f.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-879381269751/w1-fs8$f-$sha#" -e "s#__CHISPA_S3_A__#$s3a#" -e "s#__CRIT_SHA__#$crit#" -e "s#__COD_SHA256__#$(shf "$sal/codigo.tgz")#" \
    -e "s#__DEV_SHA256__#$(shf "$sal/dev.tgz")#" -e "s#__FASEA_SHA256__#$fas#" -e "s#__N_TEST__#$nt#" -e "s#__N_SOP__#$ns#" -e "s#__N_POR_TAREA__#$npt#" -e "s#__GO_SHA256__#$GO_SHA256#" > "$ud"
  bash -n "$ud"
  grep -n '__[A-Z0-9_]*__' "$ud" | grep -v 'S3_LOCAL\|S3A_LOCAL' && { echo "quedan placeholders sin sustituir en $ud" >&2; exit 1; }
  [ "$(wc -c < "$ud")" -lt 15500 ] || { echo "$ud pesa $(wc -c < "$ud") bytes: el user-data de EC2 admite 16384 como máximo (objetivo < 15500)" >&2; exit 1; }
  bash -n <(git show "$sha:model/alt/aws/w1_fs8${f}_fases.sh")
  echo "ud $f: $ud ($(wc -c < "$ud") bytes, sha256 $(shf "$ud"))"
done
(cd "$sal" && ls -l codigo.tgz dev.tgz && shasum -a 256 codigo.tgz dev.tgz)
echo "criterio sha256 $crit; dev: test=$nt soporte=$ns; n por tarea $npt"
echo "sube codigo.tgz y dev.tgz a $s3a/ y a s3://chispa-pruebas-879381269751/w1-fs8b-$sha/ ; A (c7i.xlarge, volumen >= 60 GB, tope 60 min = 0,18 USD) escribe en $s3a/ faseA.tar y faseA.json; B (g5.xlarge, volumen >= 100 GB, tope 75 min = 1,26 USD) los lee de ahí"
