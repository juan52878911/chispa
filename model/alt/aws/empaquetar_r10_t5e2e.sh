#!/usr/bin/env bash
# R10-t5e2e: empaqueta lo que lee r10_t5e2e.sh y escribe su user-data. No sube nada: se suben a mano, por nombre exacto, a
# s3://chispa-pruebas-cuenta-aws/r10-t5e2e-SHA/ los cuatro ficheros de DIR_SALIDA.
#   codigo.tgz       server/, runtime/{l1,tok,torre}, format/chispa3, model/alt/aws/{r10_t5_cliente.py,r10_cliente.py} y gomod-proxy/ (kindling y
#                    onnxruntime_go de la caché de módulos local); de `git archive SHA`, sin ._* ni xattrs. Comprueba que el servidor compila solo con gomod-proxy/.
#   bancos.tgz       banco-<tarea>.npy/.jsonl, referencia.json y consultas.npy (vectores de Python, para la paridad) del fixture de banco-paridad.
#   consultas.jsonl  el consultas.jsonl del fixture con texto, pregunta y tipo (model/alt/consultas_con_texto.py), 4 244 filas.
#   e5.onnx          el e5 fp32 (model/alt/exportar_e5_onnx.py), con su sha256 en el ud.
#   model/alt/aws/lanzados/ud-r10-t5e2e-SHA.sh   r10_t5e2e.sh de SHA con __CHISPA_S3__, __N_FILAS__, __GO_SHA256__ y __E5_SHA256__ sustituidos.
# Las entradas viven en el i7 (rutas por defecto de abajo). Desde el Mac, CHISPA_REMOTO=maquina-i7 las copia por ssh/scp a un directorio temporal; en el
# propio i7 (o con el NFS montado) se leen directamente. Se pueden cambiar con CHISPA_FIXTURE, CHISPA_ONNX y CHISPA_BATS (lista separada por comas).
# Uso: GO_SHA256=<sha256 de go1.27.1.linux-amd64.tar.gz> bash model/alt/aws/empaquetar_r10_t5e2e.sh SHA DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; sal=${2:?DIR_SALIDA}
[ -n "${GO_SHA256:-}" ] || { echo "falta GO_SHA256 (sha256 de go1.27.1.linux-amd64.tar.gz): nada se descarga sin sha256" >&2; exit 1; }
[[ $GO_SHA256 =~ ^[0-9a-f]{64}$ ]] || { echo "GO_SHA256 no es un sha256 en hex" >&2; exit 1; }
for f in r10_t5e2e.sh r10_t5_cliente.py r10_cliente.py consultas_con_texto.py; do
  d=model/alt/aws; [ "$f" = consultas_con_texto.py ] && d=model/alt
  git cat-file -e "$sha:$d/$f" || { echo "$sha no trae $d/$f" >&2; exit 1; }
done
NFS=/mnt/nfs-lab/chispa-audio
FIXTURE=${CHISPA_FIXTURE:-$NFS/escalado/banco-paridad}
ONNX=${CHISPA_ONNX:-$NFS/escalado/e5-onnx/e5.onnx}
BATS=${CHISPA_BATS:-$NFS/p3-eval/full/bat,$NFS/tareas-nuevas/bat}
mkdir -p "$sal"
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
# --- entradas (local o por ssh)
if [ -n "${CHISPA_REMOTO:-}" ]; then
  mkdir -p "$t/fix"; scp -q -r "$CHISPA_REMOTO:$FIXTURE/." "$t/fix/"
  bats=""; i=0; for b in ${BATS//,/ }; do i=$((i+1)); mkdir -p "$t/bat$i"; scp -q "$CHISPA_REMOTO:$b/*.jsonl" "$t/bat$i/"; bats="${bats:+$bats,}$t/bat$i"; done
  scp -q "$CHISPA_REMOTO:$ONNX" "$t/e5.onnx"
  FIXTURE=$t/fix; BATS=$bats; ONNX=$t/e5.onnx
fi
[ -f "$FIXTURE/consultas.jsonl" ] && [ -f "$FIXTURE/referencia.json" ] && [ -f "$ONNX" ] || { echo "faltan consultas.jsonl, referencia.json o e5.onnx" >&2; exit 1; }
# --- código de SHA
git archive "$sha" server runtime/l1 runtime/tok runtime/torre format/chispa3 model/alt/aws/r10_t5_cliente.py model/alt/aws/r10_cliente.py model/alt/consultas_con_texto.py | tar -x -C "$t"
rm -rf "$t/server/deploy/resultados" "$t/server/deploy/aws/resultados"
mod=$(go env GOMODCACHE)/cache/download; pq=$t/gomod-proxy
copiar() { local m=$1 v=$2 d=$pq/$1/@v s=$mod/$1/@v; mkdir -p "$d"; for e in info mod zip ziphash; do cp "$s/$v.$e" "$d/$v.$e"; done; echo "$v" > "$d/list"; }
kv=$(sed -n 's#^require github.com/juan52878911/kindling \(v[^ ]*\)#\1#p; s#^[[:space:]]*github.com/juan52878911/kindling \(v[^ ]*\)#\1#p' "$t/server/go.mod" | head -1)
ov=$(sed -n 's#^require github.com/yalue/onnxruntime_go \(v[^ ]*\)#\1#p; s#^[[:space:]]*github.com/yalue/onnxruntime_go \(v[^ ]*\)#\1#p' "$t/server/go.mod" | head -1)
[ -n "$kv" ] && [ -n "$ov" ] || { echo "no se pudo leer las versiones de server/go.mod" >&2; exit 1; }
copiar github.com/juan52878911/kindling "$kv"; copiar github.com/yalue/onnxruntime_go "$ov"
( cd "$t/server" && GOMODCACHE=$t/.mc GOPATH=$t/.gp GOFLAGS=-mod=readonly GOSUMDB=off GOPROXY=file://$pq GOTOOLCHAIN=local go build -tags ort -o /dev/null ./cmd/chispa-server ) \
  || { echo "el servidor no compila (-tags ort) solo con gomod-proxy/" >&2; exit 1; }
chmod -R u+w "$t/.mc" "$t/.gp" 2>/dev/null || true; rm -rf "$t/.mc" "$t/.gp"
find "$t" -name '._*' -delete
tarz() { COPYFILE_DISABLE=1 tar --no-xattrs -czf "$1" -C "$2" "${@:3}" 2>/dev/null || COPYFILE_DISABLE=1 tar -czf "$1" -C "$2" "${@:3}"; }
rm -f "$sal/codigo.tgz" "$sal/bancos.tgz"
tarz "$sal/codigo.tgz" "$t" server runtime format model gomod-proxy
# --- bancos, consultas con texto y onnx
mkdir -p "$t/bancos"; cp "$FIXTURE"/banco-*.npy "$FIXTURE"/banco-*.jsonl "$FIXTURE/referencia.json" "$FIXTURE/consultas.npy" "$t/bancos/"
tarz "$sal/bancos.tgz" "$t/bancos" .
python3 "$t/model/alt/consultas_con_texto.py" --fixture "$FIXTURE" --bat "$BATS" --salida "$sal/consultas.jsonl"
cp "$ONNX" "$sal/e5.onnx"
for f in codigo.tgz bancos.tgz; do tar -tzf "$sal/$f" | grep -q '\._' && { echo "$f trae ._*" >&2; exit 1; }; done
n=$(wc -l < "$sal/consultas.jsonl" | tr -d ' ')
e5sha=$(shasum -a 256 "$sal/e5.onnx" | cut -d' ' -f1)
ud=model/alt/aws/lanzados/ud-r10-t5e2e-$sha.sh
mkdir -p model/alt/aws/lanzados
git show "$sha:model/alt/aws/r10_t5e2e.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/r10-t5e2e-$sha#" -e "s#__N_FILAS__#$n#" -e "s#__GO_SHA256__#$GO_SHA256#" -e "s#__E5_SHA256__#$e5sha#" > "$ud"
bash -n "$ud"
grep -n '__[A-Z0-9_]*__' "$ud" | grep -v 'GO_SHA256#__' && { echo "quedan placeholders sin sustituir" >&2; exit 1; }
(cd "$sal" && ls -l codigo.tgz bancos.tgz consultas.jsonl e5.onnx && shasum -a 256 codigo.tgz bancos.tgz consultas.jsonl e5.onnx)
echo "ud: $ud ($(wc -c < "$ud") bytes, sha256 $(shasum -a 256 "$ud" | cut -d' ' -f1))"
echo "sube a s3://chispa-pruebas-cuenta-aws/r10-t5e2e-$sha/: codigo.tgz bancos.tgz consultas.jsonl e5.onnx; lanza $ud"
