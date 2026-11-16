#!/usr/bin/env bash
# R9-e2e: empaqueta lo que lee r9_e2e.sh y escribe su user-data. No sube nada.
#   DIR_SALIDA/codigo.tgz   server/, runtime/{l1,tok,torre}, format/chispa3 (los replace de server/go.mod), model/alt/aws/r9_cliente.py y
#                           gomod-proxy/ (solo kindling v0.17.0 y onnxruntime_go v1.38.0, de la caché de módulos local); sin ._* ni xattrs.
#                           Todo sale de `git archive SHA`: lo que se lanza es lo commiteado.
#   DIR_SALIDA/filas.jsonl  copia de DIR_MUESTRA/filas.jsonl (la muestra de 3 116 filas); se sube a mano junto a codigo.tgz.
#   model/alt/aws/lanzados/ud-r9-e2e-SHA.sh   r9_e2e.sh de SHA con __CHISPA_S3__=s3://chispa-pruebas-cuenta-aws/r9-e2e-SHA,
#                           __N_FILAS__=nº de líneas de filas.jsonl y __GO_SHA256__=$GO_SHA256.
# Antes de empaquetar comprueba que el servidor compila SOLO con gomod-proxy/ y go.sum (GOPROXY=file://, GOFLAGS=-mod=readonly, sin red).
# Uso: GO_SHA256=<sha256 de go1.27.1.linux-amd64.tar.gz, de https://go.dev/dl/> bash model/alt/aws/empaquetar_r9_e2e.sh SHA DIR_MUESTRA DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; muestra=${2:?DIR_MUESTRA}; sal=${3:?DIR_SALIDA}
[ -n "${GO_SHA256:-}" ] || { echo "falta GO_SHA256 (sha256 de go1.27.1.linux-amd64.tar.gz en https://go.dev/dl/): nada se descarga sin sha256" >&2; exit 1; }
[[ $GO_SHA256 =~ ^[0-9a-f]{64}$ ]] || { echo "GO_SHA256 no es un sha256 en hex" >&2; exit 1; }
[ -f "$muestra/filas.jsonl" ] || { echo "falta $muestra/filas.jsonl" >&2; exit 1; }
git cat-file -e "$sha:model/alt/aws/r9_e2e.sh" || { echo "$sha no trae model/alt/aws/r9_e2e.sh" >&2; exit 1; }
mkdir -p "$sal"
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
git archive "$sha" server runtime/l1 runtime/tok runtime/torre format/chispa3 model/alt/aws/r9_cliente.py | tar -x -C "$t"
rm -rf "$t/server/deploy/resultados" "$t/server/deploy/aws/resultados"
# gomod-proxy: solo los dos módulos externos de server/go.mod, en su versión
mod=$(go env GOMODCACHE)/cache/download
pq=$t/gomod-proxy
copiar() {  # copiar RUTA_MODULO VERSION
  local m=$1 v=$2 d=$pq/$1/@v s=$mod/$1/@v
  mkdir -p "$d"
  for e in info mod zip ziphash; do cp "$s/$v.$e" "$d/$v.$e"; done
  echo "$v" > "$d/list"
}
kv=$(sed -n 's#^require github.com/juan52878911/kindling \(v[^ ]*\)#\1#p; s#^[[:space:]]*github.com/juan52878911/kindling \(v[^ ]*\)#\1#p' "$t/server/go.mod" | head -1)
ov=$(sed -n 's#^require github.com/yalue/onnxruntime_go \(v[^ ]*\)#\1#p; s#^[[:space:]]*github.com/yalue/onnxruntime_go \(v[^ ]*\)#\1#p' "$t/server/go.mod" | head -1)
[ -n "$kv" ] && [ -n "$ov" ] || { echo "no se pudo leer las versiones de server/go.mod" >&2; exit 1; }
copiar github.com/juan52878911/kindling "$kv"; copiar github.com/yalue/onnxruntime_go "$ov"
# comprobación sin red: caché de módulos vacía, solo el proxy empaquetado
( cd "$t/server" && GOMODCACHE=$t/.mc GOPATH=$t/.gp GOFLAGS=-mod=readonly GOSUMDB=off GOPROXY=file://$pq GOTOOLCHAIN=local go build -o /dev/null ./cmd/chispa-server ) \
  || { echo "el servidor no compila solo con gomod-proxy/ (falta algún módulo)" >&2; exit 1; }
chmod -R u+w "$t/.mc" "$t/.gp" 2>/dev/null || true; rm -rf "$t/.mc" "$t/.gp"
find "$t" -name '._*' -delete
rm -f "$sal/codigo.tgz"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$sal/codigo.tgz" -C "$t" . 2>/dev/null || COPYFILE_DISABLE=1 tar -czf "$sal/codigo.tgz" -C "$t" .
tar -tzf "$sal/codigo.tgz" | grep -q '\._' && { echo "codigo.tgz trae ._*" >&2; exit 1; }
cp "$muestra/filas.jsonl" "$sal/filas.jsonl"
n=$(wc -l < "$sal/filas.jsonl" | tr -d ' ')
ud=model/alt/aws/lanzados/ud-r9-e2e-$sha.sh
mkdir -p model/alt/aws/lanzados
git show "$sha:model/alt/aws/r9_e2e.sh" | sed -e "s#__CHISPA_S3__#s3://chispa-pruebas-cuenta-aws/r9-e2e-$sha#" -e "s#__N_FILAS__#$n#" -e "s#__GO_SHA256__#$GO_SHA256#" > "$ud"
bash -n "$ud"
grep -n '__[A-Z0-9_]*__' "$ud" | grep -v 'GO_SHA256#__' && { echo "quedan placeholders sin sustituir" >&2; exit 1; }
wc -c "$ud"; (cd "$sal" && shasum -a 256 codigo.tgz filas.jsonl; ls -l codigo.tgz filas.jsonl)
echo "sube a s3://chispa-pruebas-cuenta-aws/r9-e2e-$sha/: codigo.tgz y filas.jsonl; lanza $ud"
