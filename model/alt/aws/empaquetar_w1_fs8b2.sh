#!/usr/bin/env bash
# W1d: empaqueta la fase B2 (g5.xlarge, tope 45 min = 0,75 USD): escribe model/alt/aws/lanzados/ud-w1-fs8b2-SHA.sh y SAL/codigo.tgz. NO sube ni lanza nada.
#   codigo.tgz   de `git archive SHA`: solo el Python que usa B2 (sin server/, sin Go, sin dev.tgz: las entradas salen de faseA.tar y resultados.tgz de B, comprobados por sha256).
# El ud fija por sha256: el criterio (CRIT_SHA256, congelado por el orquestador y distinto de nada nuevo), el resultados.tgz de la fase B (RESB_SHA256) y faseA.tar (FASEA_SHA256).
# El Python de la medida (llm_logprobs, lineas_base_v2, lineas_base_v2_pred, w1_demos, w1_medir, el criterio) tiene que ser BYTE A BYTE el de la fase B (BASE_B, por defecto ad2b70e): si difiere, no se empaqueta.
# Se sube a mano, por nombre exacto: codigo.tgz a s3://chispa-pruebas-cuenta-aws/w1-fs8b2-SHA/ (B2 escribe ahí log.txt, listo y resultados.tgz); lee s3://.../w1-fs8b-ad2b70e/resultados.tgz y s3://.../w1-fs8a-ad2b70e/faseA.{tar,json}.
# CHISPA_UD_DIR cambia el directorio del ud (pruebas en seco); CHISPA_PERMITIR_CRIT_SIN_CONGELAR=1 solo en la prueba en seco; S3B_PREFIJO y S3A_PREFIJO (nombres de carpeta) cambian de dónde lee.
# Uso: CRIT_SHA256=343365bd...498e2 RESB_SHA256=b95e87a3...6d9 FASEA_SHA256=c0e7d227...ad6 bash model/alt/aws/empaquetar_w1_fs8b2.sh SHA DIR_SALIDA
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=${1:?SHA}; sal=${2:?DIR_SALIDA}
BASE_B=${BASE_B:-ad2b70e}
hex() { [[ ${!1:-} =~ ^[0-9a-f]{64}$ ]] || { echo "$1 falta o no es un sha256 en hex" >&2; exit 1; }; }
hex CRIT_SHA256; hex RESB_SHA256; hex FASEA_SHA256
mkdir -p "$sal"; sal=$(cd "$sal" && pwd)
MODELALT="model/alt/__init__.py model/alt/crit-w1-fs8-e2e.json model/alt/llm_logprobs.py model/alt/llm_jevbench.py model/alt/lineas_base_v2.py model/alt/fusion_knn_l2.py model/alt/lineas_base_v2_pred.py
  model/alt/w1_filas.py model/alt/w1_demos.py model/alt/w1_medir.py model/alt/w1_juntar.py model/alt/aws/w1_fs8b2.sh model/alt/aws/w1_fs8b2_fases.sh"
for f in $MODELALT model/__init__.py; do git cat-file -e "$sha:$f" || { echo "$sha no trae $f" >&2; exit 1; }; done
crit=$(git show "$sha:model/alt/crit-w1-fs8-e2e.json" | shasum -a 256 | cut -d' ' -f1)
[ "$crit" = "$CRIT_SHA256" ] || { echo "el criterio de $sha tiene sha256 $crit y no el congelado ($CRIT_SHA256)" >&2; exit 1; }
if [ "${CHISPA_PERMITIR_CRIT_SIN_CONGELAR:-0}" != 1 ]; then
  git show "$sha:model/alt/crit-w1-fs8-e2e.json" | python3 -I -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("congelado_antes_de_medir") is True else 1)' \
    || { echo "el criterio de $sha no está congelado (congelado_antes_de_medir: true)" >&2; exit 1; }
fi
for f in model/alt/llm_logprobs.py model/alt/lineas_base_v2.py model/alt/lineas_base_v2_pred.py model/alt/w1_demos.py model/alt/w1_medir.py model/alt/crit-w1-fs8-e2e.json; do
  [ "$(git rev-parse "$sha:$f")" = "$(git rev-parse "$BASE_B:$f")" ] || { echo "$f de $sha no es el de la fase B ($BASE_B): B2 mediría con otro código" >&2; exit 1; }
done
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
git archive "$sha" model/__init__.py $MODELALT | tar -x -C "$t"
find "$t" -name '._*' -delete
export COPYFILE_DISABLE=1
rm -f "$sal/codigo.tgz"
( cd "$t" && { tar --no-xattrs -czf "$sal/codigo.tgz" model 2>/dev/null || tar -czf "$sal/codigo.tgz" model; } )
tar -tzf "$sal/codigo.tgz" | grep -q '\._' && { echo "codigo.tgz trae ._*" >&2; exit 1; }
shf() { shasum -a 256 "$1" | cut -d' ' -f1; }
udd=${CHISPA_UD_DIR:-model/alt/aws/lanzados}; mkdir -p "$udd"
ud=$udd/ud-w1-fs8b2-$sha.sh; P=s3://chispa-pruebas-cuenta-aws
git show "$sha:model/alt/aws/w1_fs8b2.sh" | sed -e "s#__CHISPA_S3__#$P/w1-fs8b2-$sha#" -e "s#__CHISPA_S3_A__#$P/${S3A_PREFIJO:-w1-fs8a-$BASE_B}#" -e "s#__CHISPA_S3_B__#$P/${S3B_PREFIJO:-w1-fs8b-$BASE_B}#" \
  -e "s#__CRIT_SHA__#$crit#" -e "s#__COD_SHA256__#$(shf "$sal/codigo.tgz")#" -e "s#__RESB_SHA256__#$RESB_SHA256#" -e "s#__FASEA_SHA256__#$FASEA_SHA256#" > "$ud"
bash -n "$ud"
grep -n '__[A-Z0-9_]*__' "$ud" | grep -v 'S3_LOCAL\|S3A_LOCAL\|S3B_LOCAL' && { echo "quedan placeholders sin sustituir en $ud" >&2; exit 1; }
[ "$(wc -c < "$ud")" -lt 15500 ] || { echo "$ud pesa $(wc -c < "$ud") bytes: el user-data de EC2 admite 16384 como máximo" >&2; exit 1; }
bash -n <(git show "$sha:model/alt/aws/w1_fs8b2_fases.sh")
echo "ud B2: $ud ($(wc -c < "$ud") bytes, sha256 $(shf "$ud"))"
(cd "$sal" && ls -l codigo.tgz && shasum -a 256 codigo.tgz)
echo "criterio sha256 $crit; resultados.tgz de B $RESB_SHA256; faseA.tar $FASEA_SHA256; Python igual al de $BASE_B"
echo "sube codigo.tgz a $P/w1-fs8b2-$sha/ ; B2 (g5.xlarge, volumen >= 100 GB, tope 45 min = 45/60 x 1,006 = 0,75 USD) lee de ahí y de la fase B y A"
